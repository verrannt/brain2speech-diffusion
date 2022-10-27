from functools import partial
import multiprocessing as mp
import os
import time
from typing import Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import wandb

from generate import generate
from dataloaders import dataloader
from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from models import construct_model
from utils import MaskedMSELoss, find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory


def distributed_train(
    rank: int, 
    num_gpus: int, 
    group_name: str, 
    cfg: DictConfig
) -> None:
    """
    Train DiffWave in the distributed setting, by appropriately initializing distributed training and calling
    the train function, given configuration settings.

    Parameters
    ---
    rank: Rank of the GPU this function instance is called on
    num_gpus: Total number of GPUs available in this run
    group_name: Identifier of the process group
    cfg: configuration options defined in the Hydra config file
    """

    # Initialize logger
    if rank == 0 and cfg.wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        
        # Use the name of the training run also as the run name in W&B
        wandb_cfg['name'] = cfg['train']['name']
        
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, 
        num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )

def train(
    rank: int,
    num_gpus: int,
    diffusion_cfg: DictConfig,
    model_cfg: DictConfig,
    dataset_cfg: DictConfig,
    generate_cfg: DictConfig,
    ckpt_epoch: Union[int, str],
    n_epochs: int,
    epochs_per_ckpt: int,
    iters_per_logging: int,
    learning_rate: float,
    batch_size_per_gpu: int,
    name: Union[str, None],
) -> None:
    """
    Train DiffWave on a single GPU of given rank, with given configuration settings.

    Parameters
    ---
    rank:                   rank (ID) of the GPU this function is executed on
    num_gpus:               total number of GPUs available in this run
    ckpt_epoch:             the pretrained checkpoint to be loaded;
                            automatically selects the maximum iteration if 'max' is selected
    n_epochs:               number of epochs to train
    epochs_per_ckpt:        how often to save a model checkpoint
    iters_per_logging:      number of iterations to save training log and compute validation loss
    learning_rate:          learning rate
    batch_size_per_gpu:     batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    name:                   prefix in front of experiment name
    """

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, dataset_cfg, 'checkpoint')

    # Map diffusion hyperparameters to GPU
    # Gives dictionary of all diffusion hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_cfg, 
        fast=False
    )

    # Initialize data loader. Val and test loaders might be None if not specified
    trainloader, valloader, testloader = dataloader(
        dataset_cfg, 
        batch_size = batch_size_per_gpu, 
        num_gpus = num_gpus, 
        unconditional = model_cfg.unconditional
    )
    print('Data loaded')

    model = construct_model(model_cfg).cuda()
    print_size(model, verbose=False)

    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = MaskedMSELoss()

    # Load checkpoint
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(checkpoint_directory)
    if ckpt_epoch >= 0:
        try:
            # Load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_epoch))
            checkpoint = torch.load(model_path, map_location='cpu')

            # Feed model dict and optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at epoch {}'.format(ckpt_epoch))
        except:
            print(f"Model checkpoint found at epoch {ckpt_epoch}, but was not successfully loaded - training from scratch.")
            ckpt_epoch = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_epoch = -1

    # TRAINING

    # Continue iteration from checkpoint epoch number
    if ckpt_epoch == -1:
        start_epoch = 1
    else:
        start_epoch = ckpt_epoch + 1

    for epoch in range(start_epoch, n_epochs+start_epoch):
        print(f"\n{'-'*100}\nEPOCH {epoch}/{start_epoch+n_epochs-1}")
        epoch_loss = 0.
        print()
        for i, data in enumerate(tqdm(trainloader, desc='Training', ncols=100)):
            if model_cfg["unconditional"]:
                audio, _, _, mask = data
                audio = audio.cuda()
                if mask is not None:
                    mask = mask.cuda()
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()
                mask = None

            optimizer.zero_grad()
            loss = compute_loss(
                model, 
                loss_fn, 
                audio, 
                diffusion_hyperparams, 
                mel_spec=mel_spectrogram, 
                mask=mask,
            )

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            if i % iters_per_logging == 0 and i != 0 and rank == 0:
                wandb.log({
                    'train/loss': reduced_loss,
                    'train/log_loss': np.log(reduced_loss),
                    'epoch': epoch,
                }, commit=False) # commit only at end of epoch
                print(f"\nStep {i} Train Loss: {reduced_loss}")

        # Log average epoch loss
        if rank == 0:
            epoch_loss /= len(trainloader)
            wandb.log({
                'train/loss_epoch': epoch_loss, 
                'train/log_loss_epoch': np.log(epoch_loss),
                'epoch': epoch,
            })
            print(f"Loss: {epoch_loss}")

        # Save checkpoint
        if epoch % epochs_per_ckpt == 0 and rank == 0:
            checkpoint_name = f'{epoch}.pkl'
            checkpoint_path_full = os.path.join(checkpoint_directory, checkpoint_name)

            # Save to local dir
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                checkpoint_path_full,
            )
            
            # Save to W&B
            artifact = wandb.Artifact(wandb.run.name, type="model")
            artifact.add_file(checkpoint_path_full, name=f"epoch_{epoch}")
            wandb.log_artifact(artifact)
            
            print('Created model checkpoint')

            # Generate samples
            if not model_cfg["unconditional"]: 
                assert generate_cfg.mel_name is not None

            generate_cfg["ckpt_epoch"] = epoch
            
            samples = generate( 
                rank,
                diffusion_cfg, 
                model_cfg, 
                dataset_cfg,
                name=name,
                **generate_cfg,
            )

            samples = [
                wandb.Audio(
                    sample.squeeze().cpu(), 
                    sample_rate = dataset_cfg['sampling_rate']
                ) for sample in samples
            ]

            wandb.log(
                {'inference/audio': samples, 'epoch': epoch},
            )
        

        # VALIDATION

        if valloader and rank == 0:
            print()
            val_loss = 0
            for data in tqdm(valloader, desc='Validating', ncols=100):
                if model_cfg["unconditional"]:
                    audio, _, _, mask = data
                    audio = audio.cuda()
                    if mask is not None:
                        mask = mask.cuda()
                    mel_spectrogram = None
                else:
                    mel_spectrogram, audio = data
                    mel_spectrogram = mel_spectrogram.cuda()
                    audio = audio.cuda()
                    mask = None

                loss_value = compute_loss(
                    model, 
                    loss_fn, 
                    audio, 
                    diffusion_hyperparams, 
                    mel_spec=mel_spectrogram, 
                    mask=mask,
                ).item()

                # Note that we do not call `reduce_tensor` on the loss here like
                # we do in the train loop, since this validation loop only
                # runs on one GPU.

                val_loss += loss_value

            val_loss /= len(valloader)
            wandb.log({
                'val/loss': val_loss,
                'val/log_loss': np.log(val_loss),
                'epoch': epoch,
            })
            print(f"Loss: {val_loss}")


    # TESTING

    if testloader and rank == 0:
        print("\n" + "-"*100 + "\n")
        test_loss = 0.
        for data in tqdm(testloader, desc='Testing', ncols=100):
            if model_cfg["unconditional"]:
                audio, _, _, mask = data
                audio = audio.cuda()
                if mask is not None:
                    mask = mask.cuda()
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()
                mask = None

            loss_value = compute_loss(
                model, 
                loss_fn, 
                audio, 
                diffusion_hyperparams, 
                mel_spec=mel_spectrogram, 
                mask=mask,
            ).item()

            test_loss += loss_value

        test_loss /= len(testloader)
        wandb.run.summary["test/loss"] = test_loss
        wandb.run.summary["test/log_loss"] = np.log(test_loss)
        print(f"Loss: {test_loss}")


    # Close logger
    if rank == 0:
        wandb.finish()

def compute_loss(model, loss_fn, audio, diffusion_hyperparams, mel_spec=None, mask=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters
    ---
    model (torch network):          the wavenet model
    loss_fn (torch loss function):  the loss function
    audio (torch.tensor):           training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns
    ---
    Output of the loss function
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    
    # randomly sample diffusion steps from 1~T
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()
    z = torch.normal(0, 1, size=audio.shape).cuda()

    # compute x_t from q(x_t|x_0)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
    
    # predict epsilon according to epsilon_theta
    epsilon_theta = model((transformed_X, diffusion_steps.view(B,1),), mel_spec=mel_spec)

    return loss_fn(epsilon_theta, z, mask)


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Allow writing keys
    OmegaConf.set_struct(cfg, False)

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
