import os
from typing import Union, Tuple, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor
from tqdm import tqdm
import wandb

from dataloaders import dataloader
from utils.distributed_util import apply_gradient_allreduce, reduce_tensor
from models import construct_model
from sampler import Sampler
import utils.training as train_utils


class Learner():
    """
    Learner object that trains a model on a single GPU, given configuration options. Can be used in distributed training
    as well, as long as multiprocessing is handled outside of an instance of this object. 

    Parameters
    ----------
    cfg:
        The configuration options as defined in the Hydra config file
    num_gpus:
        The total number of GPUs used for the training run. This is needed for the correct initialization of the
        data loaders.
    rank:
        Rank (ID) of the GPU this function is executed on
    ckpt_epoch:
        Pretrained checkpoint to be loaded; automatically selects the maximum iteration if 'max' is selected
    n_epochs:
        Number of epochs to train
    epochs_per_ckpt:
        How often to save a model checkpoint
    iters_per_logging:
        Number of iterations to save training log and compute validation loss
    learning_rate:
        Learning rate
    batch_size_per_gpu:
        Batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    name:
        Prefix in front of experiment name

    See also
    --------
    `train_single` and `train_distributed` in `train.py`, which utilize this object for model training.
    """

    def __init__(
        self, 
        cfg: DictConfig,
        num_gpus: int,
        rank: int,
        ckpt_epoch: Union[int, str],
        n_epochs: int,
        epochs_per_ckpt: int,
        iters_per_logging: int,
        learning_rate: float,
        batch_size_per_gpu: int,
        name: Union[str, None],
    ) -> None:
        # Create output directory if it doesn't exist
        if not os.path.isdir("exp/"):
            os.makedirs("exp/")
            os.chmod("exp/", 0o775)

        # Configs for training
        self.ckpt_epoch = ckpt_epoch
        self.n_epochs = n_epochs
        self.epochs_per_ckpt = epochs_per_ckpt
        self.iters_per_logging = iters_per_logging
        self.learning_rate = learning_rate
        self.batch_size = batch_size_per_gpu
        self.name = name

        # All other configs
        self.diffusion_cfg = cfg.diffusion
        self.model_cfg = cfg.model
        self.dataset_cfg = cfg.dataset
        self.generate_cfg = cfg.generate

        # Overwrite name for generation
        self.generate_cfg.name = self.name

        # Used to make sure that only one instance reports training metrics
        self.is_master = rank == 0

        # Since we don't collect metrics from all GPUs in the distributed setting, 
        # only the master instance should log to W&B
        if self.is_master:
            self.wandb_cfg = cfg.wandb
            # Use the name of the training run also as the run name in W&B
            self.wandb_cfg.name = self.name
            # Convert config to simple dict for logging in W&B
            self.config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Update the paths in the dataset config with the base path
        self.dataset_cfg = train_utils.prepend_data_base_dir(self.dataset_cfg)

        self.rank = rank

        self.num_gpus = num_gpus


    def train(self) -> None:
        """
        Train model on a single GPU.
        """

        # Initialize logger
        if self.is_master:
            wandb.init(**self.wandb_cfg, config=self.config_dict)

        _, checkpoint_directory = train_utils.create_output_directory(
            self.name, self.model_cfg, self.diffusion_cfg, self.dataset_cfg, 'checkpoint')

        # Map diffusion hyperparameters to GPU
        # Gives dictionary of all diffusion hyperparameters
        self.diffusion_hyperparams = train_utils.calc_diffusion_hyperparams(
            **self.diffusion_cfg, fast=False)

        self.sampler = Sampler(
            rank=self.rank,
            diffusion_cfg=self.diffusion_cfg,
            dataset_cfg=self.dataset_cfg,
            **self.generate_cfg,
        )

        # Initialize data loader. Val and test loaders might be None if not specified
        trainloader, valloader, testloader = dataloader(
            self.dataset_cfg, 
            batch_size = self.batch_size, 
            is_distributed = self.num_gpus > 1, 
            unconditional = self.model_cfg.unconditional
        )
        print('Data loaded')

        self.model = construct_model(self.model_cfg).cuda()
        train_utils.print_size(self.model, verbose=False)

        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_fn = torch.nn.MSELoss()

        # Load checkpoint
        self.load_checkpoint(checkpoint_directory)
        self.load_pretrained_generator()

        # Continue iteration from checkpoint epoch number
        if self.ckpt_epoch == -1:
            start_epoch = 1
        else:
            start_epoch = self.ckpt_epoch + 1

        # TRAINING
        for epoch in range(start_epoch, self.n_epochs+start_epoch):
            print(f"\n{'-'*100}\nEPOCH {epoch}/{start_epoch+self.n_epochs-1}")
            train_loss = 0.
            print()
            for i, data in enumerate(tqdm(trainloader, desc='Training', ncols=100, disable=not self.is_master)):
                train_step_loss = self.train_step(data)
                train_loss += train_step_loss

                # Log step loss
                if self.is_master and i % self.iters_per_logging == 0 and i != 0:
                    wandb.log({
                        'train/loss': train_step_loss,
                        'train/log_loss': np.log(train_step_loss),
                        'epoch': epoch,
                    }, commit=False) # commit only at end of epoch
                    print(f"\nStep {i} Train Loss: {train_step_loss}")

            if self.is_master:
                # Log average epoch loss
                train_loss /= len(trainloader)
                wandb.log({
                    'train/loss_epoch': train_loss, 
                    'train/log_loss_epoch': np.log(train_loss),
                    'epoch': epoch,
                })
                print(f"Loss: {train_loss}")

                # Save checkpoint
                if epoch % self.epochs_per_ckpt == 0:
                    self.save_checkpoint(epoch, checkpoint_directory)
                    self.generate_samples(epoch)

                # VALIDATION
                if valloader is not None:
                    print()
                    val_loss = 0.
                    for data in tqdm(valloader, desc='Validating', ncols=100):
                        val_step_loss = self.val_step(data)
                        val_loss += val_step_loss

                    val_loss /= len(valloader)
                    wandb.log({
                        'val/loss': val_loss,
                        'val/log_loss': np.log(val_loss),
                        'epoch': epoch,
                    })
                    print(f"Loss: {val_loss}")

        # TESTING
        if self.is_master and testloader is not None:
            print("\n" + "-"*100 + "\n")
            test_loss = 0.
            for data in tqdm(testloader, desc='Testing', ncols=100):
                test_step_loss = self.val_step(data)
                test_loss += test_step_loss

            test_loss /= len(testloader)
            wandb.run.summary["test/loss"] = test_loss
            wandb.run.summary["test/log_loss"] = np.log(test_loss)
            print(f"Loss: {test_loss}")
            
        # Close logger
        if self.is_master:
            wandb.finish()


    def train_step(self, data: Tuple[Tensor, int, Union[Tensor, str], Tensor]):
        audio, conditional_input = self.unpack_input(data)

        self.optimizer.zero_grad()
        loss = self.compute_loss(
            audio,
            conditional_input=conditional_input,
        )

        if self.num_gpus > 1:
            loss_value = reduce_tensor(loss.data, self.num_gpus).item()
        else:
            loss_value = loss.item()
        
        loss.backward()
        self.optimizer.step()

        return loss_value


    def val_step(self, data: Tuple[Tensor, int, Union[Tensor, str], Tensor]):
        audio, conditional_input = self.unpack_input(data)

        loss_value = self.compute_loss(
            audio, 
            conditional_input=conditional_input, 
        )
        # Note that we do not call `reduce_tensor` on the loss here like we do 
        # in the train step, since validation only ever runs on rank 0 GPU.
        return loss_value.item()


    def unpack_input(
        self, 
        data: Tuple[Tensor, int, Union[Tensor, str], str]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Unpack a tuple of different data obtained from a dataloader, and move the unpacked data to GPU.
        
        Parameters
        ----------
        data:
            `Tuple` consisting of audio data (`Tensor`), its sampling rate (`int`), the conditional input (either 
            `Tensor` if given, or empty `str` in the unconditional setting), and the filename (`str`).

        Returns
        -------
        Tuple of the unpacked data as `Tensor`s on GPU, but without the sampling rate.
        """
        audio, _, conditional_input, _ = data
        
        audio = audio.cuda()
        
        if self.model_cfg.unconditional:
            conditional_input = None
        else:
            conditional_input = conditional_input.cuda()
        
        return audio, conditional_input


    def load_checkpoint(self, checkpoint_directory):
        """
        Try loading a checkpoint from `checkpoint_directory` using the epoch specified in `self.ckpt_epoch`. If
        `'max'`, will try to find the highest avaialable epoch, else the one specified.

        In case a checkpoint is found but cannot be loaded successfully, the model shall be trained from scratch.

        Parameters
        ----------
        checkpoint_directory:
            Path to the checkpoint directory on disk, in which the pickled state dictionaries are to be found.
        """
        if self.ckpt_epoch == 'max':
            self.ckpt_epoch = train_utils.find_max_epoch(checkpoint_directory)
        if self.ckpt_epoch >= 0:
            try:
                # Load checkpoint file
                model_path = os.path.join(checkpoint_directory, f'{self.ckpt_epoch}.pkl')
                checkpoint = torch.load(model_path, map_location='cpu')

                if self.model_cfg.unconditional:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(
                        checkpoint['model_state_dict'],
                        checkpoint['conditioner_state_dict'],
                    )
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # HACK to reset learning rate
                    self.optimizer.param_groups[0]['lr'] = self.learning_rate

                print('Successfully loaded model at epoch {}'.format(self.ckpt_epoch))
            except:
                print(f"Model checkpoint found at epoch {self.ckpt_epoch}, but was not successfully loaded - training from scratch.")
                self.ckpt_epoch = -1
        else:
            print('No valid checkpoint model found - training from scratch.')
            self.ckpt_epoch = -1


    def load_pretrained_generator(self):
        """
        For fine-tuning runs, load only the generator part of the model from a pretrained checkpoint. Will also freeze
        the generator part if specified in the model config.
        """
        # If provided, load the weights from pretraining of the generator
        if self.model_cfg.get("pretrained_generator", False):
            if self.model_cfg.unconditional:
                print("Pretrained generator assigned, but will not be loaded since this is an unconditional model.")
            else:
                if self.ckpt_epoch != -1:
                    # If model is training from scratch, we only load the pretrained generator if the model is
                    # also set to be frozen, because otherwise we would overwrite the learned parameters
                    if self.model_cfg.get("freeze_generator", False):
                        checkpoint = torch.load(self.model_cfg.pretrained_generator, map_location='cpu')
                        self.model.load_pretrained_generator(
                            checkpoint, freeze=True)
                    else:
                        print("Pretrained generator assigned, but will not be loaded since model has been loaded "
                        "from checkpoint and is not set to be frozen.")
                else:
                    checkpoint = torch.load(self.model_cfg.pretrained_generator, map_location='cpu')
                    self.model.load_pretrained_generator(
                        checkpoint, freeze=self.model_cfg.get("freeze_generator", False))


    def save_checkpoint(self, epoch: int, checkpoint_directory: str):
        """
        Save a model checkpoint at `epoch` both to disk as well as to W&B as a model artifact.

        Parameters
        ----------
        epoch:
            The current epoch in training, which should be used for the save name.
        checkpoint_directory:
            The path on disk where to save the model checkpoint at.
        """

        checkpoint_name = f'{epoch}.pkl'
        checkpoint_path_full = os.path.join(checkpoint_directory, checkpoint_name)

        # Save to local dir
        save_dict = {
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        if self.model_cfg.unconditional:
            save_dict['model_state_dict'] = self.model.state_dict()
        else:
            # In the conditional setting, we get two model states from the two sub-models
            save_dict['model_state_dict'] = self.model.generator_state_dict()
            save_dict['conditioner_state_dict'] = self.model.encoder_state_dict()
        
        torch.save(
            save_dict,
            checkpoint_path_full,
        )
        
        # Save to W&B
        artifact = wandb.Artifact(wandb.run.name, type="model")
        artifact.add_file(checkpoint_path_full, name=f"epoch_{epoch}")
        wandb.log_artifact(artifact)
        
        print('Created model checkpoint')


    def generate_samples(self, epoch: int) -> None:
        """
        Run generation (inference) using the current model state, and save the results to disk as well as to W&B.

        Parameters
        ----------
        epoch:
            The current epoch in training, which should be used for the save name.
        """
        # Generate samples        
        samples = self.sampler.run( 
            epoch,
            self.model_cfg,
            self.model,
        )

        # Log generated samples to W&B
        samples = [
            wandb.Audio(
                sample.squeeze().cpu(), 
                sample_rate = self.dataset_cfg.sampling_rate
            ) for sample in samples
        ]
        wandb.log(
            {'inference/audio': samples, 'epoch': epoch},
        )


    def compute_loss(self, audio: Tensor, conditional_input: Optional[Tensor] = None):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters
        ----------
        audio:
            The audio segment used as training data of shape [batchsize, 1, length of audio]
        conditional_input:
            The conditional input in case the model being trained is a conditional model. Needs to be `None` in case
            of an unconditional model.

        Returns
        -------
        Output of the loss function
        """

        _dh = self.diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

        B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
        
        # randomly sample diffusion steps from 1~T
        diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()
        z = torch.normal(0, 1, size=audio.shape).cuda()

        # compute x_t from q(x_t|x_0)
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
        
        # predict epsilon according to epsilon_theta
        epsilon_theta = self.model((transformed_X, diffusion_steps.view(B,1)), conditional_input=conditional_input)

        return self.loss_fn(epsilon_theta, z)

