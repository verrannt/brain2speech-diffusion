from functools import partial
import multiprocessing as mp
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.io.wavfile import write as wavwrite
import torch
from torch import Tensor
from tqdm import tqdm

import dataloaders.utils as data_utils
from dataloaders.conditional_loaders import EEGLoader
from dataloaders import ClassConditionalLoader
from models import construct_model
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory


@torch.no_grad()
def generate(
    rank,
    model,
    model_cfg:DictConfig,
    diffusion_cfg:DictConfig,
    dataset_cfg:DictConfig,
    name:str=None,
    ckpt_epoch:str="max",
    n_samples:int=1,
    batch_size:int=None,
    conditional_signal:str=None,
    conditional_type:str="brain", # "brain" or "class"
):
    print("\nGenerating:")
    
    if rank is not None:
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if batch_size is None:
        batch_size = n_samples
    assert n_samples % batch_size == 0
    
    # Get output directory for waveforms for this run
    local_path, output_directory = local_directory(name, model_cfg, diffusion_cfg, dataset_cfg, 'waveforms')

    # Map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)

    # Load model if not given
    if model is None:
        model = load_model(model_cfg, local_path, conditional_signal is None)

    # Add checkpoint number to output directory
    output_directory = os.path.join(output_directory, str(ckpt_epoch))
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)

    # Compute audio length in no. of frames from segment length in milliseconds and sampling rate
    audio_length = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate / 1000)
    
    # Load conditional input (`None` if none given)
    conditional_input = load_conditional_input(conditional_signal, conditional_type, dataset_cfg)

    # Print information about inference
    print(f'Audio length: {audio_length}, Samples: {n_samples}, Batch size: {batch_size}, Reverse steps (T): {diffusion_hyperparams["T"]}')

    # Run Inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_audio = []
    
    for _ in range(n_samples // batch_size):
        _audio = sampling(
            model,
            (batch_size, 1, audio_length),
            diffusion_hyperparams,
            condition=conditional_input,
        )
        generated_audio.append(_audio)
    
    generated_audio = torch.cat(generated_audio, dim=0)

    end.record()
    torch.cuda.synchronize()

    print('Generated {} samples with shape {} in {} seconds'.format(
        n_samples,
        list(generated_audio.shape),
        int(start.elapsed_time(end)/1000)
    ))

    # Save generated audio as WAV files to disk
    for i in range(n_samples):
        outfile = f'epoch{ckpt_epoch}_{n_samples*rank + i}.wav'
        wavwrite(
            os.path.join(output_directory, outfile),
            dataset_cfg.sampling_rate,
            generated_audio[i].squeeze().cpu().numpy()
        )

    print(f'Saved generated samples at {output_directory}')
    
    return generated_audio


def sampling(model, size, diffusion_hyperparams, condition=None):
    """
    Perform the complete reverse process p(x_0|x_T)

    Parameters:
    model (torch network):          the model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    # Sample x_T from isotropic standard Gaussian
    x = torch.normal(0, 1, size=size).cuda()
    
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1), desc='Sampling', ncols=100):
            # Broadcast timestep to batchsize shape
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()

            # Predict added noise
            epsilon_theta = model((x, diffusion_steps), condition)
            
            # Update x_{t-1} to mu_theta(x_t)
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  
            if t > 0:
                # Add the variance term to x_{t-1}
                x = x + Sigma[t] * torch.normal(0, 1, size=size).cuda()  
    return x


def load_model(model_cfg, local_path, unconditional: bool):
    model = construct_model(model_cfg).cuda()
    print_size(model)
    model.eval()

    print(f'Loading checkpoint at epoch {ckpt_epoch}')
    ckpt_path = os.path.join('exp', local_path, 'checkpoint')
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path)
    ckpt_epoch = int(ckpt_epoch)

    try:
        model_path = os.path.join(ckpt_path, f'{ckpt_epoch}.pkl')
        checkpoint = torch.load(model_path, map_location='cpu')

        if unconditional:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(
                checkpoint['model_state_dict'],
                checkpoint.get('conditioner_state_dict', None),
            )

        print(f'Successfully loaded model at epoch {ckpt_epoch}')
    except:
        raise Exception(f'No valid model found for epoch {ckpt_epoch} at path {ckpt_path}')

    return model


def load_conditional_input(conditional_signal: str, conditional_type: str, dataset_cfg: DictConfig) -> Tensor:
    """
    Load conditional input, either by reading an EEG file from disk or getting the token for a word class. This depends
    on whether `conditional_type` is `brain` or `class`, respectively.

    Params
    ------
    conditional_signal :
        Either path to EEG file on disk (in case of `conditional_type==brain`), OR word class (in case of 
        `conditional_type==class`)
    conditional_type : 
        The type of conditional signal to load.
    dataset_cfg :
        Dataset configuration, needed to get desired EEG length (if `conditional_type==brain`) or data dir for words
        used for class-conditional sampling (if `conditional_type==class`).

    Raises
    ------
    ValueError, if `conditional_type` is neither `brain` nor `class`.

    """
    if conditional_signal is None:
        return None
    else:
        # For class-conditional sampling, get the token from the provided word class
        if conditional_type == "class":
            conditional_loader = ClassConditionalLoader(
                os.path.join(dataset_cfg.data_base_dir, 'HP_VariaNTS_intersection.txt'))
            return conditional_loader(conditional_signal).unsqueeze(0).cuda()
        
        # For brain-conditional sampling, load the provided file directly
        elif conditional_type == "brain":
            eeg_length = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate_eeg / 1000)
            return EEGLoader.process_eeg(conditional_signal, eeg_length).unsqueeze(0).cuda()
        
        else:
            raise ValueError(f"Unknown conditional type: {conditional_type}")


def load_eeg_file(file_path: str, length: int) -> Tensor:
    """ 
    Load an EEG input file from Numpy array stored on disk and post-process it (normalization and length fixing)
    """
    x = torch.from_numpy(np.load(file_path)).float()
    x = data_utils.standardize_eeg(x)
    x = data_utils.fix_length_3d(x, length)
    return x


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=generate_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
