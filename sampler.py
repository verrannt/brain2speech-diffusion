import os
from typing import Optional, Tuple, Union

from omegaconf import DictConfig
from scipy.io.wavfile import write as wavwrite
import torch
from torch import Tensor
from tqdm import tqdm

from dataloaders.conditional_loaders import ECOGLoader, get_word_from_filepath
from dataloaders import ClassConditionalLoader
from models import construct_model
import utils


class Sampler:
    """
    Sampler to generate audio samples from noise using a diffusion process, with and without conditional input.

    Parameters
    ----------
    rank:
        Rank of the GPU the class is instantiated on.
    diffusion_cfg:
        Config options for the diffusion process
    dataset_cfg:
        Config options of the data to be generated (e.g. sampling rate)
    name:
        Name of the experiment for saving outputs to disk, and maybe loading the model if none is given
    n_samples:
        How many outputs to generate
    batch_size:
        How many outputs to generate at once. If `None`, `n_samples` will be used.
    conditional_type:
        The type of conditional signal, in case a conditional signal is used (ignored otherwise). Currently supported
        are only `'brain'` (for ECoG Tensor input) or `'class'` (for class-conditional input).
    conditional_signal:
        The actual conditional signal to input to the model, in case that it is a conditional model. If
        `conditional_type=='brain'`, must be a filepath pointing to an appropriate ECoG file (`.npy`) on disk. If
        `conditional_type=='class'`, can simply be the class name, e.g. 'dag'. If the model is unconditional, must be
        `None`.

    Raises
    ------
    AssertionError:
        If `conditional_type` is neither `'brain'` nor `'class'`
    AssertionError:
        If `n_samples` is not a multiple of `batch_size`, i.e. the desired number of samples cannot be generated in
        appropriately sized batches.
    """
    def __init__(
        self,
        rank: int,
        diffusion_cfg: DictConfig,
        dataset_cfg: DictConfig,
        name: str,
        n_samples: int = 1,
        batch_size: int = None,
        conditional_signal: Optional[str] = None,
        conditional_type: str = "brain",
        **kwargs,
    ) -> None:
        self.rank = rank
        self.diffusion_cfg = diffusion_cfg
        self.dataset_cfg = dataset_cfg
        self.name = name
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.conditional_signal = conditional_signal
        assert conditional_type == "brain" or conditional_type == "class", \
            "The type of conditional input can only be 'brain' or 'class'"
        self.conditional_type = conditional_type

        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        if batch_size is None:
            self.batch_size = self.n_samples
        assert self.n_samples % self.batch_size == 0

        # Map diffusion hyperparameters to GPU
        self.diffusion_hyperparams = utils.calc_diffusion_hyperparams(**self.diffusion_cfg)

    @torch.no_grad()
    def run(self, ckpt_epoch: Union[str, int], model_cfg: DictConfig, model: torch.nn.Module = None) -> None:
        print("\nGenerating:")
    
        # Get output directory for waveforms for this run
        experiment_name, waveform_directory = utils.create_output_directory(
            self.name, model_cfg, self.diffusion_cfg, self.dataset_cfg, 'waveforms')


        # Load model if not given
        if model is None:
            model, ckpt_epoch = self.load_model(experiment_name, model_cfg, ckpt_epoch, self.conditional_signal is None)

        # Add checkpoint number to output directory
        waveform_directory = os.path.join(waveform_directory, str(ckpt_epoch))
        if self.rank == 0:
            if not os.path.isdir(waveform_directory):
                os.makedirs(waveform_directory)
                os.chmod(waveform_directory, 0o775)

        # Compute audio length in no. of frames from segment length in milliseconds and sampling rate
        audio_length = int(self.dataset_cfg.segment_length * self.dataset_cfg.sampling_rate / 1000)
        
        # Load conditional input
        if model_cfg.unconditional and self.conditional_signal is not None:
            raise ValueError('Model configured to be unconditional, but a conditional input is specified')
        if not model_cfg.unconditional and self.conditional_signal is None:
            raise ValueError('Model configured to be conditional, but no conditional input is specified')
        conditional_input = self.load_conditional_input()

        # Print information about inference
        print(f'Audio length: {audio_length}, Samples: {self.n_samples}, '
              f'Batch size: {self.batch_size}, Reverse steps (T): {self.diffusion_hyperparams["T"]}')

        # Run Inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        generated_audio = []
        
        for _ in range(self.n_samples // self.batch_size):
            _audio = self.sampling(
                model,
                audio_length,
                condition=conditional_input,
            )
            generated_audio.append(_audio)
        
        generated_audio = torch.cat(generated_audio, dim=0)

        end.record()
        torch.cuda.synchronize()

        print('Generated {} samples with shape {} in {} seconds'.format(
            self.n_samples,
            list(generated_audio.shape),
            int(start.elapsed_time(end)/1000)
        ))

        # Save generated audio as WAV files to disk
        
        # In case of conditional sampling, get the word from filepath, but do not remove numbering: numbering is only
        # there in the brain-conditional case and is relevant for telling us which ECoG recording was used.
        if self.conditional_signal is not None:
            conditional_signal_name = get_word_from_filepath(self.conditional_signal, uses_numbering=False) + '_'
        else:
            conditional_signal_name = ''
        for i in range(self.n_samples):
            outfile = f'{conditional_signal_name}{self.n_samples*self.rank + i}.wav'
            wavwrite(
                os.path.join(waveform_directory, outfile),
                self.dataset_cfg.sampling_rate,
                generated_audio[i].squeeze().cpu().numpy()
            )

        print(f'Saved generated samples at {waveform_directory}')
        
        return generated_audio

    def load_model(
        self, 
        experiment_name: str, 
        model_cfg: DictConfig, 
        ckpt_epoch: Union[int, str], 
        unconditional: bool
    ) -> Tuple[torch.nn.Module, int]:
        """
        Load a trained model from disk.

        Parameters
        ----------
        experiment_name:
            Name of the training experiment, which is the path the model state was saved under during training. Must
            contain a directory 'checkpoint' that contains the model checkpoints.
        model_cfg:
            Configuration options of the model needed to construct the model.
        ckpt_epoch:
            Which epoch checkpoint to load. If `'max'`, finds the latest checkpoint. Else if `int`, will look for the 
            exact epoch.
        unconditional:
            Whether the model is a conditional or unconditional model.

        Returns
        -------
        The loaded model, and the epoch that was loaded (only different from the input parameter if the latter was 
        `'max'` and had to be resolved)

        Raises
        ------
        Exception:
            If no valid model could be found or loading failed.
        """


        model = construct_model(model_cfg).cuda()
        utils.print_size(model)
        model.eval()

        print(f'Loading checkpoint at epoch {ckpt_epoch}')
        ckpt_path = os.path.join('exp', experiment_name, 'checkpoint')
        if ckpt_epoch == 'max':
            ckpt_epoch = utils.find_max_epoch(ckpt_path)
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

        return model, ckpt_epoch

    def load_conditional_input(self) -> Optional[Tensor]:
        """
        Load conditional input, either by reading an ECoG file from disk or getting the token for a word class. This 
        depends on whether `conditional_type` is `brain` or `class`, respectively. If no conditional signal is given,
        simply return `None`.

        Returns
        -------
        Either the conditional signal as CUDA Tensor or `None`.

        Raises
        ------
        ValueError, if `conditional_type` is neither `brain` nor `class`.
        """
        if self.conditional_signal is None:
            return None
        else:
            # For class-conditional sampling, get the token from the provided word class
            if self.conditional_type == "class":
                conditional_loader = ClassConditionalLoader(
                    os.path.join(self.dataset_cfg.data_base_dir, 'HP_VariaNTS_intersection.txt'))
                return conditional_loader(self.conditional_signal).unsqueeze(0).cuda()
            
            # For brain-conditional sampling, load the provided file directly
            elif self.conditional_type == "brain":
                ecog_length = int(self.dataset_cfg.segment_length * self.dataset_cfg.sampling_rate_ecog / 1000)
                return ECOGLoader.process_ecog(self.conditional_signal, ecog_length).unsqueeze(0).cuda()
            
            else:
                raise ValueError(f"Unknown conditional type: {self.conditional_type}")

    def sampling(self, model: torch.nn.Module, audio_length: int, condition: Optional[Tensor] = None) -> Tensor:
        """
        Perform the complete reverse process p(x_0|x_T)

        Parameters
        ----------
        model:
            The model
        audio_length:
            Length of the audio output tensor to be generated.
        condition:
            Conditional input to the model, if it is a conditional model. Needs to be `None` for unconditional models.

        Returns
        -------
        The generated audio(s) as Tensor or shape  `[batchsize, 1, audio_length]`
        """

        size = (self.batch_size, 1, audio_length)

        _dh = self.diffusion_hyperparams
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