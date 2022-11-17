import os

from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np
from typing import Tuple


class EEGDataset(Dataset):
    """
    Create a Dataset for audio files conditioned on EEG signals.
    """

    def __init__(
        self, 
        path: str, 
        subset: str, 
        segment_length: int = 1000, # in milliseconds
        sampling_rate_audio: int = 16000,
        sampling_rate_eeg: int = 100, 
        shuffle: bool = True,
        seed: int = None,
        standardize_eeg: bool = True,
    ):
        self._path = Path(path) / subset

        rng = np.random.default_rng(seed)

        all_files = os.listdir(self._path)
        self._eeg_files = [file for file in all_files if file.endswith('.npy')]
        self._audio_files = [file for file in all_files if file.endswith('.wav')]
        assert len(self._eeg_files) == len(self._audio_files), 'There is an unequal number of brain and speech data files'

        if shuffle:
            rng.shuffle(self._audio_files) # inplace operation
        else:
            self._audio_files = sorted(self._audio_files)
        
        self._files = []
        for file in self._audio_files:
            self._files.append((
                self._path / file,                          # audio file
                self._path / f"{file.split('.')[0]}.npy"    # eeg matrix
            ))
        
        # This might be more elegant but does not guarantee that eeg and audio files match correctly:
        # self._files = list(zip(*[self._eeg_files, self._audio_files]))

        if not segment_length:
            raise ValueError("Sample length cannot be None")
        self.segment_length = segment_length
        self.segment_length_audio = int(segment_length * sampling_rate_audio / 1000) # TODO is this correct?
        self.segment_length_eeg = int(segment_length * sampling_rate_eeg / 1000)

        self.should_standardize_eeg = standardize_eeg

    def __getitem__(self, n: int):
        audio_path, eeg_path = self._files[n]
        return self.load_data(audio_path, eeg_path)

    def __len__(self) -> int:
        return len(self._files)

    @classmethod
    def standardize_eeg(cls, tensor: Tensor) -> Tensor:
        """
        Standardize EEG input tensor to mean 0 and standard deviation 1 along the last dimension (time).
        """
        mean = torch.mean(tensor, dim=2).unsqueeze(2).expand(-1,-1, tensor.shape[2])
        std = torch.std(tensor, dim=2).unsqueeze(2).expand(-1,-1, tensor.shape[2])
        return (tensor - mean) / std

    def fix_length_audio(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Fix the length of `tensor` to `self.sample_length`. Returns both the
        altered tensor as well as a boolean mask tensor that depicts how much
        has been padded to the original tensor. This is useful for excluding
        padded regions from the loss computation.

        Expects an input `tensor` of shape `(1,n)`.


        Example
        -------
        If `sample_length == 6`, but `tensor.shape[1] == 4`, this function will
        return a tensor with shape `(1,6)`, where `tensor[:,4:6]` is filled with
        zeros, and a mask `[[True,True,True,True,False,False]]`.

        If `sample_length >= tensor.shape[1]`, this will return the tensor cut off
        at `sample_length` and a mask of only `True` values.

        """
        assert len(tensor.shape) == 2 and tensor.shape[0] == 1

        if tensor.shape[1] > self.segment_length_audio:
            # If tensor is longer than desired length, the mask is only True
            # values
            mask = torch.ones((1,self.segment_length_audio), dtype=torch.bool)

            return tensor[:, :self.segment_length_audio], mask
        
        elif tensor.shape[1] < self.segment_length_audio:
            # If tensor is shorter than desired length, the mask is True until
            # the original length of the tensor, and False afterwards
            mask = torch.zeros((1,self.segment_length_audio), dtype=torch.bool)
            mask[:,:tensor.shape[1]] = True
            
            # We pad the tensor with zero values to increase its size to the
            # desired length
            padded_tensor = torch.cat([
                tensor, 
                torch.zeros(1, self.segment_length_audio-tensor.shape[1])
            ], dim=1)

            return padded_tensor, mask
        
        else:
            # In case the tensor has already the desired length, we use a mask
            # of only True values again and can return the tensor unaltered
            mask = torch.ones((1,self.segment_length_audio), dtype=torch.bool)
            return tensor, mask

    @classmethod
    def fix_length_eeg(cls, tensor: Tensor, desired_length: int) -> Tensor:
        # Assumes the eeg matrix has dimensions: 2 (Channels), 32 (Electrodes), T (Timesteps)
        assert len(tensor.shape) == 3 and tensor.shape[0] == 2 and tensor.shape[1] == 32

        if tensor.shape[2] > desired_length:
            return tensor[:, :, :desired_length]

        elif tensor.shape[2] < desired_length:
            return torch.cat([
                tensor, 
                torch.zeros(2, 32, desired_length - tensor.shape[2])
            ], dim=2)

        else:
            return tensor
    
    def load_data(self, audio_path: str, eeg_path: str) -> Tuple[Tensor, int, Tensor, Tensor]:
        waveform, sample_rate = torchaudio.load(audio_path)

        eeg = torch.from_numpy(np.load(eeg_path)).float()

        if self.should_standardize_eeg:
            eeg = self.standardize_eeg(eeg)

        # Norm waveform to designated length and get padding mask
        waveform, mask = self.fix_length_audio(waveform)
        # Norm eeg to designated length
        eeg = self.fix_length_eeg(eeg, self.segment_length_eeg)

        return (waveform, sample_rate, eeg, mask)
