import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
from typing import Tuple

import dataloaders.utils as utils


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
        assert len(self._eeg_files) == len(self._audio_files), \
            'There is an unequal number of brain and speech data files'

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

        self.segment_length_audio = int(segment_length * sampling_rate_audio / 1000) # TODO is this correct?
        self.segment_length_eeg = int(segment_length * sampling_rate_eeg / 1000)

        self.standardize_eeg = standardize_eeg

    def __getitem__(self, n: int):
        audio_path, eeg_path = self._files[n]
        return self.load_data(audio_path, eeg_path)

    def __len__(self) -> int:
        return len(self._files)
    
    def load_data(self, audio_path: str, eeg_path: str) -> Tuple[Tensor, int, Tensor, Tensor]:
        waveform, sample_rate = torchaudio.load(audio_path)

        eeg = torch.from_numpy(np.load(eeg_path)).float()

        if self.standardize_eeg:
            eeg = utils.standardize_eeg(eeg)

        # Norm waveform to designated length and get padding mask
        waveform, mask = utils.fix_length_1d(waveform, self.segment_length_audio)

        # Norm eeg to designated length
        eeg = utils.fix_length_3d(eeg, self.segment_length_eeg)

        return (waveform, sample_rate, eeg, mask)
