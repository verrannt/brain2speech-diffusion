# Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py
# and https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py


from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np
from typing import Tuple

import dataloaders.utils as utils


class CSVDataset(Dataset):
    """
    Create a Dataset from a .csv file list of audio files.
    """

    def __init__(
        self, 
        path: str, 
        subset: str, 
        file_base_path: str, 
        sample_length: int = 16000,
        shuffle: bool = True,
        seed: int = None,
        min_max_norm: bool = False,
        conditional_loader = None,
    ):
        """
        Read files from .csv file on disk. File must be present in `path` as
        `subset.csv`, e.g. `/data/user/train.csv`.

        Parameters
        ----------
        path : 
            Path to the directory on disk where the .csv file is stored
        
        subset : 
            One of "train", "val", "test". Which subset of the data to 
            load. The chosen subset must be present as "<subset>.csv" in the
            `path` given as argument, e.g. "train.csv".
        
        file_base_path : 
            If given, this path is prepended to every filename in
            the loaded .csv file.
        
        sample_length : 
            Desired length of audio sequence (in sampled points). 
            Any files shorter will be padded, any files longer will be cut to
            this length.
        
        shuffle : 
            Whether to shuffle the files read from the .csv file
        
        seed : 
            Seed for the random number generator used for shuffling
        
        min_max_norm : 
            Whether to scale the data to the range [-1, 1]
        """
        self._path = Path(path)

        rng = np.random.default_rng(seed)

        with open(self._path / f"{subset}.csv", 'r') as f:
            self._files = f.read().split(',')
            if shuffle:
                rng.shuffle(self._files) # inplace operation
            else:
                self._files = sorted(self._files)

        if file_base_path:
            self._file_base_path = Path(file_base_path)
            self._files = [
                str(self._file_base_path/file_path) for file_path in self._files
            ]
        else:
            self._file_base_path = None

        if sample_length is None:
            raise ValueError("Sample length must not be None")
        self.sample_length = sample_length

        self.should_norm = min_max_norm

        self.conditional_loader = conditional_loader

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, Tensor]:
        file_path = self._files[n]
        return self.load_audio(file_path, n)

    def __len__(self) -> int:
        return len(self._files)

    def min_max_norm(self, tensor: Tensor) -> Tensor:
        """
        Min-max scale input tensor from `Int16` range to `[-1, 1]`.

        The tensor can theoretically have any dtype, but min and max values
        are assumed to be the min and max values of `Int16`, which are
        `-32768` and `32767`.

        The general formula for range `[a, b]` is:
        ```
        a + ( (x - min) * (b - a) ) / (max - min)
        ```
        """
        return -1 + (tensor + 32768) * 2 / 65535

    def load_audio(self, file_path: str, n: int = None) -> Tuple[Tensor, int, str, Tensor]:
        waveform, sample_rate = torchaudio.load(file_path)

        # Maybe perform min-max normalization
        if self.should_norm:
            waveform = self.min_max_norm(waveform)

        # Norm waveform to designated length and get padding mask
        waveform, mask = utils.fix_length(waveform)

        if self.conditional_loader is not None:
            conditional_signal = self.conditional_loader(file_path, n)
        else:
            conditional_signal = ""

        return (waveform, sample_rate, conditional_signal, mask)
