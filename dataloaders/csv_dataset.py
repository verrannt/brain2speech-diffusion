# Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py
# and https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py


from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor

import random
from typing import Tuple


class CSVDataset(Dataset):
    """
    Create a Dataset from a .csv file list of audio files.
    Each returned item is a tuple of the form: waveform, sample_rate
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
    ):
        """
        Read files from .csv file on disk. File must be present in `path` as
        `subset.csv`, e.g. `/data/user/train.csv`.

        path : Path to the directory on disk where the .csv file is stored
        subset : One of "train", "val", "test". Which subset of the data to 
            load. The chosen subset must be present as "<subset>.csv" in the
            `path` given as argument, e.g. "train.csv".
        file_base_path : If given, this path is prepended to every filename in
            the loaded .csv file.
        sample_length : Desired length of audio sequence (in sampled points). 
            Any files shorter will be padded, any files longer will be cut to
            this length.
        shuffle : Whether to shuffle the files read from the .csv file
        seed : Seed for the random number generator used for shuffling
        min_max_norm : Whether to scale the data to the range [-1, 1]
        """
        self._path = Path(path)

        if seed:
            random.seed(seed)

        with open(self._path / f"{subset}.csv", 'r') as f:
            self._files = f.read().split(',')
            if shuffle:
                random.shuffle(self._files) # inplace operation
            else:
                self._files = sorted(self._files)

        if file_base_path:
            self._file_base_path = Path(file_base_path)
            self._files = [
                str(self._file_base_path/file_path) for file_path in self._files
            ]
        else:
            self._file_base_path = None

        if not sample_length:
            raise ValueError("Sample length cannot be None")
        self.sample_length = sample_length

        self.should_norm = min_max_norm

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, Tensor]:
        file_path = self._files[n]
        return self.load_audio(file_path)

    def __len__(self) -> int:
        return len(self._files)

    def fix_length(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
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

        if tensor.shape[1] > self.sample_length:
            # If tensor is longer than desired length, the mask is only True
            # values
            mask = torch.ones((1,self.sample_length), dtype=torch.bool)

            return tensor[:,:self.sample_length], mask
        
        elif tensor.shape[1] < self.sample_length:
            # If tensor is shorter than desired length, the mask is True until
            # the original length of the tensor, and False afterwards
            mask = torch.zeros((1,self.sample_length), dtype=torch.bool)
            mask[:,:tensor.shape[1]] = True
            
            # We pad the tensor with zero values to increase its size to the
            # desired length
            padded_tensor = torch.cat([
                tensor, 
                torch.zeros(1, self.sample_length-tensor.shape[1])
            ], dim=1)

            return padded_tensor, mask
        
        else:
            # In case the tensor has already the desired length, we use a mask
            # of only True values again and can return the tensor unaltered
            mask = torch.ones((1,self.sample_length), dtype=torch.bool)
            return tensor, mask

    def standardize(self, tensor: Tensor) -> Tensor:
        """
        Standardize an input tensor to mean 0 and standard deviation 1.
        """
        return (tensor - torch.mean(tensor)) / torch.std(tensor)


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

    def load_audio(self, file_path: str) -> Tuple[Tensor, int, str, Tensor]:
        waveform, sample_rate = torchaudio.load(file_path)

        # Maybe perform min-max normalization
        if self.should_norm:
            waveform = self.min_max_norm(waveform)

        # Norm waveform to designated length and get padding mask
        waveform, mask = self.fix_length(waveform)

        # NOTE We return an empty label here as the third element to 
        # ensure compatibility across the APIs of all dataset loaders
        return (waveform, sample_rate, "", mask)
