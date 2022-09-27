# Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py
# and https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py


from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor

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
        sample_length: int = 16000
    ):
        """
        Read files from .csv file on disk. File must be present in `path` as
        `subset.csv`, e.g. `/data/user/train.csv`.

        path : Path to the directory on disk where the .csv files are stored
        subset : One of "train", "val", "test". Which subset of the data to 
            load. The chosen subset must be present as "subset.csv" in the
            `path` given as argument, e.g. "train.csv".
        file_base_path : If given, this path is prepended to every filename in
            the loaded .csv file.
        sample_length : Desired length of audio sequence (in sampled points). 
            Any files shorter will be padded, any files longer will be cut to
            this length.
        """
        self._path = Path(path)

        with open(self._path / f"{subset}.csv", 'r') as f:
            self._files = sorted(f.read().split(','))

        self._file_base_path = Path(file_base_path) if file_base_path else None

        if file_base_path:
            self._files = [
                str(self._file_base_path/file_path) for file_path in self._files
            ]

        if not sample_length:
            raise ValueError("Sample length cannot be None")
        self.sample_length = sample_length

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

    def load_audio(self, file_path: str) -> Tuple[Tensor, int, str, Tensor]:
        waveform, sample_rate = torchaudio.load(file_path)

        # Norm waveform to designated length and get padding mask
        waveform, mask = self.fix_length(waveform)

        # NOTE We return an empty label here as the third element to 
        # ensure compatibility across the APIs of all dataset loaders
        return (waveform, sample_rate, "", mask)
