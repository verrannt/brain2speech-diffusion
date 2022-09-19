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

    def __init__(self, path: str, subset: str, sample_length: int = 16000):
        """
        Read files from .csv file on disk. File must be present in `path` as
        `subset.csv`, e.g. `/data/user/train.csv`.

        path : Path to the directory on disk where the .csv files are stored
        subset : One of "train", "val", "test". Which subset of the data to 
            load. The chosen subset must be present as "subset.csv" in the
            `path` given as argument, e.g. "train.csv".
        sample_length : Desired length of audio sequence (in sampled points). 
            Any files shorter will be padded, any files longer will be cut to
            this length.
        """
        self._path = Path(path)

        with open(self._path / f"{subset}.csv", 'r') as f:
            self._files = sorted(f.read().split(','))

        if not sample_length:
            raise ValueError("Sample length cannot be Null")
        self.sample_length = sample_length

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        file_path = self._files[n]
        return self.load_audio(file_path)

    def __len__(self) -> int:
        return len(self._files)

    def fix_length(self, tensor):
        assert len(tensor.shape) == 2 and tensor.shape[0] == 1

        if tensor.shape[1] > self.sample_length:
            return tensor[:,:self.sample_length]
        
        elif tensor.shape[1] < self.sample_length:
            return torch.cat([
                tensor, 
                torch.zeros(1, self.sample_length-tensor.shape[1])
            ], dim=1)
        
        else:
            return tensor

    def load_audio(self, file_path: str) -> Tuple[Tensor, int]:
        waveform, sample_rate = torchaudio.load(file_path)
        # NOTE We return an empty label here as the third tuple element to 
        # ensure compatibility across the APIs of all dataset loaders
        return (self.fix_length(waveform), sample_rate, "")
