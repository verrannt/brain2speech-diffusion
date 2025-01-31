# -----------------------------------------------------------------------------
#
# Adapted from:
#   https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py
#   https://github.com/albertfgu/diffwave-sashimi/blob/master/dataloaders/sc.py
#
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np


class CSVDataset(Dataset):
    """
    Create a Dataset from a list of audio files stored as .csv file on disk. File must be present
    in `csv_path` as `<subset>.csv`, e.g. `/data/user/train.csv`.

    Parameters
    ----------
    csv_path:
        Path to the directory on disk where the .csv file is stored

    subset:
        One of "train", "val", "test". Which subset of the data to
        load. The chosen subset must be present as "<subset>.csv" in the
        `csv_path` given as argument, e.g. "train.csv".

    audio_path:
        If given, this path is prepended to every filename in
        the loaded .csv file.

    shuffle:
        Whether to shuffle the files read from the .csv file

    seed:
        Seed for the random number generator used for shuffling

    min_max_norm:
        Whether to scale the data to the range [-1, 1]
    """

    def __init__(
        self,
        csv_path: str,
        subset: str,
        audio_path: str,
        shuffle: bool = True,
        seed: int = None,
        conditional_loader=None,
    ) -> None:
        self._path = Path(csv_path)

        rng = np.random.default_rng(seed)

        with open(self._path / f"{subset}.csv", "r") as f:
            self._files = f.read().split(",")
            if shuffle:
                rng.shuffle(self._files)  # inplace operation
            else:
                self._files = sorted(self._files)

        if audio_path:
            audio_path = Path(audio_path)
            self._files = [str(audio_path / file_path) for file_path in self._files]

        self.conditional_loader = conditional_loader

        self.subset = subset

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Union[Tensor, str], str]:
        """
        Load the `n`-th file from the dataset

        Parameters
        ----------
        n:
            The index at which to load from the internal list of files.

        Returns
        -------
        The loaded audio, sampling rate, optional conditional signal, and file path.
        See `load_audio()` for details.

        Raises
        ------
        IndexError:
            if `n >= len(self._files)`
        """
        file_path = self._files[n]
        return self.load_audio(file_path)

    def __len__(self) -> int:
        return len(self._files)

    def load_audio(self, file_path: str) -> Tuple[Tensor, int, Union[Tensor, str], str]:
        """
        Load an audio file, given its path on disk. Also returns the audio's sampling rate, and a
        matching conditional input if `self.conditional_loader` is given.

        Parameters
        ----------
        file_path:
            Path to the location of the audio file on disk

        Returns
        -------
        A tuple of the audio waveform (`Tensor`), the sampling rate (`int`), the conditional signal
        (either a `Tensor` if a conditional loader is given, or an empty `str` if not), and the
        file path itself (`str`).
        """
        waveform, sample_rate = torchaudio.load(file_path)

        if self.conditional_loader is not None:
            conditional_signal = self.conditional_loader(file_path, set=self.subset)
        else:
            conditional_signal = ""

        return (waveform, sample_rate, conditional_signal, file_path)
