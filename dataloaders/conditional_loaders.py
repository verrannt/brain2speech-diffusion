import os

import numpy as np
import torch

import dataloaders.utils as utils


class EEGRandomLoader():
    """
    Given a filepath pointing to an audio file for a given word, randomly 
    load an EEG file corresponding to the word.
    """

    def __init__(
        self,
        path: str,
        seed: int | None, 
        segment_length: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.files = [
            file for file in os.listdir(path)
            if file.endswith('.npy')
        ]

        self.segment_length = segment_length

    def __call__(
        self,
        file_path: str, # path to the word audio file
        n: int = None, # only here for compatibility
    ):
        # Isolate word from given file path
        word = file_path.split('/')[-1].split('.')[0]

        # Find all EEG files for this word
        files = [
            file for file in self.files 
            if file.split('/')[-1].split('.')[0] == word
        ]

        # Randomly select one of the EEG files
        file = self.rng.choice(files)

        # Load from path
        eeg = torch.from_numpy(np.load(file)).float()

        # Standardize
        eeg = utils.standardize_eeg(eeg)

        # Adjust to designated length
        eeg = utils.fix_length_3d(eeg, self.segment_length)

        return eeg
