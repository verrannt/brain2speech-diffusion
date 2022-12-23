from abc import ABC, abstractmethod
import os
from pathlib import Path
import re

import numpy as np
import torch

import dataloaders.utils as utils


def get_word_from_filepath(filepath: str, uses_augmentation: bool = True, uses_numbering: bool = True) -> str:
    """ Extract the word from a given filepath pointing to an audio file on disk. """
    # Get the last part of the path (i.e. just the filename)
    filepath = filepath.split('/')[-1]
    # Get the filename before the file extension
    filepath = filepath.split('.')[0]
    # Augmented files are named according to {word}_{aug-type}.wav, so this removes the augmentation from the name
    if uses_augmentation:
        filepath = filepath.split('_')[0]
    # Some files (e.g. EEG files) are numbered (e.g. goed1.npy), so this removes any digits from the name
    if uses_numbering:
        filepath = re.sub(r'[0-9]', '', filepath)
    return filepath


class ClassConditionalLoader:
    """
    Get one-hot encoded class labels based on words from a given file.
    """

    def __init__(self, words_file) -> None:
        with open(words_file, 'r') as file:
            words = file.read().split(',')
        self.word_tokens = { words[i] : i for i in range(len(words))}
        self.num_classes = len(words)

    def __call__(self, audio_file_path: str):
        word = get_word_from_filepath(audio_file_path)
        try:
            idx = self.word_tokens[word]
        except KeyError:
            raise ValueError(f"Unrecognized word: {word}")
        return torch.LongTensor([idx])


class EEGLoader(ABC):
    """
    Abstract class for implementing an EEG loader. Subclasses must implement the `retrieve_file` function which returns
    an EEG file when given an audio file as input.
    """

    def __init__(self, segment_length: int) -> None:
        self.segment_length = segment_length

    def __call__(self, audio_file_path: str):
        # Retrieving a matching EEG file must be handled by the inheriting classes
        eeg_file = self.retrieve_file(audio_file_path)
        # Load from path
        eeg = torch.from_numpy(np.load(eeg_file)).float()
        # Standardize
        eeg = utils.standardize_eeg(eeg)
        # Adjust to designated length
        eeg = utils.fix_length_3d(eeg, self.segment_length)
        return eeg

    @abstractmethod
    def retrieve_file(self, audio_file_path: str) -> str:
        pass


class EEGRandomLoader(EEGLoader):
    """
    Given a filepath pointing to an audio file for a given word, randomly 
    load an EEG file corresponding to the word.
    """

    def __init__(
        self,
        path: str,
        seed: int, 
        segment_length: int,
    ) -> None:
        super().__init__(segment_length)
        self.rng = np.random.default_rng(seed)
        self.files = [
            file for file in os.listdir(path)
            if file.endswith('.npy')
        ]
        self.path = Path(path)

    def retrieve_file(self, audio_file_path: str) -> str:
        # Isolate word from given file path
        word = get_word_from_filepath(audio_file_path, uses_numbering=False)
        # Find all EEG files for this word
        fitting_files = []
        for file in self.files:
            cut_word = get_word_from_filepath(file, uses_augmentation=False)
            if cut_word == word:
                fitting_files.append(file)

        if len(fitting_files) == 0:
            raise ValueError(f"No files found for {word}")

        # Randomly select one of the EEG files
        file = self.rng.choice(fitting_files)

        # Prepend path
        file = self.path / file

        return file


class EEGExactLoader(EEGLoader):
    """
    Given a filepath pointing to an audio file for a given word, load the EEG
    file corresponding to exactly that audio recording.
    """

    def __init__(
        self,
        path: str,
        segment_length: int,
    ) -> None:
        super().__init__(segment_length)
        self.path = Path(path)

    def retrieve_file(self, audio_file_path: str) -> str:
        # Isolate word from given file path
        # Note that we must not remove numbering, since the number tells us which EEG file to load (e.g. 'goed7.wav'
        # corresponds to 'goed7.npy')
        word = get_word_from_filepath(audio_file_path, uses_numbering=False, uses_augmentation=False)

        # Select the EEG file for this word
        file = self.path / f'{word}.npy'

        return file
