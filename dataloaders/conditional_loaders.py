from abc import ABC, abstractmethod
import os
from pathlib import Path
import re

import numpy as np
import torch
from torch import Tensor

import dataloaders.utils as utils


AUDITORY_CORTEX_IDX = [14, 15, 18, 19, 20, 21, 22, 23]


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
    Load class conditional input vector for a given word. Returns a callable object that upon being called with a word
    (or filepath containing a the word in the filename), returns a one-hot encoded class vector. The indexes of the
    encoding are specified in a words file provided at initialization.
    """

    def __init__(self, words_file: str) -> None:
        """
        Params:
        ---
        `words_file`: path to the file containing the words to index. Has to be comma-separated, without blank spaces.
        """
        with open(words_file, 'r') as file:
            words = file.read().split(',')
        self.word_tokens = { words[i] : i for i in range(len(words))}
        self.num_classes = len(words)

    def __call__(self, audio_file_path: str, **kwargs) -> Tensor:
        word = get_word_from_filepath(audio_file_path)
        try:
            idx = self.word_tokens[word]
        except KeyError:
            raise ValueError(f"Unrecognized word: {word}")
        out = torch.LongTensor([idx])
        out = torch.nn.functional.one_hot(out, self.num_classes)
        out = out.type(torch.float32)
        return out


class EEGLoader(ABC):
    """
    Abstract class for implementing an EEG loader. Subclasses must implement the `retrieve_file` function which returns
    an EEG file when given an audio file as input.
    """

    def __init__(self, segment_length: int) -> None:
        self.segment_length = segment_length

    def __call__(self, audio_file_path: str, **kwargs) -> Tensor:
        # Retrieving a matching EEG file must be handled by the inheriting classes
        eeg_file = self.retrieve_file(audio_file_path, **kwargs)
        eeg = self.process_eeg(eeg_file, self.segment_length)
        return eeg

    @staticmethod
    def process_eeg(eeg_file: str, segment_length: int) -> Tensor:
        # Load from path
        eeg = torch.from_numpy(np.load(eeg_file)).float()
        # Select auditory cortex electrodes only
        eeg = eeg[:, AUDITORY_CORTEX_IDX, :]
        # Standardize
        eeg = utils.standardize_eeg(eeg)
        # Adjust to designated length
        eeg = utils.fix_length_3d(eeg, segment_length)
        return eeg

    @abstractmethod
    def retrieve_file(self, audio_file_path: str, **kwargs) -> str:
        pass


class EEGRandomLoader(EEGLoader):
    """
    Given a filepath pointing to an audio file for a given word, randomly 
    load an EEG file corresponding to the word.
    """

    def __init__(
        self,
        path: str,
        splits_path: str,
        seed: int, 
        segment_length: int,
    ) -> None:
        super().__init__(segment_length)

        self.rng = np.random.default_rng(seed)

        with open(Path(splits_path) / 'train.csv', 'r') as f_t, \
                open(Path(splits_path) / 'val.csv', 'r') as f_v:
            self.train_words = [word.split('.')[0] for word in f_t.read().split(',')]
            self.val_words = [word.split('.')[0] for word in f_v.read().split(',')]

        train_files = [
            file for file in os.listdir(path)
            if file.endswith('.npy') and \
                get_word_from_filepath(file, uses_numbering=False) in self.train_words
        ]

        val_files = [
            file for file in os.listdir(path)
            if file.endswith('.npy') and \
                get_word_from_filepath(file, uses_numbering=False) in self.val_words
        ]

        self.files = {'train': train_files, 'val': val_files}

        self.path = Path(path)

    def retrieve_file(self, audio_file_path: str, set: str) -> str:
        # Isolate word from given file path. Note that there is no need to remove numbers as the dataset supposed to
        # provide the audio files has no numbering in the top-level filename. This will break in case of file numbering.
        word = get_word_from_filepath(audio_file_path, uses_numbering=False)
        
        # Find all EEG files for this word
        fitting_files = [
            file for file in self.files[set]
            if get_word_from_filepath(file, uses_augmentation=False) == word
        ]

        if len(fitting_files) == 0:
            raise ValueError(f"No EEG files found for {audio_file_path}")

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

    def retrieve_file(self, audio_file_path: str, **kwargs) -> str:
        # Isolate word from given file path
        # Note that we must not remove numbering, since the number tells us which EEG file to load 
        # (e.g. 'goed7.wav' corresponds to 'goed7.npy')
        word = get_word_from_filepath(audio_file_path, uses_numbering=False, uses_augmentation=False)

        # Select the EEG file for this word
        file = self.path / f'{word}.npy'

        return file
