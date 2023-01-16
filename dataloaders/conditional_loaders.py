from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
from typing import List

import numpy as np
import torch
from torch import Tensor

from . import utils


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
    # Some files (e.g. ECoG files) are numbered (e.g. goed1.npy), so this removes any digits from the name
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

    def batch_call(self, audio_file_list: List[str], one_hot: bool = True) -> Tensor:
        """
        For faster processing of multiple files, call this function with a list of file paths. Will return a 
        batched tensor of one-hot encoded class labels if `one_hot==True`, else just a tensor of the indexes.
        """
        words = [get_word_from_filepath(fp) for fp in audio_file_list]
        try:
            idxs = [self.word_tokens[word] for word in words]
        except KeyError:
            raise ValueError(f"Could not recognize one of {words}")
        if one_hot:
            out = torch.LongTensor(idxs)
            out = torch.nn.functional.one_hot(out, self.num_classes)
            out = out.type(torch.float32)
        else:
            out = torch.FloatTensor(idxs)
        return out


class ECOGLoader(ABC):
    """
    Abstract class for implementing an ECoG loader. Subclasses must implement the `retrieve_file` function which returns
    an ECoG file when given an audio file as input.
    """

    def __init__(self, segment_length: int) -> None:
        self.segment_length = segment_length

    def __call__(self, audio_file_path: str, **kwargs) -> Tensor:
        # Retrieving a matching ECoG file must be handled by the inheriting classes
        ecog_file = self.retrieve_file(audio_file_path, **kwargs)
        ecog = self.process_ecog(ecog_file, self.segment_length)
        return ecog

    @staticmethod
    def process_ecog(ecog_file: str, segment_length: int) -> Tensor:
        # Load from path
        ecog = torch.from_numpy(np.load(ecog_file)).float()
        # Select auditory cortex electrodes only
        ecog = ecog[:, AUDITORY_CORTEX_IDX, :]
        # Standardize
        ecog = utils.standardize_ecog(ecog)
        # Adjust to designated length
        ecog = utils.fix_length_3d(ecog, segment_length)
        return ecog

    @abstractmethod
    def retrieve_file(self, audio_file_path: str, **kwargs) -> str:
        pass


class ECOGRandomLoader(ECOGLoader):
    """
    Given a filepath pointing to an audio file for a given word, randomly 
    load an ECoG file corresponding to the word.
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
        
        # Find all ECoG files for this word
        fitting_files = [
            file for file in self.files[set]
            if get_word_from_filepath(file, uses_augmentation=False) == word
        ]

        if len(fitting_files) == 0:
            raise ValueError(f"No ECoG files found for {audio_file_path}")

        # Randomly select one of the ECoG files
        file = self.rng.choice(fitting_files)

        # Prepend path
        file = self.path / file

        return file


class ECOGExactLoader(ECOGLoader):
    """
    Given a filepath pointing to an audio file for a given word, load the ECoG
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
        # Note that we must not remove numbering, since the number tells us which ECoG file to load 
        # (e.g. 'goed7.wav' corresponds to 'goed7.npy')
        word = get_word_from_filepath(audio_file_path, uses_numbering=False, uses_augmentation=False)

        # Select the ECoG file for this word
        file = self.path / f'{word}.npy'

        return file
