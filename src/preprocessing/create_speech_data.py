"""
Create an augmented dataset of speech audio fragments from the VariaNTS corpus
as well as artificial speakers.

User must make sure before hand that no duplicate names in input sources

Run from repository root

python src/preprocessing/create_speech_data.py \
    -i data/VariaNTS/VariaNTS\ corpus/,data/VariaNTS/TTS_VariaNTS_HP_split \
    -o data/VariaNTS/VariaNTS_words_HP \
    -so data/datasplits/VariaNTS/VariaNTS_words_HP
-----------------------------------------------------------------------------
"""

import argparse
from functools import partial
import logging
from multiprocessing import Pool
import os
import shutil
import sys
from typing import Optional, Union, List

sys.path.append('src')

from audiomentations import TimeStretch, PitchShift, AirAbsorption, RoomSimulator
import numpy as np
from pydub import AudioSegment
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

from utils.generic import get_word_from_filepath, fix_length


# Define Rich as the handler for the logging module
logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")
console = Console()


def peak_value_norm(samples):
    """
    Normalize peak value of audio array to max value in Int16 range.
    Does not normalize the mean.
    """
    return samples / np.max(np.abs(samples)) * 32767

def augment_file(filename, path, base_augs, n_augs_per_type):
    
    _out = [filename]

    sr, samples = wavread(os.path.join(path, filename))

    # Aug function expects Float32 data type
    samples = samples.astype(np.float32)

    for aug_type, value in base_augs.items():
        aug_class, aug_params = value

        for i in range(n_augs_per_type):
            # Initialize augmentation function
            aug_func = aug_class(**aug_params)

            aug_name = f'{aug_type}{i+1}'

            # Apply new augmentation
            augmented = aug_func(samples=samples, sample_rate=sr)
            augmented = peak_value_norm(augmented)

            # Cast back to Int16 for exporting
            augmented = augmented.astype(np.int16)

            aug_filename = filename.replace('.wav', f'_{aug_name}.wav')

            wavwrite(
                os.path.join(path, aug_filename),
                rate = sr,
                data = augmented,
            )

            _out.append(aug_filename)

    # Normalize 
    # (produces error when done with Int16 dtype, so we do it after casting, and
    # also after augmenting, since an augmentation might increase loudness
    # further, causing clipping)
    samples = peak_value_norm(samples)

    # Cast back to Int16 and export
    samples = samples.astype(np.int16)
    wavwrite(os.path.join(path, filename), rate = sr, data = samples)

    return _out


def main(
    input_path: Union[str, List[str]],
    output_path: str,
    splits_output_path: str,
    targets: Optional[Union[str, List[str]]],
    sampling_rate: int,
    segment_length: int,
    n_augs_per_type: int,
    n_validation_speakers: int,
):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        log.error(f"Output directory already exists: {output_path}")
        sys.exit()

    try:
        os.makedirs(splits_output_path)
    except FileExistsError:
        log.error(f"Splits output directory already exists: {splits_output_path}")
        sys.exit()

    if type(input_path) == str:
        input_paths = input_path.split(',')
    else:
        input_paths = input_path
    console.log('Reading from input source(s):')
    for i, inp_path in enumerate(input_paths):
        console.log(f'{i+1}. {inp_path}')

    if targets is None:
        with open('data/HP_VariaNTS_intersection.txt', 'r') as f:
            targets = f.read().split(',')
    elif type(targets) == str:
        targets = targets.split(',')
    console.log(f'Target words: {targets}')


    # 1. Copy to output directory, using a flattened directory structure
    speaker_ids = []
    for i,inp_path in enumerate(input_paths):
        cnt = 0
        for root, subdirs, files in os.walk(inp_path):
            for file in files:
                if 'föhn' in file and 'fohn' in targets:
                    log.error(
                        "The incorrect label 'föhn' has been found in "
                        f"the following file: {os.path.join(root,file)}. "
                        "Please rename the file to the correct word 'fohn'."
                    )
                    sys.exit()
                is_target = get_word_from_filepath(file) in targets
                correct_subclass = "_words" in root.split("/")[-1]
                if file.endswith('.wav') and is_target and correct_subclass:
                    person = root.split("/")[-2]
                    if person not in speaker_ids:
                        speaker_ids.append(person)
                    full_file = os.path.join(root, file)
                    out_file = os.path.join(output_path, f'{person}_{file}')
                    shutil.copy(full_file, out_file)
                    cnt += 1
        console.log(f"Copied {cnt} files from source {i+1} to designated output path")
    speaker_ids = sorted(speaker_ids)
    console.log(f"Found the following speaker IDs: {speaker_ids}")


    # 2. Resample to desired framerate
    files = os.listdir(output_path)
    for i, fn in enumerate(track(files, description='Resampling', transient=True)):
        fn = os.path.join(output_path, fn)       
        # Read audio
        audio = AudioSegment.from_file(fn, format="wav")
        # Decrease sampling rate
        audio = audio.set_frame_rate(sampling_rate)
        # Overwrite file
        audio.export(fn, format="wav")
    console.log(f'Resampled {len(files)} files to {sampling_rate} Hz')


    # 3. Apply augmentations

    # The types of augmentations and their parameters to use. 
    # Note that pitch shifting upwards makes the audio sound very mechanical, 
    # so we only shift downwards
    base_augs = {
        'timestretch' : (TimeStretch, {'min_rate': 0.6, 'max_rate': 1.4, 'p': 1.0}),
        'pitch' : (PitchShift, {'min_semitones': -1.5, 'max_semitones': 0, 'p': 1.0}),
        'airAbs' : (AirAbsorption, {'p': 1.0}),
        'roomSim' : (RoomSimulator, {'p': 1.0}),
    }

    # Configure the augmentation function as a partial function, such that only
    # the changing parameter (the filename) has to be provided as input
    augment_function = partial(
        augment_file,
        path=output_path,
        base_augs=base_augs,
        n_augs_per_type=n_augs_per_type,
    )
    with Pool() as p:
        aug_files = [
            result for result in track(
                p.imap(augment_function, files), 
                total=len(files), 
                description="Creating augmentations",
                transient=True, 
            )
        ]

    # Flatten results
    aug_files = [item for sublist in aug_files for item in sublist]

    console.log(
        f'Created {len(aug_files)-len(files)} augmented files, '
        f'yielding a total of {len(aug_files)} files'
    )


    # 4. Unify file lengths
    for fn in track(
        os.listdir(output_path), description='Fixing lengths', transient=True
    ):
        fr, sample = wavread(os.path.join(output_path, fn))
        sample = fix_length(sample, segment_length)
        wavwrite(os.path.join(output_path, fn), fr, sample)
    
    console.log(
        f'Unified all files to length of {segment_length} samples at '
        f'{sampling_rate} Hz'
    )


    # 5. Create training and validation splits
    if n_validation_speakers >= len(speaker_ids):
        log.error(
            'Number of speakers to add to the validation set must be '
            'smaller than the total number of speakers: '
            f'{n_validation_speakers} vs {len(speaker_ids)}'
        )
        sys.exit()

    val_files = []
    train_files = os.listdir(output_path)

    rng = np.random.default_rng(1144)

    # For each word
    for word in targets:
        # Randomly pick n speakers
        speaker_ids = rng.choice(
            speaker_ids, n_validation_speakers, replace=False)
        # Pick files
        for id in speaker_ids:
            matching_files = [
                fn for fn in train_files if id in fn and word in fn
            ]
            for fn in matching_files:
                val_files.append(fn)
                train_files.remove(fn)

    val_files_noaug = [fn for fn in val_files if len(fn.split('_')) == 2]
    train_files_noaug = [fn for fn in train_files if len(fn.split('_')) == 2]

    console.log(
        f'Created training and validation split with {len(train_files)} '
        f'training and {len(val_files)} validation files.\n'
        f'Train files without augmentation: {len(train_files_noaug)}. '
        f'Val files without augmentation: {len(val_files_noaug)}'
    )

    train_files = sorted(train_files)
    val_files = sorted(val_files)
    train_files_noaug = sorted(train_files_noaug)
    val_files_noaug = sorted(val_files_noaug)

    # Write the splits as CSV files to disk
    # Note that we use the augmented files for the default training split,
    # but the non-augmented files for the default validation split
    with open(os.path.join(splits_output_path, 'train.csv'), 'w') as f:
        f.write(','.join(train_files))
    with open(os.path.join(splits_output_path, 'val.csv'), 'w') as f:
        f.write(','.join(val_files_noaug))
    with open(os.path.join(splits_output_path, 'train_noaug.csv'), 'w') as f:
        f.write(','.join(train_files_noaug))
    with open(os.path.join(splits_output_path, 'val_aug.csv'), 'w') as f:
        f.write(','.join(val_files))

    console.log(
        f'Saved training and validation splits at {splits_output_path}'
    )

    console.log('Done')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    parser.add_argument('--splits_output_path', '-so', type=str, required=True)
    parser.add_argument('--targets', '-t', type=str)
    parser.add_argument('--sampling_rate', '-sr', type=int, default=16000)
    parser.add_argument('--segment_length', '-sl', type=int, default=16000)
    parser.add_argument('--n_augs_per_type', type=int, default=4)
    parser.add_argument('--n_validation_speakers', type=int, default=3)
    
    main(**vars(parser.parse_args()))