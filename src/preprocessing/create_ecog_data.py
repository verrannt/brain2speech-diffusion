"""
python src/create_ecog_data.py \
    -i data/hp_reading \
    -o data/HP1_ECoG_conditional/sub-002/ \
    -t data/HP_VariaNTS_intersection.txt \
    -sr 100 \
    -s 2
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from textgrids import TextGrid


def main(
    input_path: str,
    targets: str,
    output_path: str,
    subject: int,
    sampling_rate: int,
):
    subject_id = f"sub-00{subject}"
    data_dir = Path(input_path) / subject_id
    output_dir = Path(output_path)

    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        print("[ FATAL ] Output directory already exists, aborting.")
        sys.exit()

    with open(targets, "r") as f:
        targets = f.read().split(",")

    word_counter = defaultdict(int)

    # NOTE When working on different datasets, these have to be adjusted.
    # Currently, these are only being used for subject 2.
    if subject == 2:
        selected_electrodes = np.concatenate([np.arange(8) + 16 * i for i in range(0, 6)])
        print(f"Selecting {len(selected_electrodes)} electrodes.")
    else:
        selected_electrodes = None
        print("Using all electrodes")

    # lfb_filename = 'lfb_hp_reading_ecog_car_1-30_avgfirst_100Hz_log_norm.npy'
    hfb_filename = "hfb_hp_reading_ecog_car_70-170_avgfirst_100Hz_log_norm.npy"

    intervals = []
    ecogs = []
    audios = []

    for _run in [1, 2]:
        # The TextGrid which contains all detected word intervals
        textgrid = TextGrid(data_dir / f"{subject_id}_ses-iemu_acq-ECOG_run-0{_run}_audio.TextGrid")

        # Pick the relevant intervals from the text grid, i.e. only those of words in the intersection
        for interval in textgrid["words"]:
            if interval.text in targets:
                interval.run = _run
                intervals.append(interval)

        ecog_file_path = (
            data_dir / f"{subject_id}_ses-iemu_acq-ECOG_run-0{_run}_ieeg/{hfb_filename}"
        )
        ecogs.append(np.load(ecog_file_path))

        audio_file_path = (
            data_dir / f"{subject_id}_ses-iemu_acq-ECOG_run-0{_run}_audio_pitch_shifted.wav"
        )
        audio_sr, audio_signal = wavread(audio_file_path)
        audio_mean = np.mean(audio_signal)
        audios.append((audio_sr, audio_signal, audio_mean))

    # Find length of longest interval, which we will use as length for all
    max_len = round(
        np.max([interval.xmax - interval.xmin for interval in intervals]) * sampling_rate
    )

    print("Longest interval:", max_len)

    for interval in tqdm(intervals, desc="Processing Intervals"):
        word_counter[interval.text] += 1

        ecog = ecogs[interval.run - 1][
            int(interval.xmin * sampling_rate) : int(interval.xmin * sampling_rate) + max_len
        ]  # -> Shape (TIMESTEPS, ELECTRODES)

        if selected_electrodes is not None:
            ecog = ecog[:, selected_electrodes]

        # Transpose to shape (E, T)
        ecog = ecog.T

        # Chop section out of audio file
        audio_samplerate, audio, audio_mean = audios[interval.run - 1]
        audio_interval = audio[
            int(interval.xmin * audio_samplerate) : int(interval.xmax * audio_samplerate)
        ]

        # Align audio to (around) mean 0 and normalize peak value to Int16 range
        audio_interval = audio_interval.astype(np.float32)
        audio_interval = audio_interval - audio_mean
        audio_interval = audio_interval / np.max(np.abs(audio_interval)) * 32767
        audio_interval = audio_interval.astype(np.int16)

        output_name = f"{interval.text}{word_counter[interval.text]}"

        # Save interval audio as wavfile
        wavwrite(
            output_dir / f"{output_name}.wav",
            rate=audio_samplerate,
            data=audio_interval,
        )

        # Save ECoG data as numpy array
        np.save(output_dir / f"{output_name}.npy", ecog)

    print("Writing dataset information to file.")

    word_counter = dict(reversed(sorted(word_counter.items(), key=lambda item: item[1])))
    with open(output_dir / "about.txt", "w") as f:
        f.write(f"Longest interval: {max_len}\n")
        f.write(f"Unique words: {len(word_counter)}\n")
        f.write(f"Total words: {len(intervals)}\n")

        f.write("\nWord counts:\n-------")
        for word, count in word_counter.items():
            f.write(f"\n{word}: {count}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    parser.add_argument("--targets", "-t", type=str)
    parser.add_argument("--sampling_rate", "-sr", type=int, default=100)
    parser.add_argument("--subject", "-s", type=int)

    main(**vars(parser.parse_args()))
