"""

NOTE: Assumes the ECoG data is already normalized


# Recreating the data used in the thesis

Run the below. This uses the 100 Hz ECoG data and TextGrid files on the 55 word classes that are in
both VariaNTS and Harry Potter data, with a selection of electrodes as specified in the thesis.

python src/preprocessing/create_ecog_data.py \
    --ecog data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_ieeg/hfb_hp_reading_ecog_car_70-170_avgfirst_100Hz_log_norm.npy,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_ieeg/hfb_hp_reading_ecog_car_70-170_avgfirst_100Hz_log_norm.npy \
    --audio data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_audio_pitch_shifted.wav,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_audio_pitch_shifted.wav \
    --timestamps data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_audio.TextGrid,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_audio.TextGrid \
    -o data/HP1_ECoG_conditional/sub-002 \
    --use_subject_two_selection True

    
# Recreating the data for the follow-up experiment

Run the below. This uses the 15 Hz ECoG data and .csv files of the timestamps of the 6 most frequent
intersection words. These 6 words are specified using the --targets flag. Further, we use the 
interval [-0.2, 0.25] for the ECoG fragments.

python src/preprocessing/create_ecog_data.py \
    --ecog data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_ieeg/hfb_hp_reading_ecog_car_70-170_15Hz_ch_inter_norm.npy,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_ieeg/hfb_hp_reading_ecog_car_70-170_15Hz_ch_inter_norm.npy \
    --audio data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_audio_pitch_shifted.wav,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_audio_pitch_shifted.wav \
    --timestamps data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-01_audio_variant_words_most_frequent_n6_with_substrings.csv,data/hp_reading/sub-002/sub-002_ses-iemu_acq-ECOG_run-02_audio_variant_words_most_frequent_n6_with_substrings.csv \
    -o data/HP1_ECoG_conditional/sub-002_top6 \
    -t dag,heel,kan,keer,man,wel \
    -sr 15 \
    -min -0.2 \
    -max 0.25
"""

import argparse
from collections import defaultdict
import logging
from math import floor, ceil
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from textgrids import TextGrid


# Define Rich as the handler for the logging module
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")
console = Console()


def main(
    ecog: str,
    audio: str,
    timestamps: str,
    output_path: str,
    targets: str = None,
    sampling_rate: int = 100,
    range_min=0,
    range_max="max",
    use_subject_two_selection=False,
):
    # /////////////////////////////////////////////////////////////////////////////////////////////
    # Preparation
    # /////////////////////////////////////////////////////////////////////////////////////////////

    try:
        output_dir = Path(output_path)
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        log.error(f"Output directory already exists: {output_path}")
        sys.exit(1)

    if targets is None:
        with open("data/HP_VariaNTS_intersection.txt", "r") as f:
            targets = f.read().split(",")
    elif type(targets) == str:
        targets = targets.split(",")
    console.log(f"Target words: {targets}")

    # NOTE When working on different datasets, these have to be adjusted.
    # Currently, these are only being used for subject 2.
    if use_subject_two_selection:
        selected_electrodes = np.concatenate([np.arange(8) + 16 * i for i in range(0, 6)])
        console.log(f"Selecting {len(selected_electrodes)} electrodes.")
    else:
        selected_electrodes = None
        console.log("Using all electrodes")

    # Make sure all data for all runs is provided
    timestamps = timestamps.split(",")
    audio = audio.split(",")
    ecog = ecog.split(",")
    assert len(timestamps) == len(audio) == len(ecog)

    # /////////////////////////////////////////////////////////////////////////////////////////////
    # Timestamps
    # /////////////////////////////////////////////////////////////////////////////////////////////

    intervals = []

    for run, path in enumerate(timestamps, start=1):
        ext = os.path.splitext(path)[1]
        if ext == ".TextGrid":
            for tg_interval in TextGrid(path)["words"]:
                if tg_interval.text in targets:
                    interval = tg_interval.__dict__
                    interval["run"] = run
                    intervals.append(interval)

        elif ext == ".csv":
            df = pd.read_csv(path, usecols=["text", "xmin", "xmax"])
            df = df.loc[df["text"].isin(targets)]
            run_intervals = df.to_dict("records")
            for i in range(len(run_intervals)):
                run_intervals[i]["run"] = run
            intervals.extend(run_intervals)

        else:
            log.error(f"Unsupported file format for timestamps: {ext}")
            sys.exit(1)

    # /////////////////////////////////////////////////////////////////////////////////////////////
    # Audio files
    # /////////////////////////////////////////////////////////////////////////////////////////////
    audios = []
    for run, path in enumerate(audio):
        audio_sr, audio_signal = wavread(path)
        audio_mean = np.mean(audio_signal)
        audios.append((audio_sr, audio_signal, audio_mean))

    # /////////////////////////////////////////////////////////////////////////////////////////////
    # ECoG files
    # /////////////////////////////////////////////////////////////////////////////////////////////
    ecogs = []
    for run, path in enumerate(ecog):
        ecog = np.load(path)
        ecogs.append(ecog)

    # /////////////////////////////////////////////////////////////////////////////////////////////
    # Creating paired fragments
    # /////////////////////////////////////////////////////////////////////////////////////////////
    word_counter = defaultdict(int)
    # Find length of longest interval, which we will use as length for all if range_max == "max"
    max_len = round(np.max([interval["xmax"] - interval["xmin"] for interval in intervals]), 4)
    console.log(f"Longest interval (seconds): {max_len}")
    console.log("Processing Intervals")

    for interval in track(intervals, description="Processing Intervals", transient=True):
        word_counter[interval["text"]] += 1

        ecog_begin = int(interval["xmin"] * sampling_rate) + int(range_min * sampling_rate)
        # Either use the longest fragment length as the length for every fragment, or use the user-
        # specified maximum range
        if range_max == "max":
            ecog_end = int(interval["xmin"] * sampling_rate) + int(max_len * sampling_rate) + 1
        else:
            ecog_end = (
                int(interval["xmin"] * sampling_rate) + int(float(range_max) * sampling_rate) + 1
            )
        ecog = ecogs[interval["run"] - 1][ecog_begin:ecog_end]
        # -> Shape (TIMESTEPS, ELECTRODES)

        if selected_electrodes is not None:
            ecog = ecog[:, selected_electrodes]

        # Transpose to shape (E, T)
        ecog = ecog.T

        # Chop section out of audio file
        audio_samplerate, audio, audio_mean = audios[interval["run"] - 1]
        audio_interval = audio[
            int(interval["xmin"] * audio_samplerate) : int(interval["xmax"] * audio_samplerate)
        ]

        # Align audio to (around) mean 0 and normalize peak value to Int16 range
        audio_interval = audio_interval.astype(np.float32)
        audio_interval = audio_interval - audio_mean
        audio_interval = audio_interval / np.max(np.abs(audio_interval)) * 32767
        audio_interval = audio_interval.astype(np.int16)

        output_name = f'{interval["text"]}{word_counter[interval["text"]]}'

        # Save interval audio as wavfile
        wavwrite(
            output_dir / f"{output_name}.wav",
            rate=audio_samplerate,
            data=audio_interval,
        )

        # Save ECoG data as numpy array
        np.save(output_dir / f"{output_name}.npy", ecog)

    console.log(f"Unique words: {len(word_counter)}")
    console.log(f"Total words: {len(intervals)}")

    # /////////////////////////////////////////////////////////////////////////////////////////////
    # Collecting info
    # /////////////////////////////////////////////////////////////////////////////////////////////
    console.log("Writing dataset information to file.")

    word_counter = dict(reversed(sorted(word_counter.items(), key=lambda item: item[1])))
    with open(output_dir / "about.txt", "w") as f:
        f.write(f"Longest interval (milliseconds): {max_len}\n")
        f.write(f"Unique words: {len(word_counter)}\n")
        f.write(f"Total words: {len(intervals)}\n")

        f.write("\nWord counts:\n-------")
        for word, count in word_counter.items():
            f.write(f"\n{word}: {count}")

    console.log("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecog", "-ei", type=str)
    parser.add_argument("--audio", "-ai", type=str)
    parser.add_argument("--timestamps", "-ti", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    parser.add_argument("--targets", "-t", type=str)
    parser.add_argument("--sampling_rate", "-sr", type=int, default=100)
    parser.add_argument("--range_min", "-min", type=float, default=0)
    parser.add_argument("--range_max", "-max", default="max")
    parser.add_argument("--use_subject_two_selection", type=bool, default=False)

    main(**vars(parser.parse_args()))
