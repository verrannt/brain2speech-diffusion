from random import shuffle
from pathlib import Path

import torch
from torch.utils.data.distributed import DistributedSampler

from .csv_dataset import CSVDataset
from .eeg_dataset import EEGDataset
from .utils import *
from .conditional_loaders import EEGRandomLoader, EEGExactLoader


SHUFFLING_SEED = 1144


def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    dataset_name = dataset_cfg.pop("_name_")

    valset, testset = None, None

    # Convert segment length from milliseconds to frames
    segment_length_audio = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate / 1000)
    if not unconditional:
        segment_length_eeg = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate_eeg / 1000)
        eeg_path = Path(dataset_cfg.eeg_path)

    if dataset_name == "variants":
        assert unconditional
        conditional_loader = None

    elif dataset_name == "variants_brain":
        assert not unconditional
        # Use random conditional loader because we don't have a 1-to-1 matching between EEG and VariaNTS data
        conditional_loader = EEGRandomLoader(
            path = eeg_path,
            seed = SHUFFLING_SEED,
            segment_length = segment_length_eeg)
            
    elif dataset_name == "brain_conditional":
        assert not unconditional
        # Use exact conditional loader to get the right EEG matrix for every audio file
        conditional_loader = EEGExactLoader(
            path = eeg_path,
            segment_length = segment_length_eeg)


    trainset = CSVDataset(
        csv_path = dataset_cfg.splits_path,
        subset = "train",
        data_path = dataset_cfg.data_path,
        segment_length = segment_length_audio,
        seed=SHUFFLING_SEED,
        conditional_loader=conditional_loader)
    
    valset = CSVDataset(
        csv_path = dataset_cfg.splits_path,
        subset = "val",
        data_path = dataset_cfg.data_path,
        segment_length = segment_length_audio,
        seed=SHUFFLING_SEED,
        conditional_loader=conditional_loader)


    # Use distributed sampling for the train set. Note that we do not use
    # it for validation and testing set, since we have currently no way of
    # collecting the results from all GPUs, and will therefore only run them
    # on the first GPU.
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        # Using shuffle=True throws ValueError on Snellius when using
        # distributed training:
        # "sampler option is mutually exclusive with shuffle"
        # shuffle=True,
    )

    if valset:
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )
    else:
        valloader = None

    if testset:
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )
    else:
        testloader = None

    return trainloader, valloader, testloader
