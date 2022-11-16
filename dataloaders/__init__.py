from random import shuffle
import torch
from torch.utils.data.distributed import DistributedSampler

from .csv_dataset import CSVDataset
from .eeg_dataset import EEGDataset


SHUFFLING_SEED = 1144


def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    dataset_name = dataset_cfg.pop("_name_")

    valset, testset = None, None

    if dataset_name == "variants":
        # assert unconditional TODO Undo commenting out
        trainset = CSVDataset(
            path = dataset_cfg.data_path,
            subset = "train",
            file_base_path = dataset_cfg.file_base_path,
            sample_length = dataset_cfg.segment_length,
            min_max_norm = dataset_cfg.get('min_max_norm', False))
        valset = CSVDataset(
            path = dataset_cfg.data_path,
            subset = "val",
            file_base_path = dataset_cfg.file_base_path,
            sample_length = dataset_cfg.segment_length,
            min_max_norm = dataset_cfg.get('min_max_norm', False))
        testset = CSVDataset(
            path = dataset_cfg.data_path,
            subset = "test",
            file_base_path = dataset_cfg.file_base_path,
            sample_length = dataset_cfg.segment_length,
            min_max_norm = dataset_cfg.get('min_max_norm', False))

    elif dataset_name == "brain_conditional":
        assert not unconditional # TODO do we really need this, or would be fine without?
        trainset = EEGDataset(
            path = dataset_cfg.data_path,
            subset = "train",
            sample_length = dataset_cfg.segment_length,
            sample_rate_audio = dataset_cfg.sample_rate_audio,
            sample_rate_eeg = dataset_cfg.sample_rate_eeg,
            seed = SHUFFLING_SEED)
        valset = EEGDataset(
            path = dataset_cfg.data_path,
            subset = "val",
            sample_length = dataset_cfg.segment_length,
            sample_rate_audio = dataset_cfg.sample_rate_audio,
            sample_rate_eeg = dataset_cfg.sample_rate_eeg,
            seed = SHUFFLING_SEED)
    
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
