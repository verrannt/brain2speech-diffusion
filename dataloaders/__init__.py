from random import shuffle
import torch
from torch.utils.data.distributed import DistributedSampler

from .csv_dataset import CSVDataset
from .sc import SpeechCommands
from .mel2samp import Mel2Samp

def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    # TODO would be nice if unconditional was decoupled from dataset

    dataset_name = dataset_cfg.pop("_name_")

    if dataset_name == "variants":
        assert unconditional
        trainset = CSVDataset(
            path = dataset_cfg.data_path, 
            subset = "train", 
            sample_length = dataset_cfg.segment_length)
        valset = CSVDataset(
            path = dataset_cfg.data_path, 
            subset = "val", 
            sample_length = dataset_cfg.segment_length)
        testset = CSVDataset(
            path = dataset_cfg.data_path, 
            subset = "test", 
            sample_length = dataset_cfg.segment_length)
    
    elif dataset_name == "sc09":
        assert unconditional
        trainset = SpeechCommands(dataset_cfg.data_path)
    
    elif dataset_name == "ljspeech":
        assert not unconditional
        trainset = Mel2Samp(**dataset_cfg)
    
    dataset_cfg["_name_"] = dataset_name # Restore

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
        shuffle=True,
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
