from os.path import join
from pathlib import Path
from typing import Tuple, Optional

from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .csv_dataset import CSVDataset
from .utils import *
from .conditional_loaders import ECOGRandomLoader, ECOGExactLoader, ClassConditionalLoader


SHUFFLING_SEED = 1144


def dataloader(
    dataset_cfg: DictConfig, 
    batch_size: int, 
    is_distributed: bool, 
    unconditional: bool = True, 
    uses_test_set: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Initialize dataloaders with given configuration options.

    Note that the dataset objects used by the dataloaders consist of two parts: the dataset itself, which loads audio 
    files from disk, and a conditional loader which loads corresponding conditional inputs, in case conditional training
    is enabled. The dataset object is the same for all datasets, but the conditional loaders vary:

    Dataset 1: VariaNTS (conditional or unconditional)
        The standard VariaNTS words dataset. Can be loaded unconditionally (just the audio data) or conditionally, in
        which case the conditional input is a one-hot encoded class vector obtained from the `ClassConditionalLoader`.

    Dataset 2: VariaNTS-Brain (always conditional)
        The VariaNTS word dataset, but paired with ECoG brain recordings. Since there is no 1-to-1 match between the
        two, for each word obtained from the VariaNTS dataset, a random ECoG recording corresponding to that word will
        be loaded. This is done using the `ECOGRandomLoader`.

    Dataset 3: Brain-Conditional (always conditional)
        The actual ECoG recording dataset which audio files matching exactly the brain recordings. For this dataset, 
        the `ECOGExactLoader` is used which fetches the correct ECoG recording for a given audio recording.

    Parameters
    ----------
    `dataset_cfg`:
        The configuration options for the dataset, loaded from a Hydra config file.
    `batch_size`:
        How many datapoints to yield at each iteration of the dataloader
    `is_distributed`:
        Whether this is a distributed training run or not. If `True`, training data will be subset such that the
        different GPUs get access to different parts of the dataset only.
    `unconditional`:
        Whether the model to be trained is a conditional or unconditional model. This determines how data is loaded for
        some datasets, or may cause an `AssertionError` if a dataset is incompatible with it.
    `uses_test_set`:
        Whether a test split is provided and a test dataloader should be created.
    
    Returns
    -------
    A tuple consisting of the train dataloader, the validation dataloader, and optionally the test dataloader if the 
    test set is enabled.

    Raises
    ------
    `AssertionError`
        if `unconditional==False` but the dataset name denoted in the `dataset_cfg` is 'brain_cond_variants' or 
        'brain_cond_hp', since both of these datasets require a conditional model.
    `AssertionError`
        if the dataset name denoted in the `dataset_cfg` is 'brain_cond_variants', but the `dataset_cfg` does not have the 
        'ecog_splits_path' key. This splits path is required since there is no 1-to-1 matching between the VariaNTS data
        and the ECoG recordings, so a separate train/val split has to be provided for the ECoG recordings.
    `ValueError`
        if the `_name_` specified in the `dataset_cfg` is unknown.
    """
    
    dataset_name = dataset_cfg.pop("_name_")

    testset = None

    # Convert segment length from milliseconds to frames
    segment_length_audio = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate / 1000)
    # For datasets using conditional brain inputs, need to also do processing for the ECoG files
    if dataset_name == "brain_cond_variants" or dataset_name == "brain_cond_hp":
        segment_length_ecog = int(dataset_cfg.segment_length * dataset_cfg.sampling_rate_ecog / 1000)
        ecog_path = Path(dataset_cfg.ecog_path)

    if dataset_name == "variants":
        if unconditional:
            conditional_loader = None
        else:
            conditional_loader = ClassConditionalLoader(
                words_file=join(dataset_cfg.data_base_dir, 'HP_VariaNTS_intersection.txt'))

    elif dataset_name == "brain_cond_variants":
        assert not unconditional
        # Use random conditional loader with separate train and val splits for the ECoG files, because we don't have a
        # 1-to-1 matching between ECoG and VariaNTS data
        assert 'ecog_splits_path' in dataset_cfg
        conditional_loader = ECOGRandomLoader(
            path = ecog_path,
            splits_path = dataset_cfg.ecog_splits_path,
            seed = SHUFFLING_SEED,
            segment_length = segment_length_ecog)
            
    elif dataset_name == "brain_cond_hp":
        assert not unconditional
        # Use exact conditional loader to get the right ECoG matrix for every audio file
        conditional_loader = ECOGExactLoader(
            path = ecog_path,
            segment_length = segment_length_ecog)

    else:
        raise ValueError(f'Unknown dataset specified: {dataset_name}')


    trainset = CSVDataset(
        csv_path = dataset_cfg.splits_path,
        subset = "train",
        audio_path = dataset_cfg.audio_path,
        segment_length = segment_length_audio,
        seed=SHUFFLING_SEED,
        conditional_loader=conditional_loader)
    
    valset = CSVDataset(
        csv_path = dataset_cfg.splits_path,
        subset = "val",
        audio_path = dataset_cfg.audio_path,
        segment_length = segment_length_audio,
        seed=SHUFFLING_SEED,
        conditional_loader=conditional_loader)
    
    if uses_test_set:
        testset = CSVDataset(
            csv_path = dataset_cfg.splits_path,
            subset = "test",
            audio_path = dataset_cfg.audio_path,
            segment_length = segment_length_audio,
            seed=SHUFFLING_SEED,
            conditional_loader=conditional_loader)


    # Use distributed sampling for the train set. Note that we do not use it for validation and testing set, since those 
    # are only run on the first GPU.
    train_sampler = DistributedSampler(trainset, shuffle=False) if is_distributed else None

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    if uses_test_set:
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )
    else:
        testloader = None

    return trainloader, valloader, testloader
