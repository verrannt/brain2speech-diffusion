"""
USAGE:
-----

Note that for the pretraining models, the value of use_val does not matter, as
there is no train/validation split when there is no brain conditional input. 
The argument still has to be provided, else an exception is thrown by Hydra.

# Unconditional Pretraining
python brain2speech-diffusion/generate_several.py \
    experiment=pretraining_uncond_variants \
    generate.name=Uncond-PT-v8 \
    generate.conditional_type=None \
    generate.ckpt_epoch=250 \
    +use_val=False

# Classconditional Pretraining
python brain2speech-diffusion/generate_several.py \
    experiment=pretraining_class_cond_variants \
    generate.name=VariaNTSWords-CC-v2 \
    generate.conditional_type=class \
    generate.ckpt_epoch=270 \
    +use_val=False

# Brainconditional Finetuning with Harry Potter speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_cond_hp \
    generate.name=BrainCond-FT-HP-v4 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=110 \
    +use_val=False

# Brainconditional Finetuning with VariaNTS speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_cond_variants \
    generate.name=BrainCond-FT-VariaNTS-v2 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=800 \
    +use_val=False

# Brainclassconditional Finetuning with VariaNTS speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_class_cond_variants \
    generate.name=BrainClassCond-FT-VariaNTS-v7 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=800 \
    +use_val=False
"""


import os
import math

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dataloaders.conditional_loaders import get_word_from_filepath
from sampler import Sampler
from utils import HidePrints, create_output_directory


def get_files(conditional_type, split, n_samples_per_word=16):
    with open('data/HP_VariaNTS_intersection.txt', 'r') as f:
        words = f.read().split(',')

    if conditional_type == 'brain':
        return get_files_from_datasplit(split, n_samples_per_word)
    elif conditional_type == 'class':
        return [(word, n_samples_per_word) for word in words]
    elif conditional_type is None:
        return [(None, n_samples_per_word) for _ in words]
    else:
        raise ValueError('Unrecognized conditional input type:', conditional_type)

def get_files_from_datasplit(split, n_samples_per_word=16):
    # Read the files for the respective split
    ecog_splits = "/home/passch/data/datasplits/HP1_ECoG_conditional/sub-002"
    with open(os.path.join(ecog_splits, f"{split}.csv"), "r") as f:
        files = f.read().split(',')
        files = [fn.replace('.wav', '.npy') for fn in files]

    # Get unique words
    words = np.unique([get_word_from_filepath(fn) for fn in files])

    chosen_files = []
    
    # For each word ...
    for word in words:
        # Chose the files for this word
        word_files = [fn for fn in files if get_word_from_filepath(fn) == word]
        # Repeat the files at least `n_samples_per_word` times
        word_files = word_files * math.ceil(n_samples_per_word / len(word_files))
        # Cut off at `n_samples_per_word` and add to chosen files
        chosen_files.extend(word_files[:n_samples_per_word])

    # Assert that counts for each word == `n_samples_per_word`
    counts = np.unique([get_word_from_filepath(fn) for fn in chosen_files], return_counts=True)[1]
    assert all(counts == n_samples_per_word), f"Not all counts equal {n_samples_per_word}: {counts}"

    return list(zip(*np.unique(chosen_files, return_counts=True)))


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False) # Allow writing keys
    
    if cfg.model.unconditional == True:
        cfg.generate.conditional_type = None

    if cfg.use_val == True:
        files = get_files(cfg.generate.conditional_type, 'val', 16) # TODO
    else:
        files = get_files(cfg.generate.conditional_type, 'train', 16) # TODO
    
    print(files)

    data_path = "/home/passch/data/HP1_ECoG_conditional/sub-002" # Only needed for brain models

    ckpt_epoch = cfg.generate.pop('ckpt_epoch')

    sampler = Sampler(
        rank=0,
        diffusion_cfg=cfg.diffusion,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )
    experiment_name, _ = create_output_directory(
        sampler.name, cfg.model, sampler.diffusion_cfg, sampler.dataset_cfg, 'waveforms')

    model, ckpt_epoch = sampler.load_model(experiment_name, cfg.model, ckpt_epoch, sampler.conditional_signal is None)

    output_subdir = f"waveforms_{'val' if cfg.use_val==True else 'train'}"

    for i, (word, count) in enumerate(files):

        print('-'*80)
        print(f"{i+1}/{len(files)}: Running {word if word is not None else 'Any'} x{count}")

        if cfg.generate.conditional_type == 'brain':
            sampler.conditional_signal = os.path.join(data_path, word)
        elif cfg.generate.conditional_type == 'class':
            sampler.conditional_signal = word
        elif cfg.generate.conditional_type is None:
            sampler.conditional_signal = None
        else:
            raise ValueError('Unrecognized conditional type:', cfg.generate.conditional_type)
        sampler.n_samples = count
        sampler.batch_size = count

        sampler.run(ckpt_epoch, cfg.model, model=model, output_subdir=output_subdir)


if __name__=='__main__':
    main()