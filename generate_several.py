"""
Note that for the pretraining models, the value of use_val does not matter, as
there is no train/validation split when there is no brain conditional input. 
The argument still has to be provided, else an exception is thrown by Hydra.

For the finetuning models, we need to decide how to generate data, since the 
distribution of the ECoG dataset is so uneven. Two options are to generate *n* 
audio samples per one ECoG sample (giving us an uneven distribution of audio 
samples), or to generate *k* audio samples *per* class, such that every word 
class has the same number of audio samples available. Here the latter option 
is chosen, as it ensures that the generated data match that of the pretraining 
models as well as that of the VariaNTS speech data.

# Classconditional Pretraining
python brain2speech-diffusion/generate_several.py \
    experiment=pretraining_class_cond_variants \
    generate.name=ClassCond-PT-v3 \
    generate.conditional_type=class \
    generate.ckpt_epoch=180 \
    +use_val=False

# Brainconditional Finetuning with Harry Potter speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_cond_hp \
    generate.name=BrainCond-FT-HP-v5 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=70 \
    +use_val=False

# Brainconditional Finetuning with VariaNTS speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_cond_variants \
    generate.name=BrainCond-FT-VariaNTS-v3 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=140 \
    +use_val=False

# Brainclassconditional Finetuning with VariaNTS speech data
python brain2speech-diffusion/generate_several.py \
    experiment=finetuning_brain_class_cond_variants \
    generate.name=BrainClassCond-FT-VariaNTS-v9 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=800 \
    +use_val=False

# Unconditional Pretraining
# Note: for the unconditional model, the standard generate.py script should be
# used, as no separation by word classes is required, and parallelization can
# therefore be used easily.
python brain2speech-diffusion/generate.py \
    experiment=pretraining_uncond_variants \
    generate.name=Uncond-PT-v9 \
    generate.conditional_type=null \
    generate.conditional_signal=null \
    generate.ckpt_epoch=230 \
    generate.n_samples=880 \
    generate.batch_size=16
"""


import os
import math

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dataloaders.conditional_loaders import get_word_from_filepath
from sampler import Sampler
from utils import create_output_directory


def get_files(conditional_type, split, data_base_dir, n_samples_per_word=16):
    if conditional_type == 'brain':
        data_path = os.path.join(data_base_dir, "HP1_ECoG_conditional/sub-002") 
        split_path = os.path.join(data_base_dir, f"datasplits/HP1_ECoG_conditional/sub-002/{split}.csv")
        return get_files_from_datasplit(split_path, n_samples_per_word), data_path

    with open(os.path.join(data_base_dir, "HP_VariaNTS_intersection.txt"), 'r') as f:
        words = f.read().split(',')
    
    if conditional_type == 'class':
        return [(word, n_samples_per_word) for word in words], None
    
    elif conditional_type is None:
        return [(None, n_samples_per_word) for _ in words], None
    
    else:
        raise ValueError('Unrecognized conditional input type:', conditional_type)

def get_files_from_datasplit(split_path, n_samples_per_word=16):
    # Read the files for the respective split
    with open(split_path, "r") as f:
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
        files, data_path = get_files(
            cfg.generate.conditional_type, 'val', 
            cfg.dataset.data_base_dir, 16)
    else:
        files, data_path = get_files(
            cfg.generate.conditional_type, 'train', 
            cfg.dataset.data_base_dir, 16)
    
    print(files)

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