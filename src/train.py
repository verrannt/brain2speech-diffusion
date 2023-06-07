# -----------------------------------------------------------------------------
#
# Main training script for this repository. Adapted from:
#   https://github.com/albertfgu/diffwave-sashimi/blob/master/train.py
# 
# Allows for single-GPU as well as distributed training across many GPUs. For
# training, the `Learner` object is utilized (refer to `learner.py` for 
# details).
# 
# Configuration options are loaded via the `configs/config.yaml` file, and can
# be overwritten when calling this script. Please see this repository's main
# `README.md` for a detailed explanation of how to use this script.
#
# -----------------------------------------------------------------------------


from functools import partial
import multiprocessing as mp
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from distributed_util import init_distributed
from learner import Learner
from utils import HidePrints


def train(rank: int, num_gpus: int, group_name: str, cfg: DictConfig) -> None:
    """
    Trains model using the `Learner` class with the configurations given. If `num_gpus > 1`, initializes distributed
    processing to allow training of a model on several GPUs.

    Parameters
    ----------
    rank: 
        Rank of the GPU this function instance is called on
    num_gpus: 
        Total number of GPUs available in this run
    group_name: 
        Identifier of the process group
    cfg: 
        Configuration options defined in the Hydra config file
    """

    print(f'GPU {rank}: Training started')

    with HidePrints(rank != 0): # Suppress print outputs for all except master GPU
        # Init distributed processing if more than one GPUs are available
        if num_gpus > 1:
            dist_cfg = cfg.pop('distributed')
            init_distributed(rank, num_gpus, group_name, **dist_cfg)

        # Running training
        train_cfg = cfg.pop('train')
        learner = Learner(cfg, num_gpus, rank, **train_cfg)
        learner.train()

    print(f'\nGPU {rank}: Training finished')


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False) # Allow writing keys

    num_gpus = torch.cuda.device_count()

    train_fn = partial(
        train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
