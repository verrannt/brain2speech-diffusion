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


import multiprocessing as mp
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from distributed_util import init_distributed
from learner import Learner


def train_single(rank: int, num_gpus: int, cfg: DictConfig):
    """
    Train model using the `Learner` class with the configurations given.

    Parameters
    ---
    rank: Rank of the GPU this function instance is called on
    num_gpus: Total number of GPUs available in this run
    cfg: configuration options defined in the Hydra config file
    """
    train_cfg = cfg.pop('train')
    learner = Learner(cfg, num_gpus, rank, **train_cfg)
    learner.train()


def train_distributed(rank: int, num_gpus: int, group_name: str, cfg: DictConfig):
    """
    Wrapper that first appropriately initializes distributed training, and then calls the train function.

    Parameters
    ---
    rank: Rank of the GPU this function instance is called on
    num_gpus: Total number of GPUs available in this run
    group_name: Identifier of the process group
    cfg: configuration options defined in the Hydra config file
    """
    # Distributed running initialization
    dist_cfg = cfg.pop('distributed')
    init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train_single(rank, num_gpus, cfg)


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False) # Allow writing keys

    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        train_single(rank=0, num_gpus=num_gpus, cfg=cfg)
    else:
        group_name = time.strftime("%Y%m%d-%H%M%S")
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(
                target=train_distributed, 
                kwargs={
                    'rank': i, 
                    'num_gpus': num_gpus, 
                    'group_name': group_name, 
                    'cfg': cfg})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
