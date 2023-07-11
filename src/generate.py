# -----------------------------------------------------------------------------
#
# Generate samples from a trained diffusion model, with conditional input or
# fully unconditional. Can be run in the distributed setting with several GPUs
# for faster performance.
#
# For sampling, the `Sampler` object is utilized (refer to `sampler.py` for
# details).
#
# Configuration options are loaded via the `configs/config.yaml` file, and can
# be overwritten when calling this script. Please see this repository's main
# `README.md` for a detailed explanation of how to use this script.
#
# -----------------------------------------------------------------------------

from functools import partial
import multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from sampler import Sampler


def generate(
    rank: int,
    diffusion_cfg: DictConfig,
    model_cfg: DictConfig,
    dataset_cfg: DictConfig,
    generate_cfg: DictConfig,
) -> None:
    ckpt_epoch = generate_cfg.pop("ckpt_epoch")
    sampler = Sampler(
        rank=rank,
        diffusion_cfg=diffusion_cfg,
        dataset_cfg=dataset_cfg,
        **generate_cfg,
    )
    sampler.run(ckpt_epoch, model_cfg)


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=generate_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
