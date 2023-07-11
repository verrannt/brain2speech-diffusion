import os
import sys
from typing import List, Tuple, Union, Dict, Any

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor


class MaskedMSELoss:
    """
    Compute Mean-Squared-Error loss between inputs and targets. Takes a boolean
    mask to keep masked-out values from contributing to the loss and thus
    gradient updates.

    The mask is expected to be of same shape as input and target, and be True
    for those values that are supposed to contribute to the loss, and False
    for those values that are supposed to be neglected.

    If the mask is None, it will simply compute the standard Mean-Squared-Error.
    """

    def __init__(self):
        # Under the hood, we use torch's MSELoss without reduction, such that
        # we can apply our own reduction based on the mask we're getting
        self.loss_fn = torch.nn.MSELoss(reduction="none")

    def __call__(self, input, target, mask=None):
        # Obtain the element-wise loss value using the non-reduced MSELoss
        loss_val = self.loss_fn(input, target)

        if mask is not None:
            # Replace all masked-out values with 0. Note that the mask is False for
            # parts that were added as padding. Hence we have to invert the mask
            # for use in the `masked_fill` function
            masked_loss = loss_val.masked_fill(~mask, 0.0)

            # For the final loss value, we sum the values in our masked loss, and
            # divide by the amount of True values in the mask. This way, we ensure
            # that only the unmasked values contribute to the final loss.
            final_loss = masked_loss.sum() / mask.sum()

            return final_loss

        else:
            # If no mask is given we simply return the standard MSELoss by
            # taking the mean
            return loss_val.mean()


class HidePrints:
    """
    Context that suppresses all outputs written to `stdout`. Useful for multiprocessing, when
    only one process should print to `stdout`.

    Does not suppress raised exceptions.

    Parameters
    ----------
    hide:
        Whether to hide or print outputs

    Usage
    -----
    ```
    print('This will be printed')

    with HidePrints(True):
        print('This will *not* be printed')

    print('This will again be printed')

    with HidePrints(False):
        print('This will also be printed')
    ```
    """

    def __init__(self, hide: bool = True) -> None:
        self.hide = hide

    def __enter__(self):
        if self.hide:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def find_max_epoch(path: str) -> int:
    """
    Find maximum epoch/iteration in path, formatted as ${n_epoch}.pkl, e.g. 100.pkl

    Parameters
    ----------
    path:
        Checkpoint path

    Returns
    -------
    The highest epoch checkpoint found, or -1 if there is no (valid) checkpoint.
    """
    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f.endswith(".pkl"):
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net: torch.nn.Module, verbose: bool = False) -> None:
    """
    Print the number of parameters of a network

    Parameters
    ----------
    net:
        The PyTorch network
    verbose:
        Whether to print the parameters of each of the network's layers
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        module_parameters = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        if verbose:
            for n, p in module_parameters:
                print(n, p.numel())

        params = sum([np.prod(p.size()) for n, p in module_parameters])
        print("{} Parameters: {:.6f}M".format(net.__class__.__name__, params / 1e6), flush=True)


def create_output_directory(name: str, sub_directory: str) -> str:
    """
    Create the experiment output directory on disk.

    Parameters
    ----------
    name:
        Name of the experiment
    sub_directory:
        The specific output directory needed. Usually 'checkpoints' for model checkpoints or
        'waveforms' for generated samples.

    Returns
    -------
    Name of the generated output directory
    """
    # Create output directory if it doesn't exist
    output_directory = os.path.join("exp", name, sub_directory)
    os.makedirs(output_directory, mode=0o775, exist_ok=True)

    return output_directory


def prepend_data_base_dir(dataset_cfg: DictConfig) -> DictConfig:
    """
    Prepend the base path of where all data is stored on disk to the different data paths that are
    in the `dataset_cfg`.

    Parameters
    ----------
    dataset_cfg:
        The dataset configuration dict to modify

    Returns
    -------
    The modified dataset configuration dict
    """
    dataset_cfg.audio_path = os.path.join(dataset_cfg.data_base_dir, dataset_cfg.audio_path)
    dataset_cfg.splits_path = os.path.join(dataset_cfg.data_base_dir, dataset_cfg.splits_path)
    if "ecog_path" in dataset_cfg:
        dataset_cfg.ecog_path = os.path.join(dataset_cfg.data_base_dir, dataset_cfg.ecog_path)
    if "ecog_splits_path" in dataset_cfg:
        dataset_cfg.ecog_splits_path = os.path.join(
            dataset_cfg.data_base_dir, dataset_cfg.ecog_splits_path
        )
    return dataset_cfg


def calc_diffusion_hyperparams(
    T: int, beta_0: float, beta_T: float, beta: List[float] = None, fast: bool = False
) -> Dict[str, Any]:
    """
    Compute hyperparameters of the diffusion process and move them onto the current GPU.

    Parameters
    ----------
    T:
        number of diffusion steps
    beta_0:
        Starting value of the beta schedule
    beta_T:
        Ending value of the beta schedule
    beta:
        A full beta schedule that will be used instead of linearly interpolating between beta_0
        and beta_T if `fast==True`
    fast:
        Whether to use the schedule defined in `beta`

    Returns
    -------
    A dictionary of diffusion hyperparameters including:
        T (int), Beta, Alpha, and Alpha_bar (torch GPU Tensors, shape=[T,]), and Sigma (torch CPU
        Tensor, shape=[T, ])

    Raises
    ------
    ValueError:
        If `fast==True` but no beta schedule is provided
    """

    if fast:
        if beta is None:
            raise ValueError("The fast option is selected but no full beta schedule is given.")
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)

    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0

    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
            1 - Alpha_bar[t]
        )  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)

    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {
        "T": T,
        "Beta": Beta.cuda(),
        "Alpha": Alpha.cuda(),
        "Alpha_bar": Alpha_bar.cuda(),
        "Sigma": Sigma,
    }

    return _dh
