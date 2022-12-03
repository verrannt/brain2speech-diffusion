from typing import Tuple

import torch
from torch import Tensor


def fix_length_1d(tensor: Tensor, desired_length: int) -> Tuple[Tensor, Tensor]:
    """
    Fix the length of `tensor` to `desired_length`. Returns both the
    altered tensor as well as a boolean mask tensor that depicts how much
    has been padded to the original tensor. This is useful for excluding
    padded regions from the loss computation.

    Expects an input `tensor` of shape `(1,n)`.


    Example
    -------
    If `sample_length == 6`, but `tensor.shape[1] == 4`, this function will
    return a tensor with shape `(1,6)`, where `tensor[:,4:6]` is filled with
    zeros, and a mask `[[True,True,True,True,False,False]]`.

    If `sample_length >= tensor.shape[1]`, this will return the tensor cut off
    at `sample_length` and a mask of only `True` values.

    """
    assert len(tensor.shape) == 2 and tensor.shape[0] == 1

    if tensor.shape[1] > desired_length:
        # If tensor is longer than desired length, the mask is only True
        # values
        mask = torch.ones((1,desired_length), dtype=torch.bool)

        return tensor[:, :desired_length], mask
    
    elif tensor.shape[1] < desired_length:
        # If tensor is shorter than desired length, the mask is True until
        # the original length of the tensor, and False afterwards
        mask = torch.zeros((1,desired_length), dtype=torch.bool)
        mask[:,:tensor.shape[1]] = True
        
        # We pad the tensor with zero values to increase its size to the
        # desired length
        padded_tensor = torch.cat([
            tensor, 
            torch.zeros(1, desired_length-tensor.shape[1])
        ], dim=1)

        return padded_tensor, mask
    
    else:
        # In case the tensor has already the desired length, we use a mask
        # of only True values again and can return the tensor unaltered
        mask = torch.ones((1,desired_length), dtype=torch.bool)
        return tensor, mask
    
def fix_length_3d(tensor: Tensor, desired_length: int) -> Tensor:
    """
    Fix the length of `tensor` to `desired_length` in dimension 2.

    Expects a 3d tensor.
    """

    assert len(tensor.shape) == 3

    if tensor.shape[2] > desired_length:
        return tensor[:, :, :desired_length]

    elif tensor.shape[2] < desired_length:
        return torch.cat([
            tensor, 
            torch.zeros(tensor.shape[0], tensor.shape[1], desired_length - tensor.shape[2])
        ], dim=2)

    else:
        return tensor

def standardize(tensor: Tensor) -> Tensor:
    """
    Standardize an input tensor to mean 0 and standard deviation 1.
    """
    return (tensor - torch.mean(tensor)) / torch.std(tensor)


def standardize_eeg(tensor: Tensor) -> Tensor:
    """
    Standardize EEG input tensor to mean 0 and standard deviation 1 along the last dimension (time).
    """
    mean = torch.mean(tensor, dim=2).unsqueeze(2).expand(-1,-1, tensor.shape[2])
    std = torch.std(tensor, dim=2).unsqueeze(2).expand(-1,-1, tensor.shape[2])
    return (tensor - mean) / std
