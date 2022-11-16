import numpy as np
import torch
import torch.nn as nn

def calc_diffusion_step_embedding(diffusion_steps, dim):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    `[sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]`

    Parameters
    ---
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
        diffusion steps for batch data
    dim (int, default=128):
        dimensionality of the embedding space for discrete diffusion steps

    Returns
    ---
    the embedding vectors (torch.tensor, shape=(batchsize, dim)):
    """

    assert dim % 2 == 0

    half_dim = dim // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed

def swish(x):
    """
    Swish activation function with beta scaling factor = 1
    """
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    """
    Convolutional Layer with Kaiming Normal initialization, weight normalization
    and dilation.

    From https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size, dilation=dilation, padding=self.padding
        )
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    """
    1x1 Convolutional Layer with zero initialization.

    From https://github.com/ksw0306/FloWaveNet/blob/master/modules.py, but the scale parameter is removed.
    """

    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out
