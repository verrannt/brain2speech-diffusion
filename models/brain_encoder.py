import torch
import torch.nn as nn

"""
Input:
[B, E, F, T] (batch size, electrodes (channels), frequency bands, timesteps)

Output:
[B, C, L=16000]
"""


class BrainEncoder(nn.Module):
    def __init__(
        self, 
        c_in : int = 32,
        c_mid : int = 64,
        c_out : int = 128,
        stride_y : int = 12,
        pad_y : int = 8,
        **kwargs,
    ):
        super().__init__()

        _kernel_size = 5

        conv1 = nn.ConvTranspose2d(c_in, c_mid, (1, _kernel_size), padding=(0, pad_y), stride=(1, stride_y))
        conv1 = nn.utils.weight_norm(conv1)
        nn.init.kaiming_normal_(conv1.weight)

        conv2 = nn.ConvTranspose2d(c_mid, c_out, (2, _kernel_size*2), padding=(1, pad_y+2), stride=(1, stride_y+2))
        conv2 = nn.utils.weight_norm(conv2)
        nn.init.kaiming_normal_(conv2.weight)

        self.net = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x, 2) # (B, C, 1, L) -> (B, C, L)
        return x
