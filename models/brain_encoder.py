import torch
import torch.nn as nn

"""
Input:
[B, C, E, T] (batch size, frequency bands (channels), electrodes, timesteps)

Output:
[B, C, L=16000]
"""


class BrainEncoder(nn.Module):
    def __init__(
        self, 
        c_in : int = 1,
        c_mid : int = 64,
        c_out : int = 128,
        stride_y : int = 12,
        pad_y : int = 8,
        stride_x : int = 4,
        pad_x : int = 2,
        **kwargs,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_mid, (3, 32), padding=(1, pad_y), stride=(1, stride_y + 2)),
            nn.ReLU(),
            nn.AvgPool2d((3, 1)),
            nn.ConvTranspose2d(c_mid, c_out, (3, 32), padding=(1, pad_y), stride=(1, stride_y)),
            nn.ReLU(),
            nn.AvgPool2d((7, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x, 2) # (B, C, E, T) -> (B, C, T)
        return x
