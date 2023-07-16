import torch
import torch.nn as nn

"""
Input:
[B, E, T] (batch size, electrodes (channels), timesteps)

Output:
[B, C, L=16000]
"""


class BrainEncoder(nn.Module):
    def __init__(
        self,
        c_in: int = 32,
        c_mid: int = 64,
        c_out: int = 128,
        kernel_size: int = 18,
        padding: int = 3,
        stride: int = 6,
        **kwargs,
    ):
        super().__init__()

        conv1 = nn.ConvTranspose1d(c_in, c_mid, kernel_size, padding=padding, stride=stride + 2)
        conv1 = nn.utils.weight_norm(conv1)
        nn.init.kaiming_normal_(conv1.weight)

        conv2 = nn.ConvTranspose1d(c_mid, c_mid, kernel_size, padding=padding, stride=stride)
        conv2 = nn.utils.weight_norm(conv2)
        nn.init.kaiming_normal_(conv2.weight)

        conv3 = nn.ConvTranspose1d(c_mid, c_out, kernel_size, padding=padding, stride=stride)
        conv3 = nn.utils.weight_norm(conv3)
        nn.init.kaiming_normal_(conv3.weight)

        self.net = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x
