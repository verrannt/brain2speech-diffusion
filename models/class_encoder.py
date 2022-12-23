import torch
import torch.nn as nn

"""
Input:
[B, 1]

Output:
[B, C, L]
"""


class ClassEncoder(nn.Module):
    def __init__(
        self, 
        n_classes : int = 10,
        c_mid : int = 64,
        c_out : int = 128,
        **kwargs,
    ):
        super().__init__()

        self.embedding = nn.Embedding(n_classes, 512)

        self.projection = nn.Sequential(
            # [B, 1, 512]
            nn.ConvTranspose1d(in_channels=1, out_channels=c_mid, kernel_size=3, padding=4, stride=8),
            # [B, C_MID, 4083]
            nn.ConvTranspose1d(in_channels=c_mid, out_channels=c_out, kernel_size=3, padding=4, stride=4),
            # [B, C_OUT, 16323]
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.projection(x)
        return x
