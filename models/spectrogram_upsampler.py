import torch
import torch.nn as nn
import torch.functional as F

from .utils import Conv

class SpectrogramUpsampler(nn.Module):
    def __init__(self, res_channels, mel_upsample=[16,16]) -> None:
        super().__init__()
              
        s = mel_upsample[0]
        self.conv1 = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
        self.conv1 = nn.utils.weight_norm(self.conv1)
        nn.init.kaiming_normal_(self.conv1.weight)

        s = mel_upsample[1]
        self.conv2 = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
        self.conv2 = nn.utils.weight_norm(self.conv2)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.mel_conv = Conv(80, 2 * res_channels, kernel_size=1)  # 80 is mel bands

    def forward(self, x, L):
        # Upsample spectrogram to size of audio
        x = torch.unsqueeze(x, dim=1)
        x = F.leaky_relu(self.conv1(x), 0.4)
        x = F.leaky_relu(self.conv2(x), 0.4)
        x = torch.squeeze(x, dim=1)

        assert(x.size(2) >= L)
        if x.size(2) > L:
            x = x[:, :, :L]

        x = self.mel_conv(x)
        return x
