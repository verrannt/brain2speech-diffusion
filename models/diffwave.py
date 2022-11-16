import math

import torch
import torch.nn as nn

from models.utils import *



class DiffusionEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim: int = 128,
        mid_dim: int = 512,
        out_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, diffusion_steps):
        # Embed diffusion step t
        x = calc_diffusion_step_embedding(
            diffusion_steps, self.input_dim
        )

        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        res_channels: int,
        skip_channels: int,
        dilation: int = 1,
        diffusion_step_embed_dim_out: int = 512,
        unconditional: bool = True,
        conditioner_channels: int = 128,
    ):
        super(ResidualBlock, self).__init__()
        self.res_channels = res_channels

        # Layer-specific fully-connected layer for diffusion step embedding
        self.layer_diffusion_embed = nn.Linear(diffusion_step_embed_dim_out, res_channels)

        # Dilated conv layer
        self.dilated_conv = Conv(
            res_channels, 2 * self.res_channels, 
            kernel_size=3, 
            dilation=dilation
        )

        if unconditional:
            self.local_conditioner = None
        else:
            # Layer-specific 1x1 convolution for conditional input embedding
            self.local_conditioner = Conv(conditioner_channels, 2*res_channels, kernel_size=1)

        # Residual conv1x1 layer, connect to next residual layer
        self.res_conv = Conv(res_channels, res_channels, kernel_size=1)

        # Skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = Conv(res_channels, skip_channels, kernel_size=1)

    def forward(self, x, diffusion_step_embed, conditional_input=None):
        B, C, L = x.shape
        assert C == self.res_channels

        # Add diffusion step embedding
        diffusion_step_embed = self.layer_diffusion_embed(diffusion_step_embed)
        diffusion_step_embed = diffusion_step_embed.view([B, self.res_channels, 1])
        h = x + diffusion_step_embed

        # Dilated convolution
        h = self.dilated_conv(h)

        # Add (local) conditioner signal. Like in the original DiffWave paper, where the conditional signal are
        # embeddings of the Mel feature spectrograms, we use the local conditioner projection and add its output
        # to the hidden states.
        if conditional_input is not None:
            assert self.local_conditioner is not None
            
            conditional_input = self.local_conditioner(conditional_input)
            
            assert conditional_input.size(2) >= L
            if conditional_input.size(2) > L:
                conditional_input = conditional_input[:,:,:L]
            
            h = h + conditional_input

        # Gated tanh nonlinearity (similar to LSTMs)
        gate, filter = h[:,self.res_channels:,:], h[:,:self.res_channels,:]
        out = torch.sigmoid(gate) * torch.tanh(filter)

        # Residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        # Normalize for training stability
        return (x + res) * math.sqrt(0.5), skip


class DiffWave(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        res_channels: int = 256, 
        skip_channels: int = 128, 
        out_channels: int = 1,
        num_res_layers: int = 30, 
        dilation_cycle: int = 10,
        diffusion_step_embed_dim_in: int = 128,
        diffusion_step_embed_dim_mid: int = 512,
        diffusion_step_embed_dim_out: int = 512,
        unconditional: bool = False,
        conditioner_channels: int = 128,
        **kwargs
    ):
        super().__init__()
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.num_res_layers = num_res_layers

        # Initial conv1x1 with relu
        self.init_conv = nn.Sequential(
            Conv(in_channels, res_channels, kernel_size=1),
            nn.ReLU()
        )

        # Diffusion step embedding
        self.diffusion_embedding = DiffusionEmbedding(
            diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out
        )

        self.unconditional = unconditional

        # All residual layers with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                res_channels, skip_channels,
                dilation=2 ** (n % dilation_cycle),
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                unconditional=unconditional, conditioner_channels=conditioner_channels,
            ) for n in range(self.num_res_layers)
        ])

        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels)
        )

    def forward(self, input_data, conditional_input=None):
        x, diffusion_steps = input_data

        x = self.init_conv(x)
        diffusion_step_embed = self.diffusion_embedding(diffusion_steps)

        # if conditional_input is not None:
        #     assert self.global_conditioner is not None, \
        #         "Model is configured without a conditional input model, and " \
        #         "therefore cannot receive conditional inputs."

        #     # Encode conditional input before it's fed to the residual layers
        #     conditional_input = self.global_conditioner(conditional_input)

        skip = 0
        for layer in self.residual_layers:
            
            # Use the output from a residual layer as input to the next one
            x, skip_n = layer(x, diffusion_step_embed, conditional_input)
            
            # Accumulate all skip outputs
            skip = skip + skip_n

        # Normalize for training stability
        x = skip * math.sqrt(1.0 / self.num_res_layers)

        x = self.final_conv(x)

        return x

    def __repr__(self):
        return f"DiffWave_h{self.res_channels}_d{self.num_res_layers}_"\
               f"{'uncond' if self.unconditional else 'cond'}"
