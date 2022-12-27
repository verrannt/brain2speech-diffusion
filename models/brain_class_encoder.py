import torch
import torch.nn as nn

"""
Input:
[B, C, E, T] (batch size, frequency bands (channels), electrodes, timesteps)

Output:
[B, C_OUT, L]
"""


class BrainClassEncoder(nn.Module):
    """
    Brain + class conditional encoder model. Takes brain (ECoG) recordings as input, and funnels them through a 
    classification bottleneck, before projecting the classification results up to the size of the audio sequence.
    
    This essentially trains two models, a first classification model that learns to classify brain input, and a second
    class conditioning model that learns a class conditioning signal suitable for DiffWave.
    
    In practice, this second model may be pretrained together with DiffWave in the class-conditional pretraining 
    setting.
    """

    def __init__(
        self, 
        n_classes : int = 10,
        c_brain_in: int = 2000,
        c_brain_mid: int = 500,
        c_mid : int = 64,
        c_out : int = 128,
        **kwargs,
    ):
        super().__init__()

        self.brain_classifier = nn.Sequential(
            # [B, C_BRAIN_IN]
            nn.Linear(c_brain_in, c_brain_mid),
            nn.ReLU(),
            # [B, C_BRAIN_MID]
            nn.Linear(c_brain_mid, n_classes),
            nn.Softmax(1),
            # [B, N_CLASSES]
        )

        # [B, N_CLASSES]
        self.embedding = nn.Linear(n_classes, 512, bias=False)
        # [B, 512]

        self.projection = nn.Sequential(
            # [B, 1, 512]
            nn.ConvTranspose1d(in_channels=1, out_channels=c_mid, kernel_size=3, padding=4, stride=8),
            # [B, C_MID, 4083]
            nn.ConvTranspose1d(in_channels=c_mid, out_channels=c_out, kernel_size=3, padding=4, stride=4),
            # [B, C_OUT, 16323]
        )

    def forward(self, x):
        print(x.shape)
        
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        
        x = self.brain_classifier(x)
        print(x.shape)
        
        x = self.embedding(x)
        print(x.shape)

        x = x.unsqueeze(1)
        print(x.shape)
        
        x = self.projection(x)
        print(x.shape)
        
        return x
