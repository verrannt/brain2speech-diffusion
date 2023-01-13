import torch
import torch.nn as nn

from .class_encoder import ClassEncoder
from .utils import Conv2D

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
        c_mid : int = 64,
        c_out : int = 128,
        **kwargs,
    ):
        super().__init__()

        # Different classifiers can be tested here by swapping out the class. Note that the required key-word arguments
        # have to be appropriately specified in the model config file.
        # self.brain_classifier = BrainClassifierV1(in_nodes=c_brain_in, mid_nodes=c_brain_mid, out_nodes=n_classes)
        self.brain_classifier = BrainClassifierV2(n_classes=n_classes)

        # The second part is identical to the class encoder, so it will be reused.
        self.class_conditioner = ClassEncoder(n_classes=n_classes, c_mid=c_mid, c_out=c_out)

    def forward(self, x):
        x = self.brain_classifier(x)
        x = x.unsqueeze(1)
        x = self.class_conditioner(x)
        return x


class BrainClassifierV1(nn.Module):
    def __init__(self, in_nodes: int, mid_nodes: int, out_nodes: int, **kwargs) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_nodes, mid_nodes)
        self.l2 = nn.Linear(mid_nodes, out_nodes)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input before processing. CxExT must be equal to IN_NODES
        # [B, C, E, T]
        x = x.reshape(x.size(0), -1)
        # [B, IN_NODES]
        x = self.l1(x)
        x = self.relu(x)
        # [B, MID_NODES]
        x = self.l2(x)
        x = self.softmax(x)
        # [B, OUT_NODES]
        return x


class BrainClassifierV2(nn.Module):
    def __init__(self, n_classes: int = 10, **kwargs) -> None:
        super().__init__()
        self.conv1 = Conv2D(2, 32, 3, (1,2), 1)
        self.conv2 = Conv2D(32, 64, 3, (1,2), 1)
        self.conv3 = Conv2D(64, 128, 3, (1,2), 1)

        self.conv4 = Conv2D(128, 256, 3, (2,3), 1)
        self.conv5 = Conv2D(256, 256, 3, (2,3), 1)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, n_classes)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal filtering
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        # Spatio-temporal filtering
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(1)
        
        # Classification
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)

        return x
