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
        self.network = nn.Sequential(
            # Temporal filtering
            nn.Conv2d(2, 32, 3, (1,2), 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            
            nn.Conv2d(32, 64, 3, (1,2), 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            
            nn.Conv2d(64, 128, 3, (1,2), 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            
            # Spatio-temporal filtering
            nn.Conv2d(128, 256, 3, (2,3), 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            
            nn.Conv2d(256, 256, 3, (2,3), 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            
            nn.MaxPool2d(2, 2),
            
            # Flatten
            nn.Flatten(),

            # Classification
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, n_classes),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x


# class BrainClassifierV3(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.conv1 = Conv2D(2, 64, kernel_size=(1,3), stride=(1,3), padding=0)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

#         self.conv2 = Conv2D(64, 128, kernel_size=(3,3), stride=(2,2), padding=0)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

#         self.lin1 = nn.Linear(1152, 512)
#         self.lin2 = nn.Linear(512, 55)


#         self.relu = nn.ReLU(inplace=True)
#         self.softmax = nn.Softmax(1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.pool1(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.pool2(x)

#         x = x.flatten(1)

#         x = self.lin1(x)
#         x = self.relu(x)
#         x = self.lin2(x)
#         x = self.softmax(x)
        
#         return x
