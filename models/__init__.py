from omegaconf import DictConfig

from .diffbrain import DiffBrain
from .diffwave import DiffWave
from .brain_encoder import BrainEncoder
from .utils import *

def construct_model(model_cfg: DictConfig):
    """
    Construct a model given a configuration dict. 
    """

    if model_cfg["unconditional"] == True:
        return DiffWave(**model_cfg)
    else:
        encoder_cfg = model_cfg.pop('encoder_config')
        model_cfg['conditioner_channels'] = encoder_cfg['c_out']
        return DiffBrain(encoder_cfg, model_cfg)