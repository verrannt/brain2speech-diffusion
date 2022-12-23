from omegaconf import DictConfig

from .diffwave_conditional import DiffWaveConditional
from .diffwave import DiffWave
from .utils import *

def construct_model(model_cfg: DictConfig):
    """
    Construct a model given a configuration dict. 
    """

    # For an unconditional model, use standalone DiffWave model
    if model_cfg["unconditional"] == True:
        return DiffWave(**model_cfg)
    
    # Else use DiffWaveConditional, that wraps a conditional encoder model and
    # DiffWave as the decoder/generator model 
    else:
        # Separate the encoder config, because we use the config for DiffWave
        encoder_cfg = model_cfg.pop('encoder_config')
        
        # DiffWave needs to know the no. of output channels of the encoder to
        # correctly initialize its layers
        model_cfg['conditioner_channels'] = encoder_cfg['c_out']

        return DiffWaveConditional(encoder_cfg, model_cfg)