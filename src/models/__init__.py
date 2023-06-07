from omegaconf import DictConfig

from .brain_encoder import BrainEncoder
from .class_encoder import ClassEncoder
from .brain_class_encoder import BrainClassEncoder
from .diffwave_conditional import DiffWaveConditional
from .diffwave import DiffWave
from . import utils


def construct_model(model_cfg: DictConfig):
    """
    Construct a model given a configuration dict. 
    """

    # For an unconditional model, use standalone DiffWave model
    if model_cfg["unconditional"] == True:
        return DiffWave(**model_cfg)
    
    # Else use DiffWaveConditional, which wraps a conditional encoder model and
    # DiffWave as the decoder/generator model 
    else:
        # Separate the encoder config, because we use the config for DiffWave
        encoder_cfg = model_cfg.pop('encoder_config')

        # DiffWave needs to know the no. of output channels of the encoder to
        # correctly initialize its layers
        model_cfg['conditioner_channels'] = encoder_cfg['c_out']
        
        encoder_name = encoder_cfg.name
        if encoder_name == 'brain_encoder':
            encoder_class = BrainEncoder
        elif encoder_name == 'class_encoder':
            encoder_class = ClassEncoder
        elif encoder_name == 'brain_class_encoder':
            encoder_class = BrainClassEncoder
        else:
            raise ValueError(f'Unknown conditional encoder: {encoder_name}')

        return DiffWaveConditional(encoder_class, encoder_cfg, model_cfg)