from omegaconf import DictConfig

from .diffwave import DiffWave

def construct_model(model_cfg: DictConfig):
    """
    Construct a model given a configuration dict. 
    
    NOTE: Currently always returns DiffWave, until extended models are supported.
    """
    return DiffWave(**model_cfg)