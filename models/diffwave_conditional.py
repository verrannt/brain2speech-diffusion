from typing import Any, Mapping, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn

from .diffwave import DiffWave


class DiffWaveConditional(nn.Module):
    """
    Conditional wrapper model, to train and infer an encoder model and DiffWave together in a single model.
    """

    def __init__(self, encoder_class: nn.Module, encoder_cfg: DictConfig, decoder_cfg: DictConfig) -> None:
        super().__init__()
        
        self.encoder: nn.Module = encoder_class(**encoder_cfg)
        self.speech_generator = DiffWave(**decoder_cfg)

    def forward(self, x, conditional_input):
        # Global projection of conditional input, before it's fed into generator
        conditional_input = self.encoder(conditional_input)

        # Generator takes both diffusion input x_t and global conditional input
        x = self.speech_generator(x, conditional_input)

        return x

    def load_state_dict(
        self, 
        generator_state_dict: Mapping[str, Any], 
        encoder_state_dict: Optional[Mapping[str, Any]] = None,
        strict: bool = True
    ):
        self.speech_generator.load_state_dict(generator_state_dict, strict)

        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict, strict)

    def load_pretrained_generator(
        self,
        checkpoint_dict: Mapping[str, Mapping[str, Any]],
        freeze: bool = True,
    ):
        # The keys for the local conditioner will always be missing from an unconditional pretraining model, so we
        # set strict==False such that missing keys are ignored.
        inc_keys = self.speech_generator.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        
        # In any experiment setting, there should never be keys we don't expect
        assert len(inc_keys.unexpected_keys) == 0, \
            f'Found unexpected keys: {inc_keys.unexpected_keys}'
        
        # Special case: if the pretraining model was fully unconditional, the 'local_conditioner' part of DiffWave's
        # residual layers will not have been initialized/trained and thus won't be in the state keys.
        assert all(['local_conditioner' in k for k in inc_keys.missing_keys]), \
            f'Found missing keys for layers other than the local conditioner: {inc_keys.missing_keys}'

        print('Pretrained generator loaded successfully')

        if freeze:
            self.speech_generator.requires_grad_(False)
            print('Generator model frozen')
            
            # Same as above: for an unconditional pretraining model, we have to leave the residual layer's local 
            # conditioners unfrozen, as they have not been trained yet
            if any(['local_conditioner' in k for k in inc_keys.missing_keys]):
                print('Local conditioner not in state dict, will be unfrozen')
                for l in self.speech_generator.residual_layers:
                    l.local_conditioner.requires_grad_(True)


        if self.encoder.__class__.__name__ == 'BrainClassEncoder':
            
            # If using the BrainClassEncoder, need to also load the conditioner part of the network, as it was trained
            # together with the speech generator (i.e. the class-conditional setting). This will only load the 
            # class-conditioner part of the encoder, *not* the brain classifier part. Note that the class_conditioner 
            # should exactly resemble the class encoder (conditioner) in the pretraining model, so we have no tolerance 
            # for key mismatches and set strict==True
            self.encoder.class_conditioner.load_state_dict(checkpoint_dict['conditioner_state_dict'], strict=True)

            # If freezing of the generator is desired, we also freeze the class-conditioner part of the network, as 
            # they were trained together
            if freeze:
                self.encoder.class_conditioner.requires_grad_(False)
                print('Class conditioner part of BrainClassEncoder frozen')


    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def generator_state_dict(self):
        return self.speech_generator.state_dict()
