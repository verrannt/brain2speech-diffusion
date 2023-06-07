"""
-----------------------------------------------------------------------------------------------------------------------

Merge weights from the pretrained class conditional model and the separately trained brain classifier into a single 
model for sampling. 

Note that the parameters specified in the `configs/model/brain_class_conditional.yaml` file must correspond to both the 
pretraining model's as well as brain classifier's parameters. The merged model is first constructed anew from this 
config, and then the pretraining and classifier weights are loaded into it. This will not work if the parameters do not 
align.


USAGE

python merge_brain_clf.py \
    --out_dir /home/passch/exp/ \
    --out_name BrainClassCond-FT-VariaNTS-v9 \
    --classifier /home/passch/exp/Brain-Classifier/Sub2_Full-Std_MLP_55classes_LR1e-05_dropout_layernorm_no-shuffle/800.pkl \
    --pretrained /home/passch/exp/VariaNTSWords-CC-v3_h256_d36_T200_betaT0.02_L1000_cond/checkpoint/180.pkl

-----------------------------------------------------------------------------------------------------------------------
""" 

import argparse
from pathlib import Path
import os
import sys

from omegaconf import OmegaConf
import torch

from models import construct_model


def main(args):
    output_dir = args.out_dir
    outp_model_name = args.out_name
    clf_path = args.classifier
    pret_path = args.pretrained

    outp_path = pret_path.split('/')[:-2][-1].split('_')[1:]
    outp_path = '_'.join(outp_path)
    outp_path = f'{outp_model_name}_{outp_path}/checkpoint'
    outp_path = os.path.join(output_dir, outp_path)

    os.makedirs(outp_path, exist_ok=True)

    outp_path = os.path.join(outp_path, clf_path.split('/')[-1]) # <epoch>.pkl

    if os.path.exists(outp_path):
        print('[ERRO] Output file already exists:', outp_path)
        sys.exit()


    # Construct BrainClassConditional model as you would for training 
    model_cfg = OmegaConf.load(Path.home() / "brain2speech-diffusion/configs/model/brain_class_conditional.yaml")
    model = construct_model(model_cfg)


    # Load pretrained generator from Classconditional pretraining checkpoint
    pret_ckpt = torch.load(pret_path, map_location='cpu')

    # Overwrite model with pretrained checkpoint. This loads both the speech generator as well as the class conditioner part
    # of the encoder network
    model.load_pretrained_generator(pret_ckpt, freeze=False)


    # Load separately trained brain classifier
    clf_ckpt = torch.load(clf_path, map_location='cpu')

    # Overwrite brain classifier part of the encoder with the brain classifier weights
    model.encoder.brain_classifier.load_state_dict(clf_ckpt['model_state_dict'], strict=True)
    print('Brain classifier weights loaded successfully')

    # Put the state dicts for generator and encoder into a final checkpoint dict
    final_ckpt = {}
    final_ckpt['model_state_dict'] = model.generator_state_dict()
    final_ckpt['conditioner_state_dict'] = model.encoder_state_dict()


    # Ensure that all values are the same
    for k in clf_ckpt['model_state_dict'].keys():
        assert (final_ckpt['conditioner_state_dict'][f'brain_classifier.{k}'] == clf_ckpt['model_state_dict'][k]).all()

    # Ensure that weights for the speech generator match the pretraining weights
    for k in pret_ckpt['model_state_dict']:
        assert (final_ckpt['model_state_dict'][k] == pret_ckpt['model_state_dict'][k]).all()

    # Ensure that weights for the class conditioner match the pretraining weights 
    for k in pret_ckpt['conditioner_state_dict']:
        # Since the fine-tuning model has both brain classifier and class conditioner as its conditioner model, we have to
        # add 'class_conditioner' to the layer ID
        assert (final_ckpt['conditioner_state_dict'][f'class_conditioner.{k}'] == pret_ckpt['conditioner_state_dict'][k]).all()

    print('All checks passed')


    torch.save(final_ckpt, outp_path)
    print('[SUCC] Combined model saved under', outp_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Merge pretrained classconditional model with brain classifier')
    parser.add_argument('--pretrained', type=str, help='Checkpoint of the class conditional pretraining model')
    parser.add_argument('--classifier', type=str, help='Checkpoint of the brain classifier')
    parser.add_argument('--out_dir', type=str, help='Directory to save merged weights to')
    parser.add_argument('--out_name', type=str, help='Name to create for the new model')
    args = parser.parse_args()

    main(args)