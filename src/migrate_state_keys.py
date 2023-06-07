# ----------------------------------------------------------------------------------------------------------------------
#
# Update model state keys of saved model checkpoints after changes to the model code have been made.
#
# To run this script, pass the path to the saved checkpoint that should be updated as first argument, and the path to
# the resolution dict as second argument:
#
# python migrate_state_keys.py path/to/model/checkpoint/epoch.pkl path/to/model_state_key_resolution.json
#
# This assumes that the resolution dict, which resolves old keys to new keys, has already been generated using the
# `create_key_resolution_dict` function, and stored as a JSON file. If not, this function has to be run first.
# 
# ----------------------------------------------------------------------------------------------------------------------


from collections import OrderedDict
import json
import os
import sys

import torch


def create_key_resolution_dict(old_model_path, new_model_path):
    """
    Create a model state key resolution dictionary given a checkpoint of a model with the old state keys, and of a
    model with the new state keys. It is assumed both correspond to the same model, with just differently named layers,
    otherwise this will not work.
    """

    # Load checkpoint files for model for which we want to change the keys
    original_model_dict = torch.load(old_model_path, map_location='cpu')['model_state_dict']

    # Load checkpoint files for model from which we take the new keys
    new_model_dict = torch.load(new_model_path, map_location='cpu')['model_state_dict']

    assert len(original_model_dict) == len(new_model_dict)

    key_translation_dict = {}

    for (k1, v1), (k2, v2) in zip(*[original_model_dict.items(), new_model_dict.items()]):
        key_translation_dict[k1] = k2
        # if k1 != k2:
        #   print('Old:', k1, ' '*(80-len(k1)), v1.shape)
        #   print('New:', k2, ' '*(80-len(k2)), v2.shape)

    # Save to disk
    with open('model_state_key_resolution.json', 'w') as f:
        json.dump(key_translation_dict, f, indent=2)


def update_model_state_keys(model_checkpoint_path, state_key_resolution_dict_path, key_resolution_dict=None):

    # Load checkpoint files for model for which we want to change the keys
    ckpt = torch.load(model_checkpoint_path, map_location='cpu')
    original_model_dict = ckpt['model_state_dict']
    optimizer_dict = ckpt['optimizer_state_dict']

    new_model_dict = OrderedDict()

    if key_resolution_dict is None:
        with open(state_key_resolution_dict_path, 'r') as f:
            key_resolution_dict = json.loads(f.read())

    assert len(original_model_dict) == len(key_resolution_dict), "Unequal number of keys"

    for k, v in original_model_dict.items():
        try:
            new_model_dict[key_resolution_dict[k]] = v
        except KeyError:
            print(f'[ WARN ] Key not present: {k}')

    torch.save(
        {
            'model_state_dict': new_model_dict,
            'optimizer_state_dict': optimizer_dict, # left unchanged
        },
        f'{model_checkpoint_path}.migrated_keys',
    )


def migrate_class_encoder_embedding_weights(checkpoint_path: str, overwrite: bool = False):
    """
    Migrate the embedding weights from old to new class encoder, which uses an `nn.Linear` layer for the class label
    embeddings instead of `nn.Embedding`. Since the weight shapes are reversed, this requires transposing the layer's
    weight matrix.

    Params:
    ---
    checkpoint_path: Path to the pickled model checkpoint on disk. Must be the `DiffWaveConditional` model with 
    `ClassEncoder` as conditioner.
    """

    # Read checkpoint
    checkpoint = torch.load(checkpoint_path,  map_location='cpu')
    
    # Transpose weights for embedding layer
    checkpoint['conditioner_state_dict']['embedding.weight'] = checkpoint['conditioner_state_dict']['embedding.weight'].T

    # Save checkpoint
    if overwrite:
        output_path = checkpoint_path
    else:
        output_path = checkpoint_path.replace('.pkl', '_updated.pkl')
    torch.save(checkpoint, output_path)

    return checkpoint


if __name__=='__main__':
    # update_model_state_keys(
    #     model_checkpoint_path=sys.argv[1],
    #     state_key_resolution_dict_path=sys.argv[2]
    # )
    migrate_class_encoder_embedding_weights(sys.argv[1])
