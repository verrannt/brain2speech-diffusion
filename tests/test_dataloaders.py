import sys
sys.path.append('.')

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader



def test_brain_conditional():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/brain_conditional.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=False)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, conditional_input, mask = data
            print("Audio:", audio.shape)
            print("EEG:  ", conditional_input.shape)
            if mask is not None:
                print("Mask: ", mask.shape)


def test_variants_brain():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/variants_brain.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=False)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, conditional_input, mask = data
            print("Audio:", audio.shape)
            print("EEG:  ", conditional_input.shape)
            if mask is not None:
                print("Mask: ", mask.shape)


def test_variants():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/variants.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=True)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, _, mask = data
            print("Audio:", audio.shape)
            if mask is not None:
                print("Mask: ", mask.shape)



if __name__=='__main__':
    test_variants_brain()