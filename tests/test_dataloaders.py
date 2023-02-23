import sys
sys.path.append('.')

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader, ClassConditionalLoader
import utils


def test_brain_conditional():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/brain_conditional.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    dataset_cfg = utils.prepend_data_base_dir(dataset_cfg)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=False)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, conditional_input, mask = data
            print("Audio:", audio.shape)
            print("ECoG:  ", conditional_input.shape)
            if mask is not None:
                print("Mask: ", mask.shape)


def test_variants_brain():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/variants_brain.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    dataset_cfg = utils.prepend_data_base_dir(dataset_cfg)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=False)
    # print(valloader.dataset.conditional_loader.files['val'])
    # print(valloader.dataset.conditional_loader.files['train'])
        
    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            if i % 1000 == 0: print(i)
            # audio, _, conditional_input, mask = data
            # print("Audio:", audio.shape)
            # print("ECoG:  ", conditional_input.shape)
            # if mask is not None:
            #     print("Mask: ", mask.shape)


def test_variants_unconditional():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/variants.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    dataset_cfg = utils.prepend_data_base_dir(dataset_cfg)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=True)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, _, mask = data
            print("Audio:", audio.shape)
            if mask is not None:
                print("Mask: ", mask.shape)


def test_variants_class_conditional():
    dataset_config_path = '/home/passch/diffwave/configs/dataset/variants.yaml'
    dataset_cfg = OmegaConf.load(dataset_config_path)
    dataset_cfg = utils.prepend_data_base_dir(dataset_cfg)
    trainloader, valloader, _ = dataloader(dataset_cfg, batch_size=8, num_gpus=1, unconditional=False)

    for loader in [trainloader, valloader]:
        for i, data in enumerate(loader):
            print(i)
            audio, _, conditional_input, mask = data
            print("Audio:", audio.shape)
            print("Class:", conditional_input.shape)
            if mask is not None:
                print("Mask: ", mask.shape)


def test_class_conditional_loader():
    conditional_loader = ClassConditionalLoader(words_file='/home/passch/data/HP_VariaNTS_intersection.txt')

    import torch
    import time
    start = time.time()
    out1 = torch.stack([conditional_loader('path/to/good/things/bed_timestretch1.wav') for _ in range(12)], dim=0)
    end = time.time()
    print(out1.shape)
    print(end-start)
    
    start = time.time()
    out2 = conditional_loader.batch_call(['path/to/good/things/bed_timestretch1.wav']*12)
    end = time.time()
    print(out2.shape)
    print(end-start)

    # print(conditional_loader('path/to/good/things/bed_timestretch1.wav'))
    # print(conditional_loader('bed'))
    # print(conditional_loader('boel'))
    # print(conditional_loader('brief'))
    # print(conditional_loader('half'))
    # print(conditional_loader('meer'))
    # print(conditional_loader('zet'))
    # print(conditional_loader('zin'))
    # print(conditional_loader('zoon'))


if __name__=='__main__':
    test_class_conditional_loader()