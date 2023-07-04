
![README Under Construction](https://img.shields.io/badge/%F0%9F%9A%A7-README%20Under%20Construction-white)

# 🧠🔊 Brain2Speech Diffusion: Speech Generation from Brain Activity using Diffusion Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/pascalschroeder/brain2speech-diffusion)


This repository proposes a diffusion-based generative model for the synthesis of natural speech from ECoG recordings of human brain activity. It supports pretraining of unconditional or class-conditional speech generators with consecutive fine-tuning on brain recordings, or fully end-to-end training on brain recordings. The diffusion model used is [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf), with different encoder models for the encoding of brain activity inputs or class labels. Originally, this repository started from [albertfgu's implementation of DiffWave](https://github.com/albertfgu/diffwave-sashimi).

<p align="center">
<img width="400" title="Speech Decoding From The Brain" alt="Model Architecture" src="images/Speech Decoding from Brain.png?raw=true">
</p>

## Samples

Speech samples generated for all models are provided [here](https://drive.google.com/drive/folders/1lk4U6UN6Ml8OI7Jb94BzS1VExi8DliMi).

## Table of Contents
* [Data](#data)
* [Usage](#usage)
  * [Training](#training)
  * [Generating](#generating)
  * [Pretrained Models](#pretrained-models)
* [Acknowledgements](#acknowledgements)
* [References](#references)


## Data

The [VariaNTS dataset](https://www.researchgate.net/publication/348012411_Development_and_structure_of_the_VariaNTS_corpus_A_spoken_Dutch_corpus_containing_talker_and_linguistic_variability) that was used as the speech dataset for this research can be downloaded [here](https://zenodo.org/record/3932039). 

You have to rename the incorrectly labelled 'föhn.wav' file for speaker p01 found in p01/p01_words/ to 'fohn.wav', as it is for the other speakers, to ensure that the data processing functions correctly.

## Usage

### Training

All training functionality is implemented in the `Learner` class (see `learner.py`). To train a model, call the `train.py` script, which loads the configurations and runs the `Learner`, and allows for single-GPU or distributed training.

#### Configurations

There are five important configuration paradigms: the model, the dataset, the diffusion process, and the training and generation controls.

The first three are collected in *experiments* for easy reproducibility. The respective configurations can be found in the `configs/experiment/` directory. Each experiment config needs to link to a model and dataset config (separately defined in the `configs/model/` and `configs/dataset/` directories, respectively) as well as define the parameters for the diffusion process. It is best to think of an experiment as a recipe, and the dataset, model, and diffusion parameters as the ingredients, i.e. ingredients are reused between experiments, but combined differently. If you want to define a new experiment, you can reuse ingredients or define new ones.

All of the default config values that will be loaded when calling the `train.py` script are defined in `configs/config.yaml`, with explanations about their effect. To overwrite values, simply pass them as command line arguments. For example, to change the experiment, pass `experiment=my_exp_config`, or to change the name of the training run, pass `train.name=Another-Run-v3`.

> **Note:** Hydra uses a hierarchical configuration system. This means that you need to prepend the commandline argument with the appropriate section name, e.g. `train.ckpt_epoch=300` or `generate.n_samples=6`. 
>
> **But**, since an 'experiment' is just a collection of configs, you must not do e.g. `experiment.diffusion.T=20`, but instead only `diffusion.T=20`. Same goes for model and dataset configs.

<!-- 
**Config options**
* `name`: The unique name of the experiment you're running, e.g. 'VariaNTS-Pretraining-v3'. This will also be used as the run name in W&B, should logging to W&B be enabled.
* `ckpt_epoch`: Which checkpoint to resume training from. If `ckpt_epoch=max`, will find the latest checkpoint, if `ckpt_iter=-1` will train from scratch. For any other positive integer, will try to load that exact checkpoint.
* `epochs_per_ckpt`: How many epoch to train between saving a model checkpoint to disk. This also controls the frequency of generating sample outputs.
* `iters_per_logging`: How many batches to feed in the train loop between logging of training metrics.
* `n_epochs`: The total number of epochs to train the model for
* `learning_rate`: The 
* `batch_size_per_gpu`: 8 -->

#### Logging

The repository implements logging via [Weights & Biases](https://wandb.ai/), as well as the storage of model artefacts and sample outputs. 

Before you start logging, you need to:
1. Setup a project on W&B,
2. Login to W&B in you terminal, and
2. Change the `project` and `entity` entries in the `wandb` section of the `configs/config.yaml` file accordingly. This can be done once in your local copy of the repository, such that you don't need to pass it every time when calling the train script.

There are additional configuration options to control the W&B logger:
* `wandb.mode`: `disabled` by default. Pass `wandb.mode=online` to activate logging to W&B. This will prompt you to login to W&B if you haven't done so.
* To continue logging to a previously tracked run, fetch the run's ID from your project page on W&B (it's in the URL, *not* the run's name) and pass `wandb.id=<run-id>` as well as `+wandb.resume=true` (note the leading '+' sign).

#### Debugging

If you want to quickly test code, you can run smaller versions of a dataset for debugging:
* Pass `dataset.segment_length=10` to cut all audio segments to 10 ms
* When using VariaNTS words as audio data, pass `dataset.splits_path=datasplits/VariaNTS/tiny_subset/` to only load few audio samples each epoch (assuming you have downloaded the provided datasplits, else you may create your own small subset)
* Pass `diffusion.T=5` to reduce the number of diffusion steps in generation

### Generating

There is a dedicated class `Sampler` in `sampler.py` that handles generation. It needs to be provided with a diffusion and dataset config and the appropriate generation parameters on initialization, and can then be used to run the full diffusion process to generate samples from noise (both conditional and unconditional).

In this repository, there are two places where generation takes place:
1. During training: The `Sampler` is initialized at the beginning of training and repeatedly runs on the updated model every `epochs_per_ckpt` epochs.
2. Using the `generate.py` script: When training is finished and a model is obtained, this script can be run individually to obtain more outputs from the model. 

Generation during training happens automatically (see `learner.py` for the implementation). Below is a description of how to run the `generate.py` script:

1. Navigate to the directory in which your `exp` output directory resides, as the script uses relative paths for model loading and output saving.
2. Call the script with the appropriate configuration options (see below). Like for training, config defaults are loaded from the `configs/config.yaml` file, and can be overwritten with command line arguments.

#### Config options

All configuration options can be found in the `configs/config.yaml` file's `generate` section, but the important ones (i.e. the ones changed most frequently) are listed below:

* `experiment`: Name of the experiment config that was used during the training run. This will load the required diffusion, dataset and model config
* `name`: Name of the training run that created the trained model that should be used
* `conditional_type`: Either `class` for class conditional sampling, or `brain` for brain conditional sampling. If the model is unconditional, can be null, or will otherwise be ignored.
* `conditional_signal`: The actual conditional input signal to use in case of conditional sampling. If the model is a class-conditional model, it suffices to simply state the word to be generated here (e.g. `dag`). If it is a brain-conditional model, this should be a full (absolute) file path to the ECoG recording file on disk (e.g. `/home/user/path/to/recording/dag1.npy`).

> **Note:** Since the `experiment` config is a top-level config, it suffices to append `experiment=...` as argument when calling the script. All other of the above mentioned options are options specifically for generation, so 'generate' has to be added to the argument description, e.g.: `generate.name=... generate.conditional_type=...` et cetera.

#### Example
```s
python brain2speech-diffusion/generate.py experiment=my-custom-experiment generate.name=Example-Model-v2 generate.conditional_type=class generate.conditional_signal=dag
```

### Pretrained Models

## Acknowledgements
This codebase was originally forked from the [DiffWave-SaShiMi repository by albertfgu](https://github.com/albertfgu/diffwave-sashimi), and some inspiration was taken from [LMNT's implementation of DiffWave](https://github.com/lmnt-com/diffwave), specifically how the model code was structured.

## References
* [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)


## Addendum

### Experiment collection

Examples of how experiments can be run. Given parameters need to be changed.

In case only a subset of all available GPUs should be used, add `CUDA_VISIBLE_DEVICES=<id>,<id>` before calling the script.

If debugging, add `diffusion.T=5` as flag, which will reduce the number of diffusion steps during generation.

#### Unconditional Pretraining

> Note: You also have to specify `generate.conditional_signal=null` for this experiment such that the script does not load conditional input, as the default is set for brain input.

```c
python brain2speech-diffusion/train.py \
    train.name=delete-me \
    experiment=SG-U \
    train.n_epochs=2 \
    train.epochs_per_ckpt=1 \
    train.iters_per_logging=1 \
    generate.conditional_signal=null
```

#### Class-Conditional Pretraining

> Note: You also have to specify `generate.conditional_type=class` and `generate.conditional_signal=<word>` for this experiment to determine which word to load for intermediate generations, as the default is set for brain input.

```c
python brain2speech-diffusion/train.py \
    train.name=delete-me \
    experiment=SG-C \
    train.n_epochs=2 \
    train.epochs_per_ckpt=1 \
    train.iters_per_logging=1 \
    generate.conditional_type=class \
    generate.conditional_signal=dag
```

#### Brain-Conditional Fine-Tuning

Harry Potter speech data:

```c
python brain2speech-diffusion/train.py \
    train.name=delete-me \
    experiment=B2S-Ur \
    model.freeze_generator=false \
    train.n_epochs=2 \
    train.epochs_per_ckpt=1 \
    train.iters_per_logging=1
```

VariaNTS speech data:

```c
python brain2speech-diffusion/train.py \
    train.name=delete-me \
    experiment=B2S-Uv \
    model.freeze_generator=true \
    train.n_epochs=2 \
    train.epochs_per_ckpt=1 \
    train.iters_per_logging=1
```


#### Brain + Class Conditional Fine-Tuning

```c
python brain2speech-diffusion/train.py \
    train.name=delete-me \
    experiment=B2S-Cv \
    train.n_epochs=2 \
    train.epochs_per_ckpt=1 \
    train.iters_per_logging=1
```

### Experiment Overview Table

Experiment                   | Model                                        | Conditional Input | Speech Data | Splits
-----------------------------|----------------------------------------------|-------------------|-------------|-----------------
Uncond. Pretraining          | DiffWave                                     | -                 | VariaNTS    | full VariaNTS
Classcond. Pretraining       | DiffWave + Class Encoder                     | Class vector      | VariaNTS    | full VariaNTS
Brainclasscond. Finetuning   | DiffWave + Class Encoder + Brain Classifier  | HP ECoG Data      | VariaNTS    | reduced VariaNTS*
Braincond. Finetuning (VNTS) | DiffWave + Brain Encoder                     | HP ECoG Data      | VariaNTS    | reduced VariaNTS*
Braincond. Finetuning (HP)   | DiffWave + Brain Encoder                     | HP ECoG Data      | HP Speech   | HP splits

\* reduced VariaNTS means that words which are not present in the Harry Potter ECoG data were removed, in order to correctly map ECoG to speech data.

---
[🔝 Back to Top](#-brain2speech-diffusion-speech-generation-from-brain-activity-using-diffusion-models)

<!-- # DiffWave Unconditional Dutch

> :warning: **Disclaimer:**
>
> This is a fork of the [DiffWave-SaShiMi repository by albertfgu](https://github.com/albertfgu/diffwave-sashimi) to train the model on Dutch spoken data in the unconditional setting.
>
> It's in development and changes are not well documented. I do not recommend working with this repository in its current state, but if you are interested in doing so anyways, please reach out to me before.
>
> The rest of this Readme is the original one from albertfgu's repository.

## :point_down: Original ReadMe

This repository is an implementation of the waveform synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).
It also has code to reproduce the SaShiMi+DiffWave experiments from [It’s Raw! Audio Generation with State-Space Models](https://arxiv.org/abs/2202.09729) (Goel et al. 2022).

This is a fork/combination of the implementations [philsyn/DiffWave-unconditional](https://github.com/philsyn/DiffWave-unconditional) and [philsyn/DiffWave-Vocoder](https://github.com/philsyn/DiffWave-Vocoder).
This repo uses Git LFS to store model checkpoints, which unfortunately [does not work with public forks](https://github.com/git-lfs/git-lfs/issues/1939).
For this reason it is not an official GitHub fork.

## Overview

This repository aims to provide a clean implementation of the DiffWave audio diffusion model.
The `checkpoints` branch of this repository has the original code used for reproducing experiments from the SaShiMi paper ([instructions](#pretrained-models)).
The `master` branch of this repository has the latest versions of the S4/SaShiMi model and can be used to train new models from scratch.


Compared to the parent fork for DiffWave, this repository has:
- Both unconditional (SC09) and vocoding (LJSpeech) waveform synthesis. It's also designed in a way to be easy to add new datasets
- Significantly improved infrastructure and documentation
- Configuration system with Hydra for modular configs and flexible command-line API
- Logging with WandB instead of Tensorboard, including automatically generating and uploading samples during training
- Vocoding does not require a separate pre-processing step to generate spectrograms, making it easier and less error-prone to use
- Option to replace WaveNet with the SaShiMi backbone (based on the [S4 layer](https://github.com/HazyResearch/state-spaces))
- Pretrained checkpoints and samples for both DiffWave (+Wavenet) and DiffWave+SaShiMi

These are some features that would be nice to add.
PRs are very welcome!
- Use the pip S4 package once it's released, instead of manually updating the standalone files
- Mixed-precision training
- Fast inference procedure from later versions of the DiffWave paper
- Can add an option to allow original Tensorboard logging instead of WandB (code is still there, just commented out)
- The different backbones (WaveNet and SaShiMi) can be consolidated more cleanly with the diffusion logic factored out

## ToC

- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Vocoding](#vocoding)
- [Pretrained Models](#pretrained-models)
  - [DiffWave+SaShiMi](#sashimi-1)
  - [DiffWave](#wavenet-1)

## Usage

A basic experiment can be run with `python train.py`.
This default config is for SC09 unconditional generation with the SaShiMi backbone.

### Hydra

Configuration is managed by [Hydra](https://hydra.cc).
Config files are under `configs/`.
Examples of different configs and running experiments via command line are provided throughout this README.
Hydra has a steeper learning curve than standard `argparse`-based workflows, but offers much more flexibility and better experiment management. Feel free to file issues for help with configs.

### Multi-GPU training
By default, all available GPUs are used (according to [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count)).
You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module, or e.g. `CUDA_VISIBLE_DEVICES=0,1 python train.py`.


## Data

Unconditional generation uses the SC09 dataset by default, while vocoding uses [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).
The entry `dataset_config.data_path` of the config should point to the desired folder, e.g. `data/sc09` or `data/LJSpeech-1.1/wavs`

### SC09
For SC09, extract the Speech Commands dataset and move the digits subclasses into a separate folder, e.g. `./data/sc09/{zero,one,two,three,four,five,six,seven,eight,nine}`

### LJSpeech

Download the LJSpeech dataset into a folder. For example (as of 2022/06/28):
```
mkdir data && cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar.bz2
```
The waveforms should be organized in the folder `./data/LJSpeech-1.1/wavs`


## Models

All models use the DiffWave diffusion model with either a WaveNet or SaShiMi backbone.
The model configuration is controlled by the dictionary `model` inside the config file.

### Diffusion

The flags `in_channels` and `out_channels` control the data dimension.
The three flags `diffusion_step_embed_dim_{in,mid,out}` control DiffWave parameters for the diffusion step embedding.
The flag `unconditional` determines whether to do unconditional or conditional generation.
These flags are common to both backbones.

### WaveNet

The WaveNet backbone is used by setting `model._name_=wavenet`.
The parameters `res_channels`, `skip_channels`, `num_res_layers`, and `dilation_cycle` control the WaveNet backbone.

### SaShiMi

The SaShiMi backbone is used by setting `model._name_=sashimi`.
The parameters are:
```yaml
unet:     # If true, use S4 layers in both the downsample and upsample parts of the backbone. If false, use S4 layers in only the upsample part.
d_model:  # Starting model dimension of outermost stage
n_layers: # Number of layers per stage
pool:     # List of pooling factors per stage (e.g. [4, 4] means three total stages, pooling by a factor of 4 in between each)
expand:   # Multiplicative factor to increase model dimension between stages
ff:       # Feedforward expansion factor: MLP block has dimensions d_model -> d_model*ff -> d_model)
```

## Training

A basic experiment can be run with `python train.py`, which defaults to `python train.py experiment=sc09` (SC09 unconditional waveform synthesis).

Experiments are saved under `exp/<run>` with an automatically generated run name identifying the experiment (model and setting).
Checkpoints are saved to `exp/<run>/checkpoint/` and generated audio samples to `exp/<run>/waveforms/`.

### Logging
Set `wandb.mode=online` to turn on WandB logging, or `wandb.mode=disabled` to turn it off.
Standard wandb arguments such as entity and project are configurable.

### Resuming

To resume from the checkpoint `exp/<run>/checkpoint/1000.pkl`, simply re-run the same training command with the additional flag `train.ckpt_iter=1000`.
Passing in `train_config.ckpt_iter=max` resumes from the last checkpoint, and `train_config.ckpt_iter=-1` trains from scratch.

Use `wandb.id=<id>` to resume logging to a previous run.

### Generating

After training with `python train.py <flags>`, `python generate.py <flags>` generates samples according to the `generate` dictionary of the config.

For example,
```
python generate.py <flags> generate.ckpt_iter=500000 generate.n_samples=256 generate.batch_size=128
```
generates 256 samples per GPU, at a batch size of 128 per GPU, from the model specified in the config at checkpoint iteration 500000.

Generated samples will be stored in `exp/<run>/waveforms/<generate.ckpt_iter>/`

## Vocoding

- After downloading the data, make the config's `dataset.data_path` point to the `.wav` files
- Toggle `model.unconditional=false`
- Pass in the name of a `.wav` file for generation, e.g. `generate.mel_name=LJ001-0001`. Every checkpoint, vocoded samples for this audio file will be logged to wandb

Currently, vocoding is only set up for the LJSpeech dataset. See the config `configs/experiment/ljspeech.yaml` for details.
The following is an example command for LJSpeech vocoding with a smaller SaShiMi model. A checkpoint for this model at 200k steps is also provided.
```
python train.py experiment=ljspeech model=sashimi model.d_model=32 wandb.mode=online
```

Another example with a smaller WaveNet backbone, similar to the results from the DiffWave paper:
```
python train.py experiment=ljspeech model=wavenet model.res_channels=64 model.skip_channels=64 wandb.mode=online
```

Generation can be done in the usual way, conditioning on any spectrogram, e.g.
```
python generate.py experiment=ljspeech model=sashimi model.d_model=32 generate.mel_name=LJ001-0002
```

### Pre-processed Spectrograms

Other DiffWave vocoder implementations such as https://github.com/philsyn/DiffWave-Vocoder and https://github.com/lmnt-com/diffwave require first generating spectrograms in a separate pre-processing step.
This implementation does not require this step, which we find more convenient.
However, pre-processing and saving the spectrograms is still possible.

To generate a folder of spectrograms according to the `dataset` config, run the `mel2samp` script and specify an output directory (e.g. here 256 refers to the hop size):
```
python -m dataloaders.mel2samp experiment=ljspeech +output_dir=mel256
```

Then during training or generation, add in the additional flag `generate.mel_path=mel256` to use the pre-processed spectrograms, e.g.
```
python generate.py experiment=ljspeech model=sashimi model.d_model=32 generate.mel_name=LJ001-0002 generate.mel_path=mel256
```


# Pretrained Models

The remainder of this README pertains only to pre-trained models from the SaShiMi paper.

The branch `git checkout checkpoints` provides checkpoints for these models.

**This branch is meant only for reproducing generated samples from the SaShiMi paper from ICML 2022 - please do not attempt train-from-scratch results from this code.**
The older models in this branch have issues that are explained below.

Training from scratch is covered in the previous part of this README and should be done from the `master` branch.

### Checkpoints

Install [Git LFS](https://git-lfs.github.com/) and `git lfs pull` to download the checkpoints.

### Samples
For each of the provided checkpoints, 16 audio samples are provided.

More pre-generated samples for all models from the SaShiMi paper can be downloaded from: https://huggingface.co/krandiash/sashimi-release/tree/main/samples/sc09

The below four models correspond to "sashimi-diffwave", "sashimi-diffwave-small", "diffwave", and "diffwave-small" respectively.
Command lines are also provided to reproduce these samples (up to random seed).


## SaShiMi

The version of S4 used in the experiments in the SaShiMi paper is an outdated version of S4 from January 2022 that predates V2 (February 2022) of the [S4 repository](https://github.com/HazyResearch/state-spaces). S4 is currently on V3 as of July 2022.

### DiffWave+SaShiMi

- Experiment folder: `exp/unet_d128_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`
- Train from scratch: `python train.py model=sashimi train.ckpt_iter=-1`
- Resume training: `python train.py model=sashimi train.ckpt_iter=max train.learning_rate=1e-4`
(as described in the paper, this model used a manual learning rate decay after 500k steps)

Generation examples:
- `python generate.py experiment=sc09 model=sashimi` (Latest model at 800k steps)
- `python generate.py experiment=sc09 model=sashimi generate.ckpt_iter=500000` (Earlier model at 500k steps)
- `python generate.py generate.n_samples=256 generate.batch_size=128` (Generate 256 samples per GPU with the largest batch that fits on an A100. The paper uses this command to generate 2048 samples on an 8xA100 machine for evaluation metrics.)

### DiffWave+SaShiMi small

Experiment folder: `exp/unet_d64_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`

Train (since the model is smaller, you can increase the batch size and logged samples per checkpoint):
```
python train.py experiment=sc09 model=sashimi model.d_model=64 train.batch_size_per_gpu=4 generate.n_samples=32
```

Generate:
```
python generate.py experiment=sc09 model=sashimi model.d_model=64 generate.n_samples=256 generate.batch_size=256
```

## WaveNet

The WaveNet backbone provided in the parent fork had a [small](https://github.com/albertfgu/diffwave-sashimi/blob/checkpoints/models/wavenet.py#L92) [bug](https://github.com/albertfgu/diffwave-sashimi/blob/checkpoints/models/wavenet.py#L163) where it used `x += y` instead of `x = x + y`.
This can cause a difficult-to-trace error in some PyTorch + environment combinations (but sometimes it works; I never figured out when it's ok).
These two lines are fixed in the master branch of this repo.

However, for some reason when models are *trained using the wrong code* and *loaded using the correct code*,
the model runs fine but produces inconsistent outputs, even in inference mode (i.e. generation produces static noise).
So this branch for reproducing the checkpoints uses the incorrect version of these two lines.
This allows generating from the pre-trained models, but may not train in some environments.
**If anyone knows why this happens, I would love to know! Shoot me an email or file an issue!**


### DiffWave(+WaveNet)
- Experiment folder: `exp/wnet_h256_d36_T200_betaT0.02_uncond/`
- Usage: `python <train|generate>.py model=wavenet`

More notes:
- The fully trained model (1000000 steps) is the original checkpoint from the original repo philsyn/DiffWave-unconditional
- The checkpoint at 500000 steps is our version trained from scratch
- These should both be compatible with this codebase (e.g. generation works with both), but for some reason the original `checkpoint/1000000.pkl` file is much smaller than our `checkpoint/500000.pkl`
- I don't remember if I changed anything in the code to cause this; perhaps it could also be differences in PyTorch or versions or environments?

### DiffWave(+WaveNet) small
Experiment folder: `exp/wnet_h128_d30_T200_betaT0.02_uncond/`

Train:
```
python train.py model=wavenet model.res_channels=128 model.num_res_layers=30 model.dilation_cycle=10 train.batch_size_per_gpu=4 generate.n_samples=32
```
A shorthand model config is also defined
```
python train.py model=wavenet_small train.batch_size_per_gpu=4 generate.n_samples=32
```


## Vocoders
The parent fork has a few pretrained LJSpeech vocoder models. Because of the WaveNet bug, we recommend not using these and simply training from scratch from the `master` branch; these vocoder models are small and faster to train than the unconditional SC09 models.
Feel free to file an issue for help with configs.
 -->
