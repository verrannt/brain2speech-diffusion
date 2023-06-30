#!/bin/bash
#SBATCH --job-name="train-diffwave-unconditional"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=65:00:00
#SBATCH --partition=gpu

echo
echo UNCONDITIONAL PRE-TRAINING RUN
echo 
echo $(date +"%D %T")
echo

# Load modules from registry
echo [$(date +"%T")] Loading modules
module purge
module load 2021
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Python/3.9.5-GCCcore-10.3.0

# Activate python virtual environment
echo [$(date +"%T")] Activating virtual environment
source venvs/diffwave/bin/activate

# Copy data
echo [$(date +"%T")] Copying data
cp -r $HOME/data $TMPDIR/data

# Copy codebase
echo [$(date +"%T")] Copying DiffWave codebase
cp -r $HOME/brain2speech-diffusion $TMPDIR/brain2speech-diffusion

# Copy checkpoints
echo [$(date +"%T")] Copying DiffWave checkpoints
cp -r $HOME/exp $TMPDIR/exp

echo [$(date +"%T")] Navigating to $TMPDIR
cd $TMPDIR

# Run computation
# It's important to run this from the base dir (i.e. $TMPDIR), such that the
# `exp` output directory is in the base dir, too, and can be appropriately copied
echo [$(date +"%T")] Executing train script
python brain2speech-diffusion/train.py \
    train.name=Uncond-PT-v9-noaug \
    experiment=pretraining_uncond_variants \
    dataset.data_base_dir="$TMPDIR"/data/ \
    dataset.splits_path=datasplits/VariaNTS/HP_90-10/ \
    generate.conditional_signal=null \
    train.n_epochs=6600 \
    train.epochs_per_ckpt=100 \
    train.iters_per_logging=50 \
    train.batch_size_per_gpu=12 \
    generate.n_samples=12 \
    wandb.mode=online \
    wandb.id=<id> \
    +wandb.resume=true

# For testing purposes, the 'tiny' version of the data can be used:
# python brain2speech-diffusion/train.py \
#     dataset.file_base_path="$TMPDIR"/data/VariaNTS_words_22kHz/ \
#     dataset.data_path="$TMPDIR"/data/datasplits/VariaNTS_Words/tiny_subset0.015-50-25-25 \
#     experiment=variants_tiny \
#     train.name=testing \
#     wandb.mode=online

# Retrieve outputs
# The train.py script will create a directory named after the model run in the
# `exp` folder, and since an `exp` folder already exists in HOME, we do not
# want to copy the whole folder, but just its contents, using the wildcard *
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/exp/* $HOME/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
