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

# Copy codebase
echo [$(date +"%T")] Copying codebase
cp -r $HOME/brain2speech-diffusion $TMPDIR/brain2speech-diffusion

# Change directory
echo [$(date +"%T")] Navigating to $TMPDIR/brain2speech-diffusion
cd $TMPDIR/brain2speech-diffusion

# Run computation
echo [$(date +"%T")] Executing train script
python src/train.py \
    train.name=Uncond-PT-v9-noaug \
    experiment=pretraining_uncond_variants \
    generate.conditional_signal=null \
    train.n_epochs=250 \
    train.epochs_per_ckpt=10 \
    train.iters_per_logging=50 \
    train.batch_size_per_gpu=12 \
    generate.n_samples=12 \
    wandb.mode=online \
    # wandb.id=<id> \
    # +wandb.resume=true

# For testing purposes, the 'tiny' version of the data can be used:
# python src/train.py \
#     dataset.file_base_path="$TMPDIR"/data/VariaNTS_words_22kHz/ \
#     dataset.data_path="$TMPDIR"/data/datasplits/VariaNTS_Words/tiny_subset0.015-50-25-25 \
#     experiment=variants_tiny \
#     train.name=testing \
#     wandb.mode=online

# Retrieve outputs
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/brain2speech-diffusion/exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
