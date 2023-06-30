#!/bin/bash
#SBATCH --job-name="diffwave-brain-finetuning-vnts"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=26:00:00
#SBATCH --partition=gpu

echo
echo FINETUNE UNCONDITIONAL DIFFWAVE MODEL ON HARRY POTTER ECOG AND VARIANTS AUDIO DATA
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
    train.name=BrainCond-FT-VariaNTS-v3 \
    experiment=finetuning_brain_cond_variants \
    dataset.data_base_dir="$TMPDIR"/data/ \
    generate.conditional_signal="$TMPDIR"/data/HP1_ECoG_conditional/sub-002/dag2.npy \
    generate.conditional_type=brain \
    model.freeze_generator=true \
    model.pretrained_generator="$TMPDIR"/exp/Uncond-PT-v9/checkpoint/230.pkl \
    train.n_epochs=170 \
    train.epochs_per_ckpt=10 \
    train.iters_per_logging=100 \
    train.batch_size_per_gpu=12 \
    generate.n_samples=12 \
    wandb.mode=online \
    # wandb.id=<id> \
    # +wandb.resume=true \

# For testing purposes, the 'tiny' version of the data can be used:
# python brain2speech-diffusion/train.py \
#     dataset.data_path="$TMPDIR"/data/hp1_eeg_conditional/ \
#     generate.conditional_file="$TMPDIR"/data/hp1_eeg_conditional/train/dag7.npy \
#     experiment=brain_conditional_tiny \
#     train.name=testing_brain_cond \
#     +model.pretrained_generator="$TMPDIR"/exp/VariaNTSWords-v6_wnet_h256_d36_T200_betaT0.02_uncond/checkpoint/550.pkl \
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
