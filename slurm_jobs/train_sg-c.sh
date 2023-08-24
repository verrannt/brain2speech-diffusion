#!/bin/bash
#SBATCH --job-name="train-sg-c"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=10:00:00
#SBATCH --partition=gpu

echo
echo TRAIN SG-C
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
    train.name=SG-C_v4 \
    experiment=SG-C \
    generate.conditional_signal=dag \
    generate.conditional_type=class \
    train.n_epochs=250 \
    train.epochs_per_ckpt=10 \
    train.iters_per_logging=50 \
    train.batch_size_per_gpu=12 \
    generate.n_samples=12 \
    wandb.mode=online \
    # wandb.id=<id> \
    # +wandb.resume=true \

# NOTE If you want to train the model on a different dataset, you might have
# to configure the following options:
# - dataset.splits_path
# - dataset.targets
# - model.encoder_config.n_classes

# Retrieve outputs
echo [$(date +"%T")] Retrieving outputs
cp -r exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
