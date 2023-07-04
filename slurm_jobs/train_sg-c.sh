#!/bin/bash
#SBATCH --job-name="train-diffwave-class-conditional"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=63:00:00
#SBATCH --partition=gpu

echo
echo CLASS-CONDITIONAL PRE-TRAINING RUN
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
    train.name=VariaNTSWords-CC-v3 \
    experiment=pretraining_class_cond_variants \
    generate.conditional_signal=dag \
    generate.conditional_type=class \
    train.n_epochs=320 \
    train.epochs_per_ckpt=10 \
    train.iters_per_logging=50 \
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
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/brain2speech-diffusion/exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
