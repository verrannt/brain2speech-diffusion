#!/bin/bash
#SBATCH --job-name="diffwave-brain-class-finetuning"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=03:30:00
#SBATCH --partition=gpu

echo
echo FINETUNE CLASS CONDITIONAL DIFFWAVE MODEL ON VARIANTS + BRAIN DATA
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
    train.name=BrainClassConditional-v6 \
    experiment=finetuning_brain_class_cond_variants \
    generate.conditional_signal=data/HP1_ECoG_conditional/sub-002/dag2.npy \
    generate.conditional_type=brain \
    model.freeze_generator=true \
    model.pretrained_generator=exp/VariaNTSWords-CC-v3/checkpoint/180.pkl \
    train.n_epochs=8 \
    train.epochs_per_ckpt=1 \
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
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/brain2speech-diffusion/exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
