# ----------------------------------------------------------------------------
# TEMPLATE for creating a job script for slurm.
# 
# This is used by all training scripts in this directory. They only differ
# in the call they make. To create your own, copy this template and change 
# lines 44-47. Required parameters slurm can be adjusted below.
# ----------------------------------------------------------------------------

#!/bin/bash
#SBATCH --job-name="template"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=65:00:00
#SBATCH --partition=gpu

echo
echo TEMPLATE TRAINING RUN
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
# ----------------------------------------------------------------------------
# Add your call to the training script here
# $ python src/train.py args...
# ----------------------------------------------------------------------------

# Retrieve outputs
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/brain2speech-diffusion/exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
