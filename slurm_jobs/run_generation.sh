#!/bin/bash
#SBATCH --job-name="generate-several"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=00:40:00
#SBATCH --partition=gpu

echo
echo GENERATE ALL OUTPUTS FOR MODEL
echo 
echo $(date +"%D %T")
echo

# Load modules from registry
echo [$(date +"%T")] Loading modules
module purge
module load 2021
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Python/3.9.5-GCCcore-10.3.0

echo [$(date +"%T")] Activating virtual environment
source venvs/diffwave/bin/activate

echo [$(date +"%T")] Copying data
cp -r $HOME/data $TMPDIR/data

echo [$(date +"%T")] Copying DiffWave codebase
cp -r $HOME/brain2speech-diffusion $TMPDIR/brain2speech-diffusion

echo [$(date +"%T")] Copying DiffWave checkpoints
cp -r $HOME/exp $TMPDIR/exp

echo [$(date +"%T")] Navigating to $TMPDIR
cd $TMPDIR

# Run computation
# It's important to run this from the base dir (i.e. $TMPDIR), such that the
# `exp` output directory is in the base dir, too, and can be appropriately copied
echo [$(date +"%T")] Executing generate script
# Note: for the unconditional model, the standard generate.py script is used, 
# as no separation by word classes is required, and parallelization can
# therefore be used easily.
python brain2speech-diffusion/generate.py \
    experiment=pretraining_uncond_variants \
    dataset.data_base_dir="$TMPDIR"/data/ \
    generate.name=Uncond-PT-v9-noaug \
    generate.conditional_type=null \
    generate.conditional_signal=null \
    generate.ckpt_epoch=1000 \
    generate.n_samples=220 \
    generate.batch_size=20
# python brain2speech-diffusion/generate_several.py \
#     experiment=pretraining_class_cond_variants \
#     dataset.data_base_dir="$TMPDIR"/data/ \
#     generate.name=VariaNTSWords-CC-v3 \
#     generate.conditional_type=class \
#     generate.ckpt_epoch=180 \
#     +use_val=False
# python brain2speech-diffusion/generate_several.py \
#     experiment=finetuning_brain_cond_hp \
#     dataset.data_base_dir="$TMPDIR"/data/ \
#     generate.name=BrainCond-FT-HP-v5 \
#     generate.conditional_type=brain \
#     generate.ckpt_epoch=70 \
#     +use_val=False
# python brain2speech-diffusion/generate_several.py \
#     experiment=finetuning_brain_cond_variants \
#     dataset.data_base_dir="$TMPDIR"/data/ \
#     generate.name=BrainCond-FT-VariaNTS-v3 \
#     generate.conditional_type=brain \
#     generate.ckpt_epoch=??? \
#     +use_val=False
# python brain2speech-diffusion/generate_several.py \
#     experiment=finetuning_brain_class_cond_variants \
#     dataset.data_base_dir="$TMPDIR"/data/ \
#     generate.name=BrainClassCond-FT-VariaNTS-v9 \
#     generate.conditional_type=brain \
#     generate.ckpt_epoch=800 \
#     +use_val=False

# Retrieve outputs
echo [$(date +"%T")] Retrieving outputs
cp -r $TMPDIR/exp/* $HOME/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
