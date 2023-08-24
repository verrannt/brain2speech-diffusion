#!/bin/bash
#SBATCH --job-name="generate-several"
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus=4
#SBATCH --time=02:00:00
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
echo [$(date +"%T")] Executing generate script
# Note: for the unconditional model, the standard generate.py script is used, 
# as no separation by word classes is required, and parallelization can
# therefore be used easily.
# python src/generate.py \
#     experiment=SG-U \
#     generate.name=SG-U_v10 \
#     generate.conditional_type=null \
#     generate.conditional_signal=null \
#     generate.ckpt_epoch=500 \
#     generate.n_samples=220 \
#     generate.batch_size=20
# python src/generate_several.py \
#     experiment=SG-C \
#     generate.name=SG-C_v4 \
#     generate.conditional_type=class \
#     generate.ckpt_epoch=490 \
#     dataset.targets=[dag,heel,kan,keer,man,wel] \
#     model.encoder_config.n_classes=6 \
#     +use_val=False
# python src/generate_several.py \
#     experiment=B2S-Ur \
#     generate.name=B2S-Ur_v6 \
#     generate.conditional_type=brain \
#     generate.ckpt_epoch=45 \
#     model.encoder_config.c_in=123 \
#     model.encoder_config.c_mid=128 \
#     model.encoder_config.c_out=128 \
#     +model.encoder_config.kernel_size=32 \
#     +model.encoder_config.stride=12 \
#     +use_val=False
python src/generate_several.py \
    experiment=B2S-Uv \
    generate.name=B2S-Uv_v4 \
    generate.conditional_type=brain \
    generate.ckpt_epoch=490 \
    model.encoder_config.c_in=123 \
    model.encoder_config.c_mid=128 \
    model.encoder_config.c_out=128 \
    +model.encoder_config.kernel_size=32 \
    +model.encoder_config.stride=12 \
    +use_val=False
# python src/generate_several.py \
#     experiment=B2S-Cv \
#     generate.name=BrainClassCond-FT-VariaNTS-v9 \
#     generate.conditional_type=brain \
#     generate.ckpt_epoch=800 \
#     +use_val=False

# Retrieve outputs
echo [$(date +"%T")] Retrieving outputs
cp -r exp/* $HOME/brain2speech-diffusion/exp

# Deactivate virtual environment
echo [$(date +"%T")] Deactivating virtual environment
deactivate

echo [$(date +"%T")] Done
