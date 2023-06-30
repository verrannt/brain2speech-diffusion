#!/bin/bash
#SBATCH --job-name="testpython"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:00:30
#SBATCH --partition=staging


# Load modules from registry
module purge
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Activate python virtual environment
source venvs/test/bin/activate

# Copy data
cp -r $HOME/data $TMPDIR/data

# Copy scripts
cp -r $HOME/scripts $TMPDIR/scripts

cd $TMPDIR

mkdir results

echo "Whats in here"
ls -R
echo
echo "Location:"
pwd

# Run computation
python scripts/test.py

# Retrieve outputs
cp -r $TMPDIR/results $HOME/results

# Deactivate virtual environment
deactivate

echo "DONE"
