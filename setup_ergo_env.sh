#!/bin/bash

# Create a new conda environment with Python 3.9
conda create -n torch_env_run_1 python=3.9 -y

# Activate the environment
source activate torch_env_run_1

# Upgrade and force reinstall numpy
pip install --upgrade --force-reinstall numpy==1.24.3

# Install PyTorch and associated libraries with CUDA toolkit 11.3
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -y

# Install additional libraries
pip install tensorboardX torchtext==0.11.0 scikit-learn

echo "Environment setup complete. Activate it with 'conda activate torch_env_run_1'."
