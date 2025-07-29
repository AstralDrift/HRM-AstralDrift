#!/bin/bash

# HRM Training Environment Setup Script for RTX 4090
# Optimized for Pop!_OS with CUDA 12.6, Python 3.10, and FlashAttention 2

set -e  # Exit on any error

echo "🚀 Starting HRM Training Environment Setup..."
echo "================================================"

# Step 1: Update system and install system dependencies
echo "📦 Step 1: Updating system and installing dependencies..."
sudo apt update -y
sudo apt install -y build-essential git g++ libomp-dev wget curl python3.10 python3.10-venv python3.10-dev
echo "✅ System dependencies installed"

# Step 2: Check/install CUDA 12.6
echo "🔧 Step 2: Checking CUDA installation..."
if ! command -v nvcc &> /dev/null || ! nvcc --version | grep "release 12.6" &> /dev/null; then
    echo "❌ CUDA 12.6 not found, installing..."
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
    wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run "$CUDA_URL"
    sudo sh cuda_installer.run --silent --toolkit --override
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    rm cuda_installer.run
else
    echo "✅ CUDA 12.6 is already installed"
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Step 3: Create Python 3.10 virtual environment
echo "🐍 Step 3: Setting up Python 3.10 virtual environment..."
if [ ! -d "venv_py310" ]; then
    python3.10 -m venv venv_py310
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv_py310/bin/activate
echo "✅ Virtual environment activated"

# Step 4: Install PyTorch with CUDA 12.6 and dependencies
echo "⚡ Step 4: Installing PyTorch and dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Install additional build dependencies
pip install packaging ninja setuptools-scm

# Install project requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "⚠️  requirements.txt not found, installing core dependencies manually..."
    pip install adam-atan2 einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub
fi

# Step 5: Install FlashAttention 2 for Ampere (RTX 4090)
echo "⚡ Step 5: Installing FlashAttention 2..."
pip install flash-attn --no-build-isolation
echo "✅ FlashAttention 2 installed"

# Step 6: Set environment variables
echo "🔧 Step 6: Setting environment variables..."
export OMP_NUM_THREADS=8
echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
echo "✅ Environment variables configured"

# Step 7: Verify setup
echo "🔍 Step 7: Verifying setup..."
echo "NVIDIA GPU Info:"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader,nounits

echo "CUDA Version:"
nvcc --version | grep "release"

echo "PyTorch CUDA availability:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'OMP_NUM_THREADS: {torch.get_num_threads()}')
"

echo "FlashAttention test:"
python -c "
try:
    import flash_attn
    print('✅ FlashAttention imported successfully')
    print(f'FlashAttention version: {flash_attn.__version__}')
except ImportError as e:
    print(f'❌ FlashAttention import failed: {e}')
"

# Step 8: W&B Setup prompt
echo "📊 Step 8: Weights & Biases setup..."
echo "Please run 'wandb login' to set up experiment tracking"

# Step 9: Verify HRM codebase
echo "🔍 Step 9: Verifying HRM codebase..."
if [ -f "pretrain.py" ]; then
    echo "✅ HRM codebase verified"
    python -c "
from utils.device import print_device_info
print_device_info()
"
else
    echo "❌ HRM codebase not found in current directory"
fi

# Step 10: Initialize submodules and create directories
echo "📚 Step 10: Initializing project structure..."
git submodule update --init --recursive
mkdir -p data logs checkpoints
echo "✅ Project structure initialized"

echo ""
echo "🎉 HRM Training Environment Setup Complete!"
echo "================================================"
echo "Next steps:"
echo "1. Activate environment: source venv_py310/bin/activate"
echo "2. Login to W&B: wandb login"
echo "3. Start training with: OMP_NUM_THREADS=8 python pretrain.py [options]"
echo ""
echo "Example training command:"
echo "OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=16 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0"