#!/bin/bash
# RTX 4090 Optimized HRM Training Launch Script

echo "🚀 Starting RTX 4090 Optimized HRM Training"
echo "Device: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Set optimal environment variables
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
export CUDA_VISIBLE_DEVICES=0

# Launch training with optimized config
python train_hrm_optimized.py \
    --data-path mixed \
    --epochs 40 \
    --batch-size 4 \
    --learning-rate 4e-05 \
    --mixed-precision \
    --gradient-checkpointing \
    --use-wandb

echo "✅ Training completed"
