#!/bin/bash

# HRM Production Training Script
# Usage: ./start_production_training.sh [epochs] [run_name]

EPOCHS=${1:-50}
RUN_NAME=${2:-"hrm-production-$(date +%Y%m%d-%H%M%S)"}

echo "🚀 Starting HRM Production Training"
echo "   Epochs: $EPOCHS"
echo "   Run Name: $RUN_NAME"
echo "   Log: hrm_training.log"

# Activate virtual environment and start training
source venv_py310/bin/activate
python train_hrm_optimized.py \
    --epochs $EPOCHS \
    --run-name "$RUN_NAME" \
    --use-wandb \
    2>&1 | tee "logs/training_${RUN_NAME}.log"

echo "✅ Training completed. Check logs/training_${RUN_NAME}.log for details."