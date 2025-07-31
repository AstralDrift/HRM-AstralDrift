#!/bin/bash
# Restart HRM training with tokenizer fix from best_model.pt (Epoch 29)
# Loss: 0.37, Token Acc: 99.9%, Tiered: 0.996

echo "🔄 Restarting HRM training with tokenizer fix..."
echo "📍 Resuming from: checkpoints/hrm-enhanced-metrics-50ep/best_model.pt"
echo "🎯 Expected: Syntax/compilation metrics > 0% in first epoch"

# Set environment
export CUDA_LAUNCH_BLOCKING=0
export MPS_AVAILABLE=1

# Create new checkpoint directory for restart
RESTART_DIR="checkpoints/hrm-tokenizer-fix-restart-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESTART_DIR"

echo "💾 Checkpoint directory: $RESTART_DIR"

# Launch training with tokenizer fix
python3 train_hrm_optimized.py \
    --epochs 100 \
    --batch-size 6 \
    --learning-rate 4e-5 \
    --data-path mixed \
    --mixed-precision \
    --gradient-checkpointing \
    --checkpoint-dir "$RESTART_DIR" \
    --resume-from "checkpoints/hrm-enhanced-metrics-50ep/best_model.pt" \
    --save-every 5 \
    --eval-every 2 \
    --use-wandb \
    2>&1 | tee "$RESTART_DIR/training.log"

echo "✅ Training restart completed!"
echo "📊 Check logs in: $RESTART_DIR/training.log"