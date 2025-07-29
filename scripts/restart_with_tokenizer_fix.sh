#!/bin/bash
# Script to restart training with the tokenizer fix

echo "🚀 Restarting HRM Training with Tokenizer Fix"
echo "============================================="

# Check if current training is still running
CURRENT_PID=$(ps aux | grep "train_hrm_optimized.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$CURRENT_PID" ]; then
    echo "⚠️  Current training (PID: $CURRENT_PID) is still running"
    echo "Please wait for it to complete or stop it manually:"
    echo "   kill $CURRENT_PID"
    echo ""
    echo "Current progress:"
    tail -5 production_training.log 2>/dev/null || echo "   No log file found"
    exit 1
fi

echo "✅ No active training found. Ready to restart!"

# Activate virtual environment
source venv_py310/bin/activate

# Create backup of current checkpoints
BACKUP_DIR="checkpoints/backup_pre_tokenizer_fix_$(date +%Y%m%d_%H%M%S)"
if [ -d "checkpoints/hrm-production-run" ]; then
    echo "📦 Creating backup: $BACKUP_DIR"
    cp -r checkpoints/hrm-production-run "$BACKUP_DIR"
fi

# Start new training with tokenizer fix
echo "🔥 Starting training with tokenizer fix..."

RUN_NAME="hrm-tokenizer-fixed-$(date +%Y%m%d-%H%M%S)"

nohup python train_hrm_optimized.py \
    --epochs 50 \
    --batch-size 6 \
    --learning-rate 4e-5 \
    --run-name "$RUN_NAME" \
    --data-path "data/livecodebench_real/livecodebench_real.json" \
    > "tokenizer_fixed_training.log" 2>&1 &

TRAINING_PID=$!

echo "✅ Training restarted with tokenizer fix!"
echo "📊 Details:"
echo "   PID: $TRAINING_PID"
echo "   Run name: $RUN_NAME"
echo "   Log file: tokenizer_fixed_training.log"
echo "   Using real LiveCodeBench dataset"
echo ""
echo "🔍 Monitor progress:"
echo "   tail -f tokenizer_fixed_training.log"
echo "   python monitor_training.py"
echo ""
echo "🎯 Expected improvements:"
echo "   - Syntax validity: 0% → 30-70%"
echo "   - Compilation success: 0% → 20-50%"
echo "   - Better code-specific feedback"
echo ""
echo "⏱️  Check progress in ~30 minutes for first metrics"