#!/bin/bash
# Extended 100-epoch training with optimizations

source venv_py310/bin/activate

echo "Starting extended 100-epoch training..."

python train_hrm_optimized.py \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --run-name "hrm-extended-100ep-optimized" \
    --data-path "data/livecodebench_real/livecodebench_real.json" \
    --max-batches-per-epoch 500 \
    > extended_training.log 2>&1 &

TRAINING_PID=$!
echo "✅ Extended training started with PID: $TRAINING_PID"
echo "📊 Monitor with: tail -f extended_training.log"
echo "🔍 Track metrics with: python monitor_training.py"
