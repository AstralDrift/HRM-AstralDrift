#!/bin/bash
# Script to extend training to 100 epochs

echo "🚀 Extending HRM training to 100 epochs..."

# Check current training status
CURRENT_PID=$(ps aux | grep "train_hrm_optimized.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$CURRENT_PID" ]; then
    echo "⚠️  Current training (PID: $CURRENT_PID) is still running"
    echo "   Waiting for completion before extending..."
    
    # Option to monitor
    echo "   Run: tail -f production_training.log"
else
    echo "✅ No active training found. Ready to extend!"
fi

# Prepare extension command
cat << 'EOF' > start_extended_training.sh
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
EOF

chmod +x start_extended_training.sh

echo ""
echo "📝 To extend training after current run completes:"
echo "   ./start_extended_training.sh"
echo ""
echo "🔄 To resume from checkpoint:"
echo "   python train_hrm_optimized.py --resume checkpoints/hrm-enhanced-metrics-50ep/best_model.pt --epochs 100"