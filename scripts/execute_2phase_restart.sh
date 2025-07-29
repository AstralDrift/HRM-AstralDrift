#!/bin/bash
# Execute 2-phase HRM training restart with all preparations

echo "🚀 HRM 2-Phase Training Restart"
echo "================================"
echo "⚠️  This will restart training after current job completes"
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pre_tokenizer_restart_${TIMESTAMP}"
PHASE1_NAME="hrm-2phase-p1-${TIMESTAMP}"
PHASE2_NAME="hrm-2phase-p2-${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if current training is running
print_step 1 "Checking current training status"
CURRENT_PID=$(ps aux | grep "train_hrm_optimized.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$CURRENT_PID" ]; then
    print_warning "Current training (PID: $CURRENT_PID) is still running"
    echo "   Please wait for completion or stop manually with: kill $CURRENT_PID"
    echo "   Monitor progress: tail -f production_training.log"
    echo "   Check stalls: python scripts/monitor_training_advanced.py"
    echo ""
    echo "🔄 Run this script again when training completes"
    exit 1
fi

print_success "No active training found - ready to proceed"

# Step 2: Backup existing checkpoints
print_step 2 "Creating checkpoint backup"
python scripts/backup_checkpoints.py --action backup --name "$BACKUP_NAME"

if [ $? -ne 0 ]; then
    print_error "Backup failed - aborting restart"
    exit 1
fi

print_success "Checkpoints backed up as: $BACKUP_NAME"

# Step 3: Fetch LiveCodeBench dataset
print_step 3 "Fetching LiveCodeBench dataset"
if [ ! -f "data/livecodebench_real/livecodebench_real.json" ]; then
    python scripts/fetch_real_livecodebench.py --version release_v2 --use-lite
    
    if [ $? -ne 0 ]; then
        print_error "LiveCodeBench fetch failed - using fallback dataset"
        print_warning "Will proceed with existing dataset if available"
    else
        print_success "LiveCodeBench dataset fetched and converted"
    fi
else
    print_success "LiveCodeBench dataset already exists"
fi

# Step 4: Activate virtual environment
print_step 4 "Activating virtual environment"
if [ -d "venv_py310" ]; then
    source venv_py310/bin/activate
    print_success "Virtual environment activated"
else
    print_warning "Virtual environment not found - proceeding with system Python"
fi

# Step 5: Start Phase 1 training
print_step 5 "Starting Phase 1 training (Epochs 1-25)"

# Determine dataset path
DATASET_PATH="data/livecodebench_real/livecodebench_real.json"
if [ ! -f "$DATASET_PATH" ]; then
    print_warning "LiveCodeBench not found, falling back to existing dataset"
    DATASET_PATH="data/livecodebench_real/livecodebench_real.json"
    # Could add more fallback paths here
fi

echo "   Dataset: $DATASET_PATH"
echo "   Run name: $PHASE1_NAME"
echo "   Log file: phase1_training.log"
echo ""

nohup python train_hrm_optimized.py \
    --epochs 25 \
    --batch-size 6 \
    --learning-rate 4e-5 \
    --run-name "$PHASE1_NAME" \
    --data-path "$DATASET_PATH" \
    --config config/optimized_restart_config.json \
    > phase1_training.log 2>&1 &

PHASE1_PID=$!

print_success "Phase 1 training started!"
echo "   PID: $PHASE1_PID"
echo "   Monitor: tail -f phase1_training.log"
echo "   Advanced monitor: python scripts/monitor_training_advanced.py"

# Step 6: Set up monitoring
print_step 6 "Setting up monitoring"

# Create monitoring script for phase transition
cat > monitor_phase_transition.sh << EOF
#!/bin/bash
# Auto-generated phase transition monitor

echo "🔍 Monitoring Phase 1 completion..."
echo "Training PID: $PHASE1_PID"
echo ""

while true; do
    # Check if Phase 1 process is still running
    if ! kill -0 $PHASE1_PID 2>/dev/null; then
        echo "✅ Phase 1 training completed"
        break
    fi
    
    # Check for completion in log
    if grep -q "Epoch 25" phase1_training.log 2>/dev/null; then
        echo "✅ Phase 1 reached Epoch 25"
        break
    fi
    
    # Show latest progress
    LATEST_LINE=\$(tail -1 phase1_training.log 2>/dev/null || echo "No log yet")
    echo "\$(date '+%H:%M:%S') - \$LATEST_LINE"
    
    sleep 300  # Check every 5 minutes
done

echo ""
echo "🎯 Phase 1 complete! Ready for Phase 2..."
echo "Run: bash start_phase2.sh"
EOF

chmod +x monitor_phase_transition.sh

# Create Phase 2 startup script
cat > start_phase2.sh << EOF
#!/bin/bash
# Auto-generated Phase 2 startup

echo "🚀 Starting Phase 2 (LiveCodeBench Integration)"
echo "=============================================="

# Check for Phase 1 checkpoint
CHECKPOINT_PATH="checkpoints/$PHASE1_NAME/epoch_25.pt"
if [ ! -f "\$CHECKPOINT_PATH" ]; then
    echo "❌ Phase 1 checkpoint not found: \$CHECKPOINT_PATH"
    echo "   Check available checkpoints:"
    ls -la checkpoints/$PHASE1_NAME/ 2>/dev/null || echo "   No checkpoints directory found"
    exit 1
fi

echo "✅ Found Phase 1 checkpoint: \$CHECKPOINT_PATH"
echo ""

# Start Phase 2
nohup python train_hrm_optimized.py \\
    --resume "\$CHECKPOINT_PATH" \\
    --epochs 50 \\
    --learning-rate 2e-5 \\
    --run-name "$PHASE2_NAME" \\
    --enable-livecodebench \\
    > phase2_training.log 2>&1 &

PHASE2_PID=\$!

echo "✅ Phase 2 training started!"
echo "   PID: \$PHASE2_PID"
echo "   Monitor: tail -f phase2_training.log"
echo "   Expected completion: ~12-24 hours"
echo ""
echo "🔍 Monitoring commands:"
echo "   python scripts/monitor_training_advanced.py --continuous"
echo "   python scripts/predict_final_metrics.py --target-epoch 50"
EOF

chmod +x start_phase2.sh

print_success "Monitoring and Phase 2 scripts created"

# Final summary
echo ""
echo "🎉 2-Phase Training Restart Complete!"
echo "===================================="
echo ""
echo "📊 Current Status:"
echo "   Phase 1 PID: $PHASE1_PID"
echo "   Backup: $BACKUP_NAME"
echo "   Dataset: $DATASET_PATH"
echo ""
echo "🔍 Monitoring:"
echo "   Live log: tail -f phase1_training.log"
echo "   Auto monitor: ./monitor_phase_transition.sh"
echo "   Advanced: python scripts/monitor_training_advanced.py --continuous"
echo ""
echo "⏭️  Next Steps:"
echo "   1. Monitor Phase 1 progress (~6-12 hours)"
echo "   2. Run ./start_phase2.sh when Phase 1 completes"
echo "   3. Evaluate at Epoch 50: python scripts/evaluate_epoch50_checkpoint.py"
echo ""
echo "📈 Expected Epoch 50 Metrics:"
echo "   Loss: ~0.52, Syntax: ~78%, Compilation: ~68%"
echo "   Combined Score: ~0.73 (target: 0.85)"
echo ""
echo "✅ Training restart initiated successfully!"