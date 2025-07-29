#!/bin/bash
# Fetch and prepare LiveCodeBench dataset for integration

echo "📦 Fetching LiveCodeBench Dataset"
echo "================================="

# Create data directory structure
mkdir -p data/external
mkdir -p data/livecodebench_enhanced
mkdir -p evaluations/livecodebench

# Clone LiveCodeBench repository
if [ ! -d "data/external/LiveCodeBench" ]; then
    echo "🔄 Cloning LiveCodeBench repository..."
    cd data/external
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git
    cd ../..
    echo "✅ LiveCodeBench cloned successfully"
else
    echo "✅ LiveCodeBench already exists, updating..."
    cd data/external/LiveCodeBench
    git pull origin main
    cd ../../..
fi

# Check dataset structure
echo ""
echo "📊 Dataset Overview:"
echo "   Problems: $(find data/external/LiveCodeBench -name '*.json' | wc -l)"
echo "   Languages: $(ls data/external/LiveCodeBench/data/ 2>/dev/null | head -5 | tr '\n' ' ')"
echo "   Size: $(du -sh data/external/LiveCodeBench 2>/dev/null | cut -f1)"

# Process for HRM integration
echo ""
echo "🔧 Processing for HRM integration..."
python scripts/process_livecodebench.py \
    --input data/external/LiveCodeBench \
    --output data/livecodebench_enhanced \
    --format hrm \
    --augment-samples 3 \
    --include-tool-use

echo ""
echo "✅ LiveCodeBench preparation complete!"
echo "📝 Next steps:"
echo "   1. Review processed data in data/livecodebench_enhanced/"
echo "   2. Update training config to use enhanced dataset"
echo "   3. Start phase 3 training (epochs 51-75) with LiveCodeBench"
echo ""
echo "🔍 Monitor integration:"
echo "   python scripts/validate_livecodebench_integration.py"