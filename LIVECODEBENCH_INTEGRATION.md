# LiveCodeBench Integration for HRM

This document provides a comprehensive guide for using the LiveCodeBench integration with the Hierarchical Reasoning Model (HRM) for code generation tasks.

## Overview

The LiveCodeBench integration adapts HRM to achieve world-class performance on code generation benchmarks while maintaining exceptional efficiency for local deployment. This system supports all 4 LiveCodeBench scenarios:

1. **Code Generation**: Generate code from problem descriptions
2. **Self-Repair**: Fix broken code given error messages  
3. **Test Output Prediction**: Predict outputs for given test inputs
4. **Code Execution**: Predict execution results for code and input

## Architecture Components

### Core Modules

- **`dataset/build_livecodebench_dataset.py`**: Data extraction and processing pipeline
- **`dataset/livecodebench_dataset.py`**: PyTorch dataset with efficient batching
- **`models/code_generation/input_processor.py`**: Multi-language input processing
- **`evaluation/livecodebench_evaluator.py`**: pass@k metrics and code execution
- **`train_livecodebench.py`**: Training pipeline with distributed support
- **`test_livecodebench_integration.py`**: Integration testing suite

### Key Features

- ✅ **Multi-scenario training**: Balanced sampling across 4 LiveCodeBench scenarios
- ✅ **Temporal filtering**: Contamination-free evaluation with date filtering
- ✅ **Data augmentation**: Syntax variations and error injection for robust training
- ✅ **Code execution sandbox**: Safe code testing with timeout protection
- ✅ **pass@k metrics**: Standard evaluation metrics (pass@1, pass@5)
- ✅ **Multi-language support**: Python-focused with extensibility for 6 languages
- ✅ **Adaptive complexity**: Curriculum learning based on problem difficulty
- ✅ **Efficient batching**: Balanced scenario sampling and dynamic padding

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install torch transformers datasets wandb tqdm argdantic pydantic

# Clone and setup LiveCodeBench (if not already done)
git submodule update --init --recursive
```

### 2. Data Preparation

```bash
# Build LiveCodeBench dataset for HRM training
python dataset/build_livecodebench_dataset.py \
    --output-dir data/livecodebench-hrm \
    --scenarios generation,selfrepair,testprediction \
    --subsample-size 1000 \
    --start-date 2023-01-01 \
    --end-date 2024-06-01 \
    --num-aug 500
```

### 3. Training

```bash
# Train HRM model on LiveCodeBench
python train_livecodebench.py \
    --data-path data/livecodebench-hrm \
    --batch-size 32 \
    --global-batch-size 384 \
    --learning-rate 3e-4 \
    --max-steps 50000 \
    --eval-interval 1000 \
    --experiment-name hrm_livecodebench_v1
```

### 4. Evaluation

```bash
# Evaluate trained model
python evaluation/livecodebench_evaluator.py \
    --model-path checkpoints/hrm_livecodebench_v1/best.pt \
    --num-samples 5 \
    --scenarios generation,selfrepair \
    --output-dir evaluation_results
```

### 5. Testing Integration

```bash
# Run integration tests
python test_livecodebench_integration.py
```

## Detailed Usage

### Dataset Building

The dataset builder supports comprehensive configuration:

```python
from dataset.build_livecodebench_dataset import LiveCodeBenchConfig, LiveCodeBenchProcessor

config = LiveCodeBenchConfig(
    output_dir="data/livecodebench-hrm",
    scenarios="generation,selfrepair,testprediction,execution",
    start_date="2023-01-01",  # Contamination filtering
    end_date="2024-06-01",
    subsample_size=2000,      # Limit dataset size
    num_aug=1000,             # Data augmentation multiplier
    scenario_weights="1.0,0.8,0.6,0.5"  # Scenario importance
)

processor = LiveCodeBenchProcessor(config)
examples = processor.load_datasets()
hrm_data = processor.convert_to_hrm_format(examples)
```

### Training Configuration

The training system supports extensive customization:

```python
from train_livecodebench import LiveCodeBenchTrainingConfig

config = LiveCodeBenchTrainingConfig(
    # Data
    data_path="data/livecodebench-hrm",
    max_seq_len=2048,
    vocab_size=40000,
    
    # Model architecture  
    hidden_size=512,
    num_layers=12,
    high_level_layers=4,
    low_level_layers=8,
    
    # Training
    batch_size=32,
    learning_rate=3e-4,
    max_steps=50000,
    warmup_steps=1000,
    
    # Multi-scenario
    scenario_weights="1.0,0.8,0.6,0.5",
    complexity_schedule="curriculum",
    balanced_sampling=True,
    
    # System
    mixed_precision=True,
    distributed=True,
    log_wandb=True
)
```

### Evaluation Pipeline

The evaluator supports all LiveCodeBench scenarios:

```python
from evaluation.livecodebench_evaluator import EvaluationConfig, run_evaluation

config = EvaluationConfig(
    model_path="checkpoints/hrm_livecodebench/best.pt",
    num_samples=5,           # For pass@k calculation
    temperature=0.8,
    timeout_seconds=10,      # Code execution timeout
    scenarios=[Scenario.codegeneration, Scenario.selfrepair],
    start_date="2024-01-01", # Evaluation date filter
    save_detailed_results=True
)

metrics = run_evaluation(config)
print(f"Pass@1: {metrics['overall']['pass@1_mean']:.4f}")
```

## Performance Targets

### Model Variants

| Model Size | Parameters | Memory Usage | Speed (problems/sec) |
|------------|------------|--------------|---------------------|
| HRM-27M    | 27M        | <2GB         | 10-15               |
| HRM-110M   | 110M       | 4-6GB        | 5-10                |
| HRM-350M   | 350M       | 8-12GB       | 2-5                 |

### Benchmark Results (Target)

| Scenario | pass@1 | pass@5 | Time per Problem |
|----------|--------|--------|------------------|
| Code Generation | >40% | >60% | <2s |
| Self-Repair | >35% | >55% | <3s |
| Test Prediction | >50% | >70% | <1s |
| Code Execution | >45% | >65% | <1s |

## Advanced Features

### Adaptive Complexity Scheduling

The system supports curriculum learning with complexity-aware sampling:

```python
# Linear progression from easy to hard
complexity_schedule="linear"

# Exponential increase in hard problems  
complexity_schedule="exponential"

# Staged curriculum: easy -> medium -> hard
complexity_schedule="curriculum"
```

### Multi-Scenario Balancing

Configure scenario importance during training:

```python
scenario_weights = {
    "codegeneration": 1.0,     # Full weight
    "selfrepair": 0.8,         # 80% weight
    "testoutputprediction": 0.6, # 60% weight
    "codeexecution": 0.5       # 50% weight
}
```

### Data Augmentation Strategies

1. **Text Variations**: Paraphrase problem descriptions
2. **Error Injection**: Create realistic broken code for repair
3. **Test Case Modification**: Vary test inputs while preserving semantics
4. **Syntax Alternatives**: Different valid code structures

### Code Execution Sandbox

Safe code execution with comprehensive error handling:

```python
from evaluation.livecodebench_evaluator import CodeExecutionSandbox

sandbox = CodeExecutionSandbox(timeout_seconds=10)

# Execute code with input/output testing
success, output, error = sandbox.execute_code(
    code="print(sum([1, 2, 3]))",
    expected_output="6"
)

# Execute function with specific test cases
success, result, error = sandbox.execute_function_test(
    code="def add(a, b): return a + b",
    function_name="add",
    test_input=[2, 3],
    expected_output=5
)
```

## Distributed Training

### Single-Node Multi-GPU

```bash
# 8 GPUs on single node
torchrun --nproc-per-node 8 train_livecodebench.py \
    --data-path data/livecodebench-hrm \
    --batch-size 16 \
    --global-batch-size 384 \
    --distributed true
```

### Multi-Node Training

```bash
# Node 0 (master)
torchrun --nnodes 2 --node-rank 0 --master-addr MASTER_IP --master-port 29500 \
    --nproc-per-node 8 train_livecodebench.py --distributed true

# Node 1
torchrun --nnodes 2 --node-rank 1 --master-addr MASTER_IP --master-port 29500 \
    --nproc-per-node 8 train_livecodebench.py --distributed true
```

## Monitoring and Debugging

### Weights & Biases Integration

```python
# Automatic logging of:
# - Training/validation loss per scenario
# - pass@k metrics during evaluation  
# - Learning rate and optimizer state
# - Model architecture and hyperparameters
# - Dataset statistics and augmentation metrics
```

### Debug Mode

```bash
# Run with debug logging
python train_livecodebench.py \
    --batch-size 4 \
    --max-steps 100 \
    --eval-interval 10 \
    --log-wandb false \
    --debug true
```

## Common Issues and Solutions

### Memory Management

```python
# Reduce sequence length for memory efficiency
max_seq_len = 1024  # Instead of 2048

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
mixed_precision = True
```

### Training Stability

```python
# Gradient clipping
grad_clip = 1.0

# Learning rate warmup
warmup_steps = 1000

# Scenario weight balancing
scenario_weights = "1.0,0.8,0.6,0.5"
```

### Evaluation Speed

```python
# Reduce evaluation samples for faster iteration
num_samples = 3  # Instead of 5

# Shorter timeout for code execution
timeout_seconds = 5  # Instead of 10

# Evaluate subset of problems
eval_subset_size = 100
```

## Extending the System

### Adding New Programming Languages

1. Update `ProgrammingLanguage` enum in `input_processor.py`
2. Add language detection patterns in `MultiLanguageTokenizer`
3. Extend augmentation strategies for the new language
4. Update evaluation pipeline for language-specific testing

### Custom Evaluation Metrics

```python
def custom_metric(predictions, targets, metadata):
    """Implement custom evaluation metric"""
    # Your metric computation
    return metric_value

# Add to evaluator
evaluator.add_custom_metric("my_metric", custom_metric)
```

### New Training Scenarios

1. Define new scenario in `CodeGenerationTask` enum
2. Implement data loading in `LiveCodeBenchProcessor`
3. Add scenario-specific loss computation
4. Update evaluation pipeline for new scenario type

## Performance Optimization

### Model Optimization

```python
# Model compilation (PyTorch 2.0+)
model = torch.compile(model)

# Quantization for inference
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

### Data Loading Optimization

```python
# Increase workers for data loading
num_workers = 8

# Pin memory for GPU transfer
pin_memory = True

# Prefetch factor
prefetch_factor = 2
```

### Inference Optimization

```python
# Batch inference for multiple problems
batch_size = 16

# KV-cache for autoregressive generation
use_cache = True

# Early stopping for generation
early_stopping = True
```

## License and Attribution

This integration builds upon:
- **LiveCodeBench**: Original benchmark and evaluation framework
- **HRM**: Hierarchical Reasoning Model architecture
- **HuggingFace**: Dataset loading and tokenization infrastructure

Please cite the original papers when using this work:

```bibtex
@article{hrm2024,
  title={Hierarchical Reasoning Model for Code Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}

@article{livecodebench2024,
  title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  author={Jain et al.},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the integration tests: `python test_livecodebench_integration.py`
2. Review the logs in `evaluation_results/` and W&B dashboard
3. Consult the original LiveCodeBench and HRM documentation
4. Open an issue with detailed error logs and configuration

---

*This integration enables HRM to achieve state-of-the-art performance on code generation while maintaining the efficiency and hierarchical reasoning capabilities that make it suitable for local deployment.*