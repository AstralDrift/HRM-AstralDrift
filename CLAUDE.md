# HRM-AstralDrift Claude Memory

## Project Overview
**Hierarchical Reasoning Model (HRM)** - A novel recurrent architecture for sequential reasoning tasks that achieves significant computational depth while maintaining training stability and efficiency.

**NEW FOCUS: Code Generation & Tool Use Excellence** - Adapting HRM to achieve world-class performance on code generation and CLI tool use benchmarks while maintaining exceptional efficiency for local deployment.

### Key Architecture
- **Two-level hierarchical system**: High-level (slow, abstract planning) + Low-level (rapid, detailed computations)
- **Single forward pass execution** without explicit supervision of intermediate processes
- **27M parameters** achieving exceptional performance with only 1000 training samples
- **No pre-training or CoT data required**

### Core Components
- `models/hrm/hrm_act_v1.py` - Main HRM model implementation with ACT (Adaptive Computation Time) wrapper
- `pretrain.py` - Training pipeline with distributed support
- `evaluate.py` - Evaluation pipeline for trained models
- `code_generation_dataset.py` - Dataset handling for code generation tasks

### Supported Tasks

#### Code Generation & Tool Use Tasks
1. **LiveCodeBench** - 400+ coding challenges from LeetCode, AtCoder, CodeForces
   - Code generation, self-repair, test prediction, execution scenarios
   - Target: pass@1 and pass@5 metrics with contamination-free evaluation
2. **Polyglot Benchmark** - 225 challenging Exercism problems across 6 languages
   - C++, Go, Java, JavaScript, Python, Rust
   - Diff-based code editing (search-replace operations)
   - Target: >80% success rate across all languages
3. **SWE-bench** - Real-world GitHub issue resolution
   - 2,294 software engineering problems from 12 popular Python repositories
   - Multi-file patch generation for bug fixes and feature implementations
   - Current SOTA: 75%+ (TRAE, Claude 4), Target: >60% success rate with <100M parameters
4. **CLI & Tool Use** - Development workflow automation
   - Git operations, package managers, build systems, debugging tools
   - Multi-step complex development workflows

### Project Goals & Success Metrics

#### Performance Targets
- **LiveCodeBench**: Top-3 performance with <100M parameters
- **Polyglot**: >80% success rate across all 6 languages
- **SWE-bench**: >60% success rate with efficient long-context processing (competing with Claude 4 Sonnet's 72.7%)
- **Local Deployment**: <2GB memory usage, <1s per problem on consumer hardware
- **Quantization**: <5% performance loss at 4-bit quantization

#### Code Generation Adaptations
- **High-Level Module**: Strategic code planning (algorithm selection, architecture decisions)
- **Low-Level Module**: Implementation details (syntax, API calls, specific logic)
- **Multi-Language Support**: Efficient handling of 6 programming languages
- **Tool Integration**: CLI command planning and execution workflows

### Training Commands
```bash
# Mixed Training (recommended - 70% SWE-Smith + 30% LiveCodeBench)
python train_hrm_optimized.py --epochs 40 --batch-size 6 --learning-rate 4e-5 --data-path mixed

# RTX 4090 Optimized Training (maximum performance)
./launch_rtx4090_training.sh

# Manual optimized training with all features
python train_hrm_optimized.py --epochs 40 --batch-size 8 --learning-rate 4e-5 --data-path mixed --mixed-precision --gradient-checkpointing
```

### Hardware Optimization
```bash
# Setup RTX 4090 optimizations
python utils/rtx4090_optimization.py

# Monitor GPU during training (run in separate terminal)
python monitor_rtx4090.py

# Verify system setup
python test_memory_optimizations.py
```

### Dependencies
- PyTorch with CUDA
- FlashAttention (v2 for Ampere, v3 for Hopper GPUs)
- W&B for experiment tracking
- Standard ML libraries (einops, tqdm, pydantic, etc.)

### Model Checkpoints
- Available on HuggingFace: LiveCodeBench-Optimized, SWE-Smith-1k, Polyglot-Benchmark

## Technical Notes
- Uses `HierarchicalReasoningModel_ACTV1` as main model class
- ACT wrapper enables adaptive computation with Q-learning for halting decisions
- Distributed training supported via PyTorch DDP
- Non-autoregressive architecture (causal=False)
- RMSNorm + SwiGLU activation
- Rotary or learned positional embeddings
- Dynamic code embeddings for multi-language support

## For Training New Models
1. Use `train_hrm_optimized.py` with appropriate dataset path
2. Configure with `config/arch/hrm_swe_search.yaml` for advanced features
3. Monitor with W&B
4. Evaluate with `evaluate.py` and code generation metrics

## Architecture Insights
- High-level module: Strategic planning with fewer cycles
- Low-level module: Detailed computation with more cycles  
- Input injection combines puzzle embeddings + token embeddings
- Q-learning based halting mechanism for adaptive computation
- Single gradient step optimization with momentum carry-over

## Code Generation Project Documentation

### Master Planning Documents
- `HRM_CODE_GENERATION_PLAN.md` - Comprehensive adaptation strategy and roadmap
- `TASK_BREAKDOWN.md` - Atomic, actionable tasks broken down by phase
- `CHANGELOG.md` - Project evolution and decision tracking

### Research & Analysis
- `RESEARCH_NOTES.md` - Ongoing findings and technical insights
- `LIVECODEBENCH_ANALYSIS.md` - Detailed LiveCodeBench requirements
- `POLYGLOT_ANALYSIS.md` - Multi-language benchmark analysis
- `SWE_BENCH_ANALYSIS.md` - Real-world software engineering benchmark analysis

### Project Management
- `TODO_TRACKER.md` - Current sprint and task management
- `PROGRESS_LOG.md` - Completed work and lessons learned
- `ISSUES_AND_BLOCKERS.md` - Problem tracking and solutions

### Development Philosophy
"Rome wasn't built in a day" - Taking time to think thoroughly and break massive tasks into smallest actionable pieces for effective incremental building.