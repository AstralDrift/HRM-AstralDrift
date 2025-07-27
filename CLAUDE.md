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
- `puzzle_dataset.py` - Dataset handling for reasoning tasks

### Supported Tasks

#### Original Reasoning Tasks
1. **ARC-AGI** (Abstraction and Reasoning Corpus) - Key AGI benchmark
2. **Sudoku** - Complex 9x9 extreme puzzles
3. **Maze** - Optimal pathfinding in large mazes

#### NEW: Code Generation & Tool Use Tasks
4. **LiveCodeBench** - 400+ coding challenges from LeetCode, AtCoder, CodeForces
   - Code generation, self-repair, test prediction, execution scenarios
   - Target: pass@1 and pass@5 metrics with contamination-free evaluation
5. **Polyglot Benchmark** - 225 challenging Exercism problems across 6 languages
   - C++, Go, Java, JavaScript, Python, Rust
   - Diff-based code editing (search-replace operations)
   - Target: >80% success rate across all languages
6. **CLI & Tool Use** - Development workflow automation
   - Git operations, package managers, build systems, debugging tools
   - Multi-step complex development workflows

### Project Goals & Success Metrics

#### Performance Targets
- **LiveCodeBench**: Top-3 performance with <100M parameters
- **Polyglot**: >80% success rate across all 6 languages
- **Local Deployment**: <2GB memory usage, <1s per problem on consumer hardware
- **Quantization**: <5% performance loss at 4-bit quantization

#### Code Generation Adaptations
- **High-Level Module**: Strategic code planning (algorithm selection, architecture decisions)
- **Low-Level Module**: Implementation details (syntax, API calls, specific logic)
- **Multi-Language Support**: Efficient handling of 6 programming languages
- **Tool Integration**: CLI command planning and execution workflows

### Training Commands
```bash
# Sudoku (laptop GPU friendly)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# ARC-AGI (8-GPU setup)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

### Dependencies
- PyTorch with CUDA
- FlashAttention (v2 for Ampere, v3 for Hopper GPUs)
- W&B for experiment tracking
- Standard ML libraries (einops, tqdm, pydantic, etc.)

### Model Checkpoints
- Available on HuggingFace: ARC-AGI-2, Sudoku Extreme, Maze 30x30 Hard

## Technical Notes
- Uses `HierarchicalReasoningModel_ACTV1` as main model class
- ACT wrapper enables adaptive computation with Q-learning for halting decisions
- Distributed training supported via PyTorch DDP
- Non-autoregressive architecture (causal=False)
- RMSNorm + SwiGLU activation
- Rotary or learned positional embeddings
- Sparse puzzle embeddings for task-specific adaptation

## For Training New Models
1. Build dataset using appropriate `dataset/build_*_dataset.py` script
2. Use `pretrain.py` with proper config (see `config/cfg_pretrain.yaml`)
3. Monitor with W&B
4. Evaluate with `evaluate.py` and `arc_eval.ipynb` for ARC tasks

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

### Project Management
- `TODO_TRACKER.md` - Current sprint and task management
- `PROGRESS_LOG.md` - Completed work and lessons learned
- `ISSUES_AND_BLOCKERS.md` - Problem tracking and solutions

### Development Philosophy
"Rome wasn't built in a day" - Taking time to think thoroughly and break massive tasks into smallest actionable pieces for effective incremental building.