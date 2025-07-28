# SWE-bench Analysis Report

## Overview

SWE-bench is a comprehensive benchmark for evaluating language models on real-world software engineering tasks. It tests models' ability to resolve GitHub issues by generating code patches that fix bugs or implement features in existing Python repositories.

## Key Characteristics

### Task Definition
- **Input**: GitHub issue description + complete codebase snapshot
- **Output**: Code patch (in .patch format) that resolves the issue
- **Evaluation**: Patch must apply successfully and pass all unit tests

### Scale and Scope
- **2,294 task instances** across 12 popular Python repositories
- **Real-world issues** from popular open-source projects
- **Complex multi-file edits** requiring cross-context understanding
- **Large codebases** with thousands of files and hundreds of thousands of lines

### Unique Properties
- **Realistic software engineering**: Unlike synthetic coding benchmarks
- **Execution-based evaluation**: Tests must pass, not just syntactic correctness
- **Long context requirements**: Average 438K lines of code per repository
- **Multi-file reasoning**: Solutions often span multiple functions/files
- **Continually updatable**: Can add new tasks from ongoing GitHub activity

## Repository Analysis

### Structure
- **12 Python repositories**: astropy, django, flask, matplotlib, pylint, pytest, requests, scikit-learn, seaborn, sphinx, sympy, xarray
- **Diverse domains**: Web frameworks, ML libraries, scientific computing, documentation tools
- **Well-maintained**: Popular projects with extensive test coverage

### Task Statistics
| Metric | Average | Maximum |
|--------|---------|---------|
| Issue length (words) | 195.1 | 4,477 |
| Codebase files | 3,010 | 5,890 |
| Lines of code | 438K | 886K |
| Files edited | 1.7 | 31 |
| Functions edited | 3.0 | 36 |
| Lines edited | 32.8 | 5,888 |
| Fail-to-pass tests | 9.1 | 1,633 |
| Total tests | 120.8 | 9,459 |

## Dataset Details

### HuggingFace Dataset
- **Location**: `princeton-nlp/SWE-bench`
- **Total instances**: 21.5K (19K train, 2.29K test, 225 dev)
- **Format**: Parquet files with structured metadata

### Key Fields
- `repo`: Repository identifier
- `instance_id`: Unique task identifier  
- `base_commit`: Starting commit hash
- `patch`: Reference solution patch
- `problem_statement`: Issue description
- `hints_text`: Optional guidance
- `created_at`: Timestamp
- `version`: Dataset version

### Variants
- **SWE-bench**: Full dataset (2,294 instances)
- **SWE-bench Lite**: Smaller subset for quick evaluation
- **SWE-bench Verified**: Human-verified subset
- **SWE-bench Multimodal**: Includes issues with images

## Evaluation Methodology

### Setup Requirements
- **Python 3.8+**
- **Docker** (mandatory for isolated evaluation)
- **Hardware**: 120GB storage, 16GB RAM, 8 CPU cores

### Evaluation Pipeline
1. Apply generated patch to codebase
2. Run repository's test suite
3. Check if fail-to-pass tests now pass
4. Verify existing tests still pass
5. Calculate success rate

### Metrics
- **Primary**: Percentage of issues resolved (all tests pass)
- **Secondary**: Patch application rate, partial success analysis

## Model Performance Baselines

### Current State-of-the-Art (2025)
| Model | Success Rate | Notes |
|-------|--------------|-------|
| TRAE | 75.2% | Current leaderboard leader |
| Refact.ai Agent | 74.4% | AI coding agent |
| Claude 4 Sonnet | 72.7% (80.2% parallel) | Anthropic's latest |
| Claude 4 Opus | 72.5% (79.4% parallel) | High-capability model |
| OpenAI O3 Pro | ~72% | OpenAI's reasoning model |

### Historical Baselines (2024 - Original Paper)
| Model | Success Rate | Apply Rate |
|-------|--------------|------------|
| Claude 2 | 1.96% | 43.07% |
| GPT-4 | 0.00% | 14.83% |
| SWE-Llama 13B | 0.70% | 53.62% |
| SWE-Llama 7B | 0.70% | 51.74% |
| ChatGPT-3.5 | 0.17% | 26.33% |

### Performance Evolution Insights
- **Dramatic improvement**: From <2% to >70% success rates in ~1 year
- **Parallel computation**: Enables 7-8% additional performance gains
- **Model scale matters**: Latest frontier models show consistent 70%+ performance
- **Agent architectures**: Tool-using agents (TRAE, Refact.ai) lead leaderboard

## Technical Challenges

### For HRM Integration

#### Context Length Requirements
- **Average codebase**: 438K lines of code
- **Context window**: Need 100K+ tokens for full context
- **Retrieval necessity**: Must identify relevant files from thousands

#### Reasoning Complexity
- **Cross-file dependencies**: Understanding how functions interact across modules
- **Large-scale edits**: Solutions span multiple files and functions
- **Test understanding**: Must grasp what tests are checking

#### Code Generation Demands
- **Patch format**: Must generate valid .patch files
- **Multi-language support**: Handle different Python coding styles
- **Integration testing**: Changes must not break existing functionality

## HRM Adaptation Strategy

### Hierarchical Decomposition
- **High-level reasoning**: Issue understanding, solution planning, file identification
- **Low-level implementation**: Specific code changes, syntax, patch formatting

### Multi-Stage Approach
1. **Issue analysis**: Parse problem statement and identify affected components
2. **Code exploration**: Navigate codebase to understand relevant context
3. **Solution design**: Plan multi-file changes and dependencies
4. **Implementation**: Generate specific code modifications
5. **Validation**: Verify patch format and logical consistency

### Training Considerations
- **Code-specific training data**: Use SWE-bench training set (19K instances)
- **Multi-task learning**: Combine with existing code generation benchmarks
- **Retrieval integration**: Train with both full context and retrieved context
- **Patch format training**: Ensure model learns proper .patch syntax

## Integration with HRM Project

### Immediate Benefits
- **Real-world validation**: Test HRM on authentic software engineering tasks
- **Multi-modal reasoning**: Combine high-level planning with detailed implementation
- **Scalability testing**: Evaluate performance on large codebases

### Long-term Goals
- **Production readiness**: Demonstrate practical software engineering capability
- **Industry relevance**: Show value for real development workflows
- **Model efficiency**: Achieve strong performance with <100M parameters

### Success Metrics for HRM
- **Target performance**: >60% success rate on SWE-bench (ambitious but achievable given HRM's proven revolutionary capabilities)
- **Stretch goal**: Match or exceed Claude 4 Sonnet performance (72.7%) with 370x fewer parameters
- **Efficiency revolution**: Achieve SOTA-competitive performance with <100M parameters vs multi-billion parameter models
- **Speed**: <1 minute per problem on consumer hardware
- **Local deployment**: <2GB memory usage during evaluation
- **Quantization**: <5% performance loss at 4-bit quantization

## Recommended Implementation Steps

### Phase 1: Setup and Baseline
1. Install SWE-bench evaluation harness
2. Implement basic patch generation pipeline
3. Establish baseline performance with existing HRM

### Phase 2: Architecture Adaptation
1. Modify HRM for software engineering tasks
2. Implement hierarchical reasoning for code understanding
3. Add patch generation capabilities

### Phase 3: Training and Optimization
1. Fine-tune on SWE-bench training data
2. Implement retrieval-augmented generation
3. Optimize for long context handling

### Phase 4: Evaluation and Analysis
1. Comprehensive evaluation on SWE-bench test set
2. Error analysis and failure mode identification
3. Performance comparison with existing models

## SWE-smith Integration

### Training Data Generation
SWE-smith (https://github.com/SWE-bench/SWE-smith) provides a toolkit for generating unlimited software engineering training data:

#### Key Features
- **Scalable data generation**: 52,000+ task instances across repositories
- **GitHub repository conversion**: Turn any repo into a "SWE-gym"
- **Automated validation**: Unit test generation and issue text synthesis
- **Agent trajectory collection**: 26,000 SWE-agent trajectories for training

#### Training Success Stories
- **Qwen 2.5 Coder → SWE-agent-LM-32B**: Achieved 40.2% pass@1 on SWE-bench Verified
- **Methodology proven**: Demonstrates effectiveness of synthetic data generation

#### HRM Integration Benefits
- **Data augmentation**: Generate additional training instances for HRM
- **Repository diversity**: Expand beyond the 12 core SWE-bench repositories
- **Custom task creation**: Generate domain-specific software engineering tasks
- **Hierarchical reasoning training**: Create tasks that specifically exercise HRM's two-level architecture

#### Technical Requirements
- **Ubuntu 22.04.4 LTS**: Currently Linux-only (no Windows/MacOS support)
- **Docker environment**: Required for task instance creation
- **Python framework**: Simple integration with existing HRM training pipeline

## Conclusion

SWE-bench represents a significant leap in complexity from traditional code generation benchmarks, with current SOTA models achieving 70%+ success rates. It tests models' ability to understand large codebases, reason about complex software engineering problems, and generate multi-file solutions that maintain existing functionality while implementing new features or fixing bugs.

For the HRM project, SWE-bench offers an exceptional opportunity to demonstrate revolutionary efficiency. **Given HRM's proven ability to beat SOTA models on ARC-AGI and complex reasoning tasks with only 27M parameters, we should aim high**: targeting >60% success rate and competing directly with Claude 4 Sonnet's 72.7% performance while using 370x fewer parameters.

This ambitious target reflects HRM's demonstrated revolutionary potential. If HRM can achieve world-class performance on complex reasoning with minimal parameters, there's no reason to settle for conservative goals on software engineering tasks. The dramatic industry improvements (from <2% to >70% in one year) show rapid progress is possible with the right architectural innovations.

The combination of SWE-bench evaluation and SWE-smith data generation provides a complete ecosystem for developing and testing software engineering AI capabilities. Achieving SOTA-competitive performance on SWE-bench would definitively prove HRM's revolutionary potential and establish a new paradigm for efficient, locally-deployable AI systems that can compete with multi-billion parameter models.