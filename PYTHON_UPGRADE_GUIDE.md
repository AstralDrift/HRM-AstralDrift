# Python 3.10+ Environment Setup Guide

This guide explains how to use the upgraded Python 3.10 environment for full SWE-smith compatibility and enhanced HRM multi-agent functionality.

## Environment Overview

We have successfully upgraded from Python 3.9.7 to Python 3.10.17 to meet SWE-smith's requirements and enable full compatibility with the multi-agent architecture.

### Key Benefits of Python 3.10+:
- **Full SWE-smith Compatibility**: All tree-sitter dependencies work correctly
- **Enhanced Performance**: Improved async/await and pattern matching features
- **Better Type Hints**: Structural pattern matching and union types
- **Modern Libraries**: Access to latest versions of ML and development libraries

## Quick Start

### 1. Activate the Python 3.10 Environment

```bash
# From the project root directory
source venv_py310/bin/activate
```

### 2. Verify Installation

```bash
# Check Python version
python --version
# Should output: Python 3.10.17

# Test SWE-smith import
python -c "import swesmith; print('SWE-smith ready!')"

# Test HRM multi-agent system
python -c "from agent_runtime.swe_rex_integration import HRMMultiAgentOrchestrator; print('Multi-agent system ready!')"
```

### 3. Run Multi-Agent Demo

```bash
# Activate environment
source venv_py310/bin/activate

# Run basic functionality test
python simple_test.py

# Run full multi-agent demonstration (mock mode)
python examples/multi_agent_demo.py --mock

# Run interactive demo
python examples/multi_agent_demo.py --mock --interactive
```

## Installed Packages

### Core HRM Dependencies
- PyTorch 2.7.1 (with M1/M2 Mac optimization)
- einops, tqdm, pydantic, wandb
- omegaconf, hydra-core
- huggingface_hub

### SWE-smith Dependencies
- tree-sitter 0.25.0 (all language parsers)
- unidiff, bashlex, pexpect
- datasets 4.0.0 (HuggingFace)
- docker, modal, openai, litellm
- rich, textual (UI components)

### Multi-Agent Runtime
- All dependencies for agent orchestration
- Mock fallbacks for SWE-ReX when not installed
- Async processing capabilities

## Usage Examples

### 1. SWE-smith Task Analysis

```python
from dataset.swe_smith_integration import SWESmithTaskRegistry, HRMTaskAnalyzer

# Initialize registry
registry = SWESmithTaskRegistry()

# Load sample tasks
tasks = await registry.load_dataset_sample(num_samples=100)
print(f"Loaded {len(tasks)} tasks")

# Analyze task complexity
analyzer = HRMTaskAnalyzer()
for task in tasks[:5]:
    analysis = analyzer.analyze_task(task)
    print(f"Task: {task['instance_id']}")
    print(f"Language: {analysis['language']}")
    print(f"Complexity: {analysis['complexity_score']}")
    print(f"Tools: {analysis['tools_required']}")
    print("---")
```

### 2. Multi-Agent Orchestration

```python
import asyncio
from agent_runtime.swe_rex_integration import HRMMultiAgentOrchestrator

async def run_multi_agent_demo():
    # Initialize orchestrator with Python 3.10 environment
    orchestrator = HRMMultiAgentOrchestrator()
    
    # Initialize and start
    await orchestrator.initialize()
    await orchestrator.start()
    
    # Submit a coding task
    task_id = orchestrator.submit_task({
        "problem_statement": "Implement a binary search algorithm",
        "language": "python",
        "complexity_score": 0.6
    })
    
    # Monitor system status
    status = orchestrator.get_system_status()
    print(f"Active agents: {status['active_agents']}")
    print(f"Pending tasks: {status['pending_tasks']}")
    
    # Cleanup
    await orchestrator.stop()

# Run the demo
asyncio.run(run_multi_agent_demo())
```

### 3. Training Data Generation

```python
from dataset.swe_smith_integration import HRMSWESmithDataset

# Create dataset for HRM training
dataset = HRMSWESmithDataset(
    num_samples=1000,
    complexity_range=(0.3, 0.9),
    languages=['python', 'javascript', 'rust']
)

# Generate training samples
training_data = []
for i in range(len(dataset)):
    sample = dataset[i]
    training_data.append(sample)
    
    if i < 3:  # Show first few samples
        print(f"Sample {i}:")
        print(f"  Problem: {sample['problem_statement'][:100]}...")
        print(f"  Language: {sample['language']}")
        print(f"  Complexity: {sample['complexity_score']}")
        print()

print(f"Generated {len(training_data)} training samples")
```

## Project Structure with Python 3.10

```
HRM-AstralDrift/
├── venv_py310/              # Python 3.10 virtual environment
│   ├── bin/activate         # Activation script
│   └── lib/python3.10/      # Installed packages
├── agent_runtime/           # Multi-agent system
│   ├── swe_rex_integration.py
│   └── example_config.yaml
├── dataset/                 # Data processing
│   └── swe_smith_integration.py
├── examples/               # Demonstration scripts
│   └── multi_agent_demo.py
├── SWE-smith/             # Full SWE-smith repository
├── SWE-ReX/              # SWE-ReX runtime (optional)
└── requirements.txt       # Core dependencies
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'swesmith'"**
   ```bash
   # Make sure you're in the Python 3.10 environment
   source venv_py310/bin/activate
   
   # Reinstall if needed
   cd SWE-smith && pip install -e .
   ```

2. **Tree-sitter compilation errors**
   ```bash
   # These should be resolved in Python 3.10, but if issues persist:
   pip install --upgrade tree-sitter tree-sitter-languages
   ```

3. **Multi-agent system hanging**
   ```bash
   # Run in mock mode for testing without SWE-ReX
   python examples/multi_agent_demo.py --mock
   ```

### Performance Optimization

1. **For M1/M2 Macs**: PyTorch is optimized for Apple Silicon
2. **Memory Usage**: The venv uses ~500MB, full system can use up to 2GB
3. **Concurrency**: Python 3.10's improved async support enables 30+ parallel agents

## Development Workflow

### 1. Activate Environment
```bash
source venv_py310/bin/activate
```

### 2. Install Additional Dependencies
```bash
pip install <package-name>
```

### 3. Update Requirements (if needed)
```bash
pip freeze > requirements_py310.txt
```

### 4. Run Tests
```bash
# Basic functionality
python simple_test.py

# Multi-agent demo
python examples/multi_agent_demo.py --mock

# SWE-smith integration
python -m pytest tests/ -v  # If test suite exists
```

## Next Steps

With Python 3.10+ environment ready, you can now:

1. **Scale Training Data**: Use SWE-smith's 52K+ task instances
2. **Deploy Multi-Agent System**: Run 30+ specialized agents in parallel
3. **Integrate SWE-ReX**: Add real sandboxed execution environments
4. **Benchmark Performance**: Test against LiveCodeBench, Polyglot, SWE-bench
5. **Develop Custom Agents**: Create specialized agents for specific domains

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the agent configuration in `agent_runtime/example_config.yaml`
3. Test individual components before running the full system
4. Use mock mode when SWE-ReX is not available

The Python 3.10+ environment provides a solid foundation for the HRM multi-agent architecture with full SWE-smith compatibility!