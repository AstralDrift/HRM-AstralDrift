# SWE-smith Setup Notes

## Repository Status
- ✅ SWE-smith repository cloned successfully
- ✅ SWE-ReX repository cloned successfully
- ⚠️ Python version requirement: 3.10+ (currently using 3.9.7)

## Key Findings

### SWE-smith Capabilities
- **52K+ Task Instances**: Unlimited software engineering training data
- **Repository Conversion**: Turn any GitHub repo into "SWE-gym" environment
- **Docker-Based Execution**: Consistent training and evaluation environments
- **26K SWE-agent Trajectories**: Rich RL training data available
- **Proven Success**: SWE-agent-LM-32B achieved 40.2% pass@1 on SWE-bench Verified

### Directory Structure
```
SWE-smith/
├── swesmith/                  # Main package
│   ├── bug_gen/              # Bug generation modules
│   ├── build_repo/           # Repository setup
│   ├── harness/              # Evaluation harness
│   ├── issue_gen/            # Issue generation
│   ├── profiles/             # Repository profiles
│   └── train/                # Training utilities
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
└── tests/                    # Test suite
```

### Dependencies
- Python 3.10+ required
- Docker for execution environments
- Tree-sitter for code parsing
- HuggingFace datasets for data loading
- Multiple programming language parsers (C, C#, Go, Java, JS, Python, Rust)

### Integration Points for HRM
1. **Task Registry**: `swesmith.profiles.registry.get_from_inst()`
2. **Container Management**: Docker-based execution environments
3. **Dataset Loading**: `datasets.load_dataset("SWE-bench/SWE-smith")`
4. **Trajectory Collection**: 26K SWE-agent trajectories for RL training

## Next Steps
1. **Environment Upgrade**: Set up Python 3.10+ environment for full compatibility
2. **Docker Setup**: Ensure Docker is available for container-based task execution
3. **API Integration**: Implement HRM-specific task processing pipeline
4. **Multi-Agent Architecture**: Design SWE-ReX integration for parallel execution

## Compatibility Notes
- Current Python 3.9.7 causes dependency conflicts
- Tree-sitter version requirements are strict
- Docker is essential for full functionality
- Ubuntu 22.04 is the recommended development environment (macOS may have limitations)

## Resources
- [SWE-smith Dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith)
- [SWE-agent-LM-32B Model](https://huggingface.co/SWE-bench/SWE-agent-LM-32B)
- [250+ Docker Environments](https://github.com/SWE-bench/SWE-smith-envs)
- [Documentation](https://swesmith.com/)