# LiveCodeBench Dataset Analysis

## Dataset Overview

**LiveCodeBench** is a holistic and contamination-free evaluation benchmark for Large Language Models (LLMs) focused on code-related capabilities. It continuously collects coding problems from competitive programming platforms to provide temporal evaluation that prevents data contamination.

### Key Characteristics
- **Continuous Updates**: Live updating benchmark that collects new problems over time
- **Contamination-Free**: Problems annotated with release dates allow evaluation on post-training data
- **Multi-Platform**: Sources problems from LeetCode, AtCoder, and CodeForces
- **Holistic Evaluation**: Tests 4 different code-related capabilities beyond just generation

## Dataset Versions and Sizes

| Version | Time Period | Problem Count | Description |
|---------|-------------|---------------|-------------|
| release_v1 | May 2023 - Mar 2024 | 400 problems | Initial release |
| release_v2 | May 2023 - May 2024 | 511 problems | Updated release |
| release_v3 | May 2023 - Jul 2024 | 612 problems | Extended collection |
| release_v4 | May 2023 - Sep 2024 | 713 problems | Further expansion |
| release_v5 | May 2023 - Jan 2025 | 880 problems | Latest version |

**Recommended Version**: `release_v2` (511 problems) - Good balance of size and quality
**Lite Version Available**: `code_generation_lite` with pruned test cases for faster processing

## Dataset Structure

### HuggingFace Hub Locations
- **Main Dataset**: `livecodebench/code_generation`
- **Lite Dataset**: `livecodebench/code_generation_lite` (recommended)
- **Additional Tasks**: `livecodebench/test_generation`

### Data Format
- **Format**: JSONL (JSON Lines)
- **Split**: Test split only (evaluation benchmark)
- **Files**: Multiple JSONL files (test.jsonl, test2.jsonl, etc.)

### Problem Schema
Based on analysis, each problem contains:

```json
{
  "question_title": "string",
  "question_content": "string", 
  "platform": "atcoder|leetcode|codeforces",
  "question_id": "string",
  "contest_id": "string", 
  "contest_date": "YYYY-MM-DD",
  "starter_code": "string",
  "difficulty": "easy|medium|hard",
  "public_test_cases": [{"input": "...", "output": "..."}],
  "private_test_cases": [{"input": "...", "output": "..."}],
  "metadata": "object"
}
```

### Statistics
- **Average Test Cases**: 59+ hidden test cases per problem
- **Difficulty Distribution**: Mix of easy, medium, and hard problems
- **Platform Distribution**: Problems from all three competitive programming sites

## Evaluation Scenarios

LiveCodeBench supports 4 evaluation scenarios:

### 1. Code Generation
- **Task**: Generate code from natural language description
- **Input**: Problem description, examples, constraints
- **Output**: Complete working program
- **Metrics**: pass@1, pass@5

### 2. Self-Repair
- **Task**: Fix incorrect code given execution feedback  
- **Input**: Problem description, buggy code, test failure, execution feedback
- **Output**: Corrected program
- **Metrics**: Repair success rate

### 3. Test Output Prediction
- **Task**: Predict program output for given input
- **Input**: Problem description, program code, test input
- **Output**: Expected output
- **Metrics**: Output accuracy

### 4. Code Execution
- **Task**: "Execute" program mentally and predict output
- **Input**: Program code, input values
- **Output**: Execution result
- **Metrics**: Execution accuracy

## Programming Languages

While not explicitly documented, based on competitive programming platforms:

**Likely Supported Languages:**
- Python (most common)
- C++
- Java
- JavaScript
- C#
- Go
- Rust

**Primary Focus**: Python (based on competitive programming trends)

## Dataset Quality Features

### Problem Sources
- **LeetCode**: Algorithm and data structure problems
- **AtCoder**: Japanese competitive programming contests
- **CodeForces**: Russian competitive programming platform

### Quality Assurance
- Problems validated by platform users
- High-quality test cases with edge cases
- Difficulty ratings from contest platforms
- Temporal annotation prevents contamination

## Technical Specifications

### File Structure
```
livecodebench/code_generation_lite/
├── README.md
├── code_generation_lite.py  # Dataset loading script
├── test.jsonl              # V1 problems
├── test2.jsonl             # V2 problems  
├── test3.jsonl             # V3 problems
├── test4.jsonl             # V4 problems
├── test5.jsonl             # V5 problems
└── test6.jsonl             # V6 problems
```

### Loading Code
```python
from datasets import load_dataset

# Load specific version
dataset = load_dataset("livecodebench/code_generation_lite", 
                      version_tag="release_v2")

# Access test split
test_data = dataset["test"]
print(f"Number of problems: {len(test_data)}")
```

### Storage Requirements
- **Lite Version**: ~5.4GB total storage
- **Main Version**: Larger due to more test cases
- **Per Problem**: Variable size depending on test cases

## Integration Recommendations for HRM

### Dataset Preparation
1. **Use Lite Version**: Faster training/evaluation while maintaining quality
2. **Version Selection**: Start with `release_v2` (511 problems)
3. **Split Strategy**: Use temporal splits for contamination-free evaluation
4. **Language Focus**: Begin with Python problems for initial implementation

### Training Adaptations
1. **Multi-Task Training**: Train on all 4 scenarios simultaneously
2. **Temporal Awareness**: Include contest dates in problem embeddings
3. **Difficulty Encoding**: Embed difficulty levels for adaptive computation
4. **Platform-Specific**: Handle different coding styles from each platform

### Evaluation Strategy  
1. **Pass@K Metrics**: Implement pass@1 and pass@5 for code generation
2. **Multi-Scenario**: Evaluate across all 4 task types
3. **Temporal Evaluation**: Test on problems released after training cutoff
4. **Platform Breakdown**: Analyze performance by source platform

### Technical Considerations
1. **Memory Usage**: Large test case sets require efficient handling
2. **Execution Environment**: Need code execution capability for evaluation
3. **Language Support**: Plan for multi-language code generation
4. **Error Handling**: Robust parsing of different code formats

## Next Steps for HRM Integration

1. **Dataset Download**: Fetch `livecodebench/code_generation_lite` v2
2. **Schema Analysis**: Examine actual problem structures
3. **Preprocessing Pipeline**: Convert to HRM-compatible format
4. **Baseline Implementation**: Start with code generation scenario
5. **Evaluation Framework**: Implement pass@k and execution testing
6. **Multi-Task Extension**: Expand to all 4 evaluation scenarios

## Benchmarking Context

**Current SOTA Performance:**
- Various closed-source models (GPT-4, Claude) achieve high performance
- Open-source models show significant performance gaps  
- **Target for HRM**: Achieve competitive performance with <100M parameters

**Key Success Metrics:**
- **Code Generation**: >30% pass@1 rate
- **Self-Repair**: >25% repair success  
- **Execution Tasks**: >70% accuracy
- **Overall**: Top-tier performance with efficient parameter usage

## References

- **Paper**: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" (arXiv:2403.07974)
- **Homepage**: https://livecodebench.github.io/
- **GitHub**: https://github.com/LiveCodeBench/LiveCodeBench
- **HuggingFace**: https://huggingface.co/datasets/livecodebench/code_generation_lite
- **Leaderboard**: Available on HuggingFace Blog