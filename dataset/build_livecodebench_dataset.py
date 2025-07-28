"""
LiveCodeBench Dataset Builder for HRM Training

This module creates HRM-compatible training datasets from LiveCodeBench problems.
It supports all 4 scenarios: code generation, self-repair, test prediction, and execution.
The datasets are temporally filtered to prevent contamination and include data augmentation.

Usage:
    python dataset/build_livecodebench_dataset.py \
        --output-dir data/livecodebench-hrm \
        --scenarios generation,selfrepair \
        --subsample-size 1000 \
        --start-date 2023-01-01 \
        --end-date 2024-06-01 \
        --num-aug 500
"""

from typing import List, Dict, Optional, Tuple, Union
import os
import json
import random
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import argparse
from pydantic import BaseModel

# LiveCodeBench imports
import sys
sys.path.append('/Users/micahoates/Developer/x/HRM-AstralDrift/LiveCodeBench')
from lcb_runner.benchmarks import (
    CodeGenerationProblem, 
    TestOutputPredictionProblem,
    CodeExecutionProblem,
    load_code_generation_dataset,
    load_test_prediction_dataset,
    load_code_execution_dataset
)
from lcb_runner.utils.scenarios import Scenario

# Simple metadata structure for LiveCodeBench
@dataclass
class LiveCodeBenchMetadata:
    puzzle_name: str
    num_examples: int
    num_augmentations: int
    vocab_size: int
    max_seq_len: int
    scenarios: str
    creation_date: str
    source_version: str
    temporal_filter: str

from models.code_generation.input_processor import (
    CodeGenerationInput, 
    CodeGenerationTask, 
    ProgrammingLanguage,
    CodeGenerationInputProcessor
)

# ArgumentParser will be created in main function

class LiveCodeBenchConfig(BaseModel):
    """Configuration for LiveCodeBench dataset building"""
    # Output configuration
    output_dir: str = "data/livecodebench-hrm"
    
    # Dataset filtering
    scenarios: str = "generation,selfrepair,testprediction,execution"  # Comma-separated
    start_date: Optional[str] = None  # Format: YYYY-MM-DD
    end_date: Optional[str] = None    # Format: YYYY-MM-DD
    subsample_size: Optional[int] = None
    
    # Data augmentation
    num_aug: int = 100  # Number of augmented examples per original
    
    # Contamination prevention
    release_version: str = "release_v1"
    
    # Training configuration
    max_seq_len: int = 2048
    vocab_size: int = 40000
    
    # Scenario weights for multi-task training
    scenario_weights: str = "1.0,0.8,0.6,0.5"  # generation,repair,test,exec

@dataclass
class LiveCodeBenchExample:
    """Unified format for all LiveCodeBench scenarios"""
    problem_id: str
    scenario: Scenario
    input_text: str
    target_output: str
    language: ProgrammingLanguage
    task_type: CodeGenerationTask
    metadata: Dict
    contest_date: datetime
    difficulty: str
    platform: str

class LiveCodeBenchProcessor:
    """Processes LiveCodeBench problems into HRM training format"""
    
    def __init__(self, config: LiveCodeBenchConfig):
        self.config = config
        self.input_processor = CodeGenerationInputProcessor(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len
        )
        
        # Parse scenarios
        self.scenarios = [
            Scenario(s.strip()) for s in config.scenarios.split(',')
        ]
        
        # Parse scenario weights
        weights = [float(w.strip()) for w in config.scenario_weights.split(',')]
        self.scenario_weights = dict(zip([
            Scenario.codegeneration,
            Scenario.selfrepair, 
            Scenario.testoutputprediction,
            Scenario.codeexecution
        ], weights))
        
        print(f"Initialized processor for scenarios: {self.scenarios}")
        print(f"Scenario weights: {self.scenario_weights}")
    
    def load_datasets(self) -> List[LiveCodeBenchExample]:
        """Load all LiveCodeBench datasets based on configuration"""
        examples = []
        
        for scenario in self.scenarios:
            if scenario == Scenario.codegeneration:
                examples.extend(self._load_generation_data())
            elif scenario == Scenario.selfrepair:
                examples.extend(self._load_repair_data())
            elif scenario == Scenario.testoutputprediction:
                examples.extend(self._load_test_prediction_data())
            elif scenario == Scenario.codeexecution:
                examples.extend(self._load_execution_data())
        
        # Apply temporal filtering
        examples = self._apply_temporal_filter(examples)
        
        # Apply subsampling if specified
        if self.config.subsample_size and len(examples) > self.config.subsample_size:
            random.shuffle(examples)
            examples = examples[:self.config.subsample_size]
        
        print(f"Loaded {len(examples)} examples after filtering and subsampling")
        return examples
    
    def _load_generation_data(self) -> List[LiveCodeBenchExample]:
        """Load code generation problems"""
        print("Loading code generation dataset...")
        problems = load_code_generation_dataset(
            release_version=self.config.release_version,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        examples = []
        for problem in tqdm(problems, desc="Processing generation problems"):
            # Create problem description with test cases
            input_text = self._format_generation_input(problem)
            target_output = self._format_generation_target(problem)
            
            example = LiveCodeBenchExample(
                problem_id=f"gen_{problem.question_id}",
                scenario=Scenario.codegeneration,
                input_text=input_text,
                target_output=target_output,
                language=ProgrammingLanguage.PYTHON,  # LiveCodeBench is Python-focused
                task_type=CodeGenerationTask.GENERATION,
                metadata={
                    "platform": problem.platform.value,
                    "has_starter_code": bool(problem.starter_code),
                    "num_public_tests": len(problem.public_test_cases),
                    "num_private_tests": len(problem.private_test_cases),
                    "function_name": problem.metadata.get("func_name", ""),
                },
                contest_date=problem.contest_date,
                difficulty=problem.difficulty.value,
                platform=problem.platform.value
            )
            examples.append(example)
        
        return examples
    
    def _load_repair_data(self) -> List[LiveCodeBenchExample]:
        """Load self-repair data by creating broken versions of generation problems"""
        print("Creating self-repair dataset from generation problems...")
        generation_problems = load_code_generation_dataset(
            release_version=self.config.release_version,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        examples = []
        for problem in tqdm(generation_problems, desc="Creating repair problems"):
            # Generate multiple broken versions of each problem
            for i in range(min(3, max(1, self.config.num_aug // len(generation_problems)))):
                broken_code, error_msg = self._create_broken_code(problem)
                
                input_text = self._format_repair_input(problem, broken_code, error_msg)
                target_output = self._format_generation_target(problem)
                
                example = LiveCodeBenchExample(
                    problem_id=f"repair_{problem.question_id}_{i}",
                    scenario=Scenario.selfrepair,
                    input_text=input_text,
                    target_output=target_output,
                    language=ProgrammingLanguage.PYTHON,
                    task_type=CodeGenerationTask.REPAIR,
                    metadata={
                        "platform": problem.platform.value,
                        "error_type": error_msg.split(":")[0] if ":" in error_msg else "SyntaxError",
                        "original_problem_id": problem.question_id,
                    },
                    contest_date=problem.contest_date,
                    difficulty=problem.difficulty.value,
                    platform=problem.platform.value
                )
                examples.append(example)
        
        return examples
    
    def _load_test_prediction_data(self) -> List[LiveCodeBenchExample]:
        """Load test output prediction problems"""
        print("Loading test prediction dataset...")
        problems = load_test_prediction_dataset(release_version=self.config.release_version)
        
        examples = []
        for problem in tqdm(problems, desc="Processing test prediction problems"):
            input_text = self._format_test_prediction_input(problem)
            target_output = problem.test[0].output  # Expected output
            
            example = LiveCodeBenchExample(
                problem_id=f"test_{problem.question_id}_{problem.test_id}",
                scenario=Scenario.testoutputprediction,
                input_text=input_text,
                target_output=target_output,
                language=ProgrammingLanguage.PYTHON,
                task_type=CodeGenerationTask.TEST_PREDICTION,
                metadata={
                    "function_name": problem.function_name,
                    "test_id": problem.test_id,
                    "test_input": problem.test[0].input,
                },
                contest_date=problem.contest_date,
                difficulty=problem.difficulty,
                platform="leetcode"  # Test prediction is typically from LeetCode
            )
            examples.append(example)
        
        return examples
    
    def _load_execution_data(self) -> List[LiveCodeBenchExample]:
        """Load code execution problems"""
        print("Loading code execution dataset...")
        problems = load_code_execution_dataset(release_version=self.config.release_version)
        
        examples = []
        for problem in tqdm(problems, desc="Processing execution problems"):
            input_text = self._format_execution_input(problem)
            target_output = problem.output
            
            example = LiveCodeBenchExample(
                problem_id=f"exec_{problem.id}",
                scenario=Scenario.codeexecution,
                input_text=input_text,
                target_output=target_output,
                language=ProgrammingLanguage.PYTHON,
                task_type=CodeGenerationTask.GENERATION,  # Execution is still generation-like
                metadata={
                    "function_name": problem.function_name,
                    "num_steps": problem.numsteps,
                    "problem_id": problem.problem_id,
                },
                contest_date=problem.contest_date,
                difficulty=problem.difficulty,
                platform="mixed"
            )
            examples.append(example)
        
        return examples
    
    def _apply_temporal_filter(self, examples: List[LiveCodeBenchExample]) -> List[LiveCodeBenchExample]:
        """Apply temporal filtering to prevent contamination"""
        if not self.config.start_date and not self.config.end_date:
            return examples
        
        filtered = []
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d") if self.config.start_date else None
        end_dt = datetime.strptime(self.config.end_date, "%Y-%m-%d") if self.config.end_date else None
        
        for example in examples:
            include = True
            if start_dt and example.contest_date < start_dt:
                include = False
            if end_dt and example.contest_date > end_dt:
                include = False
            
            if include:
                filtered.append(example)
        
        print(f"Temporal filtering: {len(examples)} -> {len(filtered)} examples")
        return filtered
    
    def _format_generation_input(self, problem: CodeGenerationProblem) -> str:
        """Format code generation input"""
        parts = [f"Problem: {problem.question_content}"]
        
        if problem.starter_code:
            parts.append(f"Starter Code:\n{problem.starter_code}")
        
        if problem.public_test_cases:
            test_examples = []
            for i, test in enumerate(problem.public_test_cases[:3]):  # Limit to 3 examples
                test_examples.append(f"Example {i+1}:\nInput: {test.input}\nOutput: {test.output}")
            parts.append("Test Cases:\n" + "\n\n".join(test_examples))
        
        return "\n\n".join(parts)
    
    def _format_generation_target(self, problem: CodeGenerationProblem) -> str:
        """Format expected code output - this would need actual solutions"""
        # For now, use a placeholder that includes the starter code structure
        if problem.starter_code:
            return f"# Complete the following function\n{problem.starter_code}\n# TODO: Implement solution"
        else:
            func_name = problem.metadata.get("func_name", "solution")
            return f"def {func_name}():\n    # TODO: Implement solution\n    pass"
    
    def _format_repair_input(self, problem: CodeGenerationProblem, broken_code: str, error_msg: str) -> str:
        """Format self-repair input"""
        parts = [
            f"Problem: {problem.question_content}",
            f"Broken Code:\n{broken_code}",
            f"Error Message: {error_msg}",
            "Fix the code to solve the problem correctly."
        ]
        return "\n\n".join(parts)
    
    def _format_test_prediction_input(self, problem: TestOutputPredictionProblem) -> str:
        """Format test prediction input"""
        parts = [
            f"Problem: {problem.question_content}",
            f"Starter Code:\n{problem.starter_code}",
            f"Function: {problem.function_name}",
            f"Test Input: {problem.test[0].input}",
            "Predict the output:"
        ]
        return "\n\n".join(parts)
    
    def _format_execution_input(self, problem: CodeExecutionProblem) -> str:
        """Format code execution input"""
        parts = [
            f"Code:\n{problem.code}",
            f"Input: {problem.input}",
            "Execute the code with the given input and provide the output:"
        ]
        return "\n\n".join(parts)
    
    def _create_broken_code(self, problem: CodeGenerationProblem) -> Tuple[str, str]:
        """Create broken code for self-repair training"""
        if not problem.starter_code:
            # Create a simple broken function
            func_name = problem.metadata.get("func_name", "solution")
            broken_variants = [
                (f"def {func_name}(:\n    pass", "SyntaxError: invalid syntax"),
                (f"def {func_name}():\n    return", "SyntaxError: invalid syntax"),
                (f"def {func_name}():\n    x = 1 +\n    return x", "SyntaxError: invalid syntax"),
                (f"def {func_name}():\n    if True\n        return 1", "SyntaxError: invalid syntax"),
            ]
            return random.choice(broken_variants)
        
        # Introduce realistic errors in starter code
        code = problem.starter_code
        error_types = [
            # Missing parenthesis
            (lambda c: c.replace("(", "", 1), "SyntaxError: invalid syntax"),
            # Missing colon
            (lambda c: c.replace(":", "", 1), "SyntaxError: invalid syntax"),
            # Wrong indentation
            (lambda c: c.replace("    ", "  ", 1), "IndentationError: unindent does not match any outer indentation level"),
            # Undefined variable
            (lambda c: c + "\n    return undefined_var", "NameError: name 'undefined_var' is not defined"),
        ]
        
        transformer, error_msg = random.choice(error_types)
        try:
            broken_code = transformer(code)
            return broken_code, error_msg
        except:
            # Fallback to simple syntax error
            return code.replace(":", "", 1), "SyntaxError: invalid syntax"
    
    def augment_examples(self, examples: List[LiveCodeBenchExample]) -> List[LiveCodeBenchExample]:
        """Apply data augmentation to increase dataset size"""
        if self.config.num_aug == 0:
            return examples
        
        print(f"Applying data augmentation: {self.config.num_aug} augmentations per example")
        augmented = list(examples)  # Start with original examples
        
        for example in tqdm(examples, desc="Augmenting examples"):
            for aug_idx in range(self.config.num_aug):
                aug_example = self._augment_single_example(example, aug_idx)
                augmented.append(aug_example)
        
        print(f"Augmentation complete: {len(examples)} -> {len(augmented)} examples")
        return augmented
    
    def _augment_single_example(self, example: LiveCodeBenchExample, aug_idx: int) -> LiveCodeBenchExample:
        """Create an augmented version of a single example"""
        aug_input = example.input_text
        aug_output = example.target_output
        
        # Different augmentation strategies based on scenario
        if example.scenario == Scenario.codegeneration:
            aug_input, aug_output = self._augment_generation_example(example)
        elif example.scenario == Scenario.selfrepair:
            aug_input, aug_output = self._augment_repair_example(example)
        elif example.scenario == Scenario.testoutputprediction:
            aug_input = self._augment_test_prediction_example(example)
        
        # Create augmented example with new ID
        aug_example = LiveCodeBenchExample(
            problem_id=f"{example.problem_id}_aug_{aug_idx}",
            scenario=example.scenario,
            input_text=aug_input,
            target_output=aug_output,
            language=example.language,
            task_type=example.task_type,
            metadata={**example.metadata, "is_augmented": True, "aug_idx": aug_idx},
            contest_date=example.contest_date,
            difficulty=example.difficulty,
            platform=example.platform
        )
        
        return aug_example
    
    def _augment_generation_example(self, example: LiveCodeBenchExample) -> Tuple[str, str]:
        """Augment code generation examples"""
        # Simple text variations for problem descriptions
        augmentations = [
            lambda text: text.replace("Write a function", "Implement a function"),
            lambda text: text.replace("that", "which"),
            lambda text: text.replace("returns", "gives back"),
            lambda text: text.replace("array", "list"),
            lambda text: text.replace("string", "text"),
        ]
        
        aug_input = example.input_text
        for aug in random.sample(augmentations, min(2, len(augmentations))):
            try:
                aug_input = aug(aug_input)
            except:
                pass
        
        return aug_input, example.target_output
    
    def _augment_repair_example(self, example: LiveCodeBenchExample) -> Tuple[str, str]:
        """Augment self-repair examples by creating different types of errors"""
        # For repair examples, we can create different broken versions
        return example.input_text, example.target_output
    
    def _augment_test_prediction_example(self, example: LiveCodeBenchExample) -> str:
        """Augment test prediction examples"""
        # Minor variations in instruction phrasing
        return example.input_text.replace("Predict the output:", "What is the expected output?")
    
    def convert_to_hrm_format(self, examples: List[LiveCodeBenchExample]) -> Dict:
        """Convert examples to HRM training format"""
        print("Converting examples to HRM format...")
        
        # Group examples by scenario for balanced sampling
        scenario_groups = {}
        for example in examples:
            if example.scenario not in scenario_groups:
                scenario_groups[example.scenario] = []
            scenario_groups[example.scenario].append(example)
        
        # Print scenario distribution
        for scenario, group in scenario_groups.items():
            print(f"  {scenario.value}: {len(group)} examples")
        
        # Convert to HRM input format
        hrm_inputs = []
        hrm_labels = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]
        
        example_id = 0
        for scenario in self.scenarios:
            if scenario not in scenario_groups:
                continue
                
            for example in tqdm(scenario_groups[scenario], desc=f"Converting {scenario.value}"):
                # Create CodeGenerationInput
                code_input = CodeGenerationInput(
                    problem_description=example.input_text,
                    language=example.language,
                    task_type=example.task_type,
                    existing_code=None,  # Extracted from input_text if needed
                    test_cases=None,     # Extracted from input_text if needed
                )
                
                # Process through HRM input processor
                processed = self.input_processor.process_input(code_input)
                
                # Store HRM format data
                hrm_inputs.append(processed.input_tokens.numpy())
                hrm_labels.append(self._tokenize_target(example.target_output))
                puzzle_identifiers.append(processed.puzzle_identifier.item())
                
                example_id += 1
                if example_id < len(examples):
                    puzzle_indices.append(example_id)
                    group_indices.append(example_id)
        
        return {
            "inputs": np.array(hrm_inputs),
            "labels": np.array(hrm_labels),
            "puzzle_identifiers": np.array(puzzle_identifiers),
            "puzzle_indices": np.array(puzzle_indices),
            "group_indices": np.array(group_indices),
        }
    
    def _tokenize_target(self, target_text: str) -> np.ndarray:
        """Tokenize target output for training"""
        tokens = self.input_processor.tokenizer.base_tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len
        )
        return tokens["input_ids"].squeeze(0).numpy()

def save_dataset(data: Dict, metadata: LiveCodeBenchMetadata, output_dir: str):
    """Save processed dataset in HRM format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data arrays
    for split in ["train", "test"]:  # LiveCodeBench typically uses test split
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(os.path.join(split_dir, f"{key}.npy"), value)
            else:
                with open(os.path.join(split_dir, f"{key}.json"), "w") as f:
                    json.dump(value, f, indent=2)
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    
    print(f"Dataset saved to {output_dir}")

def main():
    """Main function to build LiveCodeBench dataset for HRM training"""
    parser = argparse.ArgumentParser(description="Build LiveCodeBench dataset for HRM training")
    parser.add_argument("--output-dir", type=str, default="data/livecodebench-hrm", help="Output directory")
    parser.add_argument("--scenarios", type=str, default="generation,selfrepair,testprediction,execution", help="Scenarios to include")
    parser.add_argument("--start-date", type=str, help="Start date for temporal filtering (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for temporal filtering (YYYY-MM-DD)")
    parser.add_argument("--subsample-size", type=int, help="Subsample size")
    parser.add_argument("--num-aug", type=int, default=100, help="Number of augmentations")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--vocab-size", type=int, default=40000, help="Vocabulary size")
    
    args = parser.parse_args()
    
    # Create config from args
    config = LiveCodeBenchConfig(
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        start_date=args.start_date,
        end_date=args.end_date,
        subsample_size=args.subsample_size,
        num_aug=args.num_aug,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size
    )
    
    print("Building LiveCodeBench dataset for HRM training...")
    print(f"Configuration: {config}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize processor
    processor = LiveCodeBenchProcessor(config)
    
    # Load and process examples
    examples = processor.load_datasets()
    
    # Apply augmentation
    examples = processor.augment_examples(examples)
    
    # Convert to HRM format
    hrm_data = processor.convert_to_hrm_format(examples)
    
    # Create metadata
    metadata = LiveCodeBenchMetadata(
        puzzle_name="LiveCodeBench",
        num_examples=len(examples),
        num_augmentations=config.num_aug,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        scenarios=config.scenarios,
        creation_date=datetime.now().isoformat(),
        source_version=config.release_version,
        temporal_filter=f"{config.start_date} to {config.end_date}",
    )
    
    # Save dataset
    save_dataset(hrm_data, metadata, config.output_dir)
    
    print("LiveCodeBench dataset building completed!")
    print(f"Final dataset size: {len(examples)} examples")
    print(f"Scenarios included: {config.scenarios}")
    print(f"Output directory: {config.output_dir}")

if __name__ == "__main__":
    main()