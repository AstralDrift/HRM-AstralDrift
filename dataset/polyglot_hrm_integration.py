"""
Polyglot Benchmark Integration with HRM Code Generation Architecture

This module integrates the Polyglot benchmark with HRM's existing code generation
architecture, creating a unified training pipeline that leverages multi-language
data for enhanced code generation performance.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Import our Polyglot modules
from .polyglot_benchmark_extractor import PolyglotBenchmarkExtractor, PolyglotProblem
from .diff_based_training_generator import DiffBasedTrainingGenerator, DiffTrainingExample
from .cross_language_mapper import CrossLanguageProblemMapper, TransferLearningExample

# Import HRM code generation modules
from models.code_generation.input_processor import (
    CodeGenerationInputProcessor, CodeGenerationInput, CodeGenerationTask, 
    ProgrammingLanguage, ProcessedCodeInput
)
from models.code_generation.output_generator import CodeOutputGenerator
from models.code_generation.hrm_code_model import HRMCodeGenerationModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolyglotTrainingConfig:
    """Configuration for Polyglot training integration"""
    # Data parameters
    max_problems_per_language: int = 50
    max_diff_examples_per_problem: int = 10
    max_transfer_examples_per_pair: int = 5
    
    # Training parameters
    batch_size: int = 16
    max_sequence_length: int = 2048
    curriculum_learning: bool = True
    cross_language_mixing_ratio: float = 0.3
    
    # Quality filters
    min_complexity_score: float = 0.1
    max_complexity_score: float = 0.8
    min_diff_quality_score: float = 0.5
    
    # Language sampling weights
    language_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.language_weights is None:
            # Default equal weights for all languages
            self.language_weights = {
                "python": 1.0,
                "javascript": 1.0, 
                "java": 1.0,
                "cpp": 1.0,
                "go": 1.0,
                "rust": 1.0
            }

class PolyglotCodeDataset(Dataset):
    """
    PyTorch Dataset for Polyglot benchmark training data
    Integrates with HRM's existing input processing pipeline
    """
    
    def __init__(self, 
                 polyglot_data: Dict,
                 input_processor: CodeGenerationInputProcessor,
                 config: PolyglotTrainingConfig):
        """
        Initialize dataset
        
        Args:
            polyglot_data: Dictionary with problems, diff_examples, transfer_examples
            input_processor: HRM input processor
            config: Training configuration
        """
        self.input_processor = input_processor
        self.config = config
        
        # Extract data components
        self.problems = polyglot_data.get("problems", {})
        self.diff_examples = polyglot_data.get("diff_examples", [])
        self.transfer_examples = polyglot_data.get("transfer_examples", [])
        
        # Create training samples
        self.samples = self._create_training_samples()
        
        logger.info(f"Created PolyglotCodeDataset with {len(self.samples)} training samples")
    
    def _create_training_samples(self) -> List[Dict]:
        """Create unified training samples from all data sources"""
        samples = []
        
        # 1. Code generation samples from problems
        samples.extend(self._create_generation_samples())
        
        # 2. Diff editing samples
        samples.extend(self._create_diff_samples())
        
        # 3. Cross-language transfer samples
        samples.extend(self._create_transfer_samples())
        
        # Apply quality filters
        samples = self._apply_quality_filters(samples)
        
        # Shuffle and apply curriculum learning
        if self.config.curriculum_learning:
            samples = self._apply_curriculum_ordering(samples)
        else:
            random.shuffle(samples)
        
        return samples
    
    def _create_generation_samples(self) -> List[Dict]:
        """Create code generation samples from problems"""
        samples = []
        
        for problem_id, problem in self.problems.items():
            # Skip if complexity is outside desired range
            if not (self.config.min_complexity_score <= 
                   problem.complexity_score <= 
                   self.config.max_complexity_score):
                continue
            
            for language, files in problem.files_by_language.items():
                # Find implementation and test files
                impl_files = [f for f in files if f.file_type == "implementation"]
                test_files = [f for f in files if f.file_type == "test"]
                
                if not impl_files:
                    continue
                
                # Create generation task
                test_cases = []
                if test_files:
                    # Extract test cases (simplified)
                    test_content = test_files[0].content
                    test_cases = self._extract_test_cases_from_content(test_content, language)
                
                # Map language string to enum
                lang_enum = getattr(ProgrammingLanguage, language.upper(), None)
                if lang_enum is None:
                    continue
                
                sample = {
                    "type": "generation",
                    "problem_id": problem_id,
                    "language": language,
                    "input": CodeGenerationInput(
                        problem_description=problem.description,
                        language=lang_enum,
                        task_type=CodeGenerationTask.GENERATION,
                        test_cases=test_cases
                    ),
                    "target_code": impl_files[0].content,
                    "complexity": problem.complexity_score,
                    "metadata": {
                        "problem_name": problem.problem_name,
                        "file_count": len(files)
                    }
                }
                samples.append(sample)
        
        logger.info(f"Created {len(samples)} code generation samples")
        return samples
    
    def _create_diff_samples(self) -> List[Dict]:
        """Create diff editing samples"""
        samples = []
        
        for diff_example in self.diff_examples:
            # Apply quality filter
            if hasattr(diff_example, 'quality_score'):
                if diff_example.quality_score < self.config.min_diff_quality_score:
                    continue
            
            # Map language string to enum
            lang_enum = getattr(ProgrammingLanguage, diff_example.language.upper(), None)
            if lang_enum is None:
                continue
            
            sample = {
                "type": "diff_edit", 
                "problem_id": getattr(diff_example, 'problem_id', 'unknown'),
                "language": diff_example.language,
                "input": CodeGenerationInput(
                    problem_description=diff_example.problem_description,
                    language=lang_enum,
                    task_type=CodeGenerationTask.DIFF_EDIT,
                    existing_code=diff_example.original_code,
                    test_cases=diff_example.test_cases
                ),
                "target_code": diff_example.target_code,
                "complexity": diff_example.complexity_score,
                "metadata": {
                    "edit_instruction": diff_example.edit_instruction,
                    "diff_operations": len(diff_example.diff_operations)
                }
            }
            samples.append(sample)
        
        logger.info(f"Created {len(samples)} diff editing samples")
        return samples
    
    def _create_transfer_samples(self) -> List[Dict]:
        """Create cross-language transfer samples"""
        samples = []
        
        for transfer_example in self.transfer_examples:
            # Create source->target transfer sample
            source_lang = getattr(ProgrammingLanguage, transfer_example.source_language.upper(), None)
            target_lang = getattr(ProgrammingLanguage, transfer_example.target_language.upper(), None)
            
            if source_lang is None or target_lang is None:
                continue
            
            sample = {
                "type": "transfer",
                "problem_id": transfer_example.problem_id,
                "language": transfer_example.target_language,
                "source_language": transfer_example.source_language,
                "input": CodeGenerationInput(
                    problem_description=f"Convert this {transfer_example.source_language} code to {transfer_example.target_language}: {transfer_example.transfer_instructions}",
                    language=target_lang,
                    task_type=CodeGenerationTask.GENERATION,
                    existing_code=transfer_example.source_code,
                    context_files={
                        f"source.{self._get_file_extension(transfer_example.source_language)}": transfer_example.source_code
                    }
                ),
                "target_code": transfer_example.target_code,
                "complexity": transfer_example.difficulty_score,
                "metadata": {
                    "transfer_instruction": transfer_example.transfer_instructions,
                    "shared_concepts": transfer_example.shared_concepts,
                    "source_language": transfer_example.source_language
                }
            }
            samples.append(sample)
        
        logger.info(f"Created {len(samples)} cross-language transfer samples")
        return samples
    
    def _extract_test_cases_from_content(self, test_content: str, language: str) -> List[str]:
        """Extract test cases from test file content"""
        test_cases = []
        
        # Language-specific test case extraction
        if language == "python":
            # Look for assert statements
            lines = test_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('assert ') or 'assertEqual' in line:
                    test_cases.append(line)
        elif language == "javascript":
            # Look for expect statements
            lines = test_content.split('\n')
            for line in lines:
                line = line.strip()
                if 'expect(' in line or 'assert' in line:
                    test_cases.append(line)
        elif language == "java":
            # Look for assert methods
            lines = test_content.split('\n')
            for line in lines:
                line = line.strip()
                if 'assert' in line.lower() and ('(' in line and ')' in line):
                    test_cases.append(line)
        
        return test_cases[:5]  # Limit to 5 test cases
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "python": "py",
            "javascript": "js", 
            "java": "java",
            "cpp": "cpp",
            "go": "go",
            "rust": "rs"
        }
        return extensions.get(language, "txt")
    
    def _apply_quality_filters(self, samples: List[Dict]) -> List[Dict]:
        """Apply quality filters to samples"""
        filtered_samples = []
        
        for sample in samples:
            # Complexity filter
            complexity = sample.get("complexity", 0.5)
            if not (self.config.min_complexity_score <= complexity <= self.config.max_complexity_score):
                continue
            
            # Language filter (apply language weights)
            language = sample["language"]
            if language not in self.config.language_weights:
                continue
            
            # Random sampling based on language weights
            if random.random() > self.config.language_weights[language]:
                continue
            
            filtered_samples.append(sample)
        
        logger.info(f"Applied quality filters: {len(samples)} -> {len(filtered_samples)} samples")
        return filtered_samples
    
    def _apply_curriculum_ordering(self, samples: List[Dict]) -> List[Dict]:
        """Apply curriculum learning ordering (easy to hard)"""
        # Sort by complexity score
        samples.sort(key=lambda x: x.get("complexity", 0.5))
        
        # Group into curriculum phases
        phase_size = len(samples) // 3
        easy_samples = samples[:phase_size]
        medium_samples = samples[phase_size:2*phase_size]
        hard_samples = samples[2*phase_size:]
        
        # Shuffle within each phase
        random.shuffle(easy_samples)
        random.shuffle(medium_samples)
        random.shuffle(hard_samples)
        
        # Combine phases
        curriculum_samples = easy_samples + medium_samples + hard_samples
        
        logger.info(f"Applied curriculum learning: Easy({len(easy_samples)}), Medium({len(medium_samples)}), Hard({len(hard_samples)})")
        return curriculum_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample"""
        sample = self.samples[idx]
        
        # Process input through HRM input processor
        processed_input = self.input_processor.process_input(sample["input"])
        
        return {
            "input_tokens": processed_input.input_tokens,
            "language_id": processed_input.language_id,
            "task_id": processed_input.task_id,
            "attention_mask": processed_input.attention_mask,
            "puzzle_identifier": processed_input.puzzle_identifier,
            "target_code": sample["target_code"],
            "sample_type": sample["type"],
            "complexity": sample["complexity"],
            "metadata": processed_input.metadata
        }

class PolyglotTrainingPipeline:
    """
    Main training pipeline that integrates Polyglot data with HRM architecture
    """
    
    def __init__(self, 
                 benchmark_root: str,
                 hrm_model: HRMCodeGenerationModel,
                 config: PolyglotTrainingConfig):
        """
        Initialize training pipeline
        
        Args:
            benchmark_root: Path to polyglot-benchmark directory
            hrm_model: HRM code generation model
            config: Training configuration
        """
        self.benchmark_root = Path(benchmark_root)
        self.hrm_model = hrm_model
        self.config = config
        
        # Initialize components
        self.extractor = PolyglotBenchmarkExtractor(str(benchmark_root))
        self.input_processor = CodeGenerationInputProcessor()
        self.output_generator = CodeOutputGenerator()
        
        # Data storage
        self.polyglot_data = {}
        self.training_dataset = None
        self.validation_dataset = None
        
        logger.info("Initialized PolyglotTrainingPipeline")
    
    def extract_and_process_data(self) -> Dict:
        """Extract and process all Polyglot benchmark data"""
        logger.info("Extracting Polyglot benchmark data...")
        
        # 1. Extract problems
        problems = self.extractor.extract_all_problems()
        
        # 2. Generate diff-based training examples
        diff_generator = DiffBasedTrainingGenerator(problems)
        diff_examples = diff_generator.generate_all_training_examples(
            max_examples_per_problem=self.config.max_diff_examples_per_problem
        )
        
        # 3. Generate cross-language transfer examples
        cross_lang_mapper = CrossLanguageProblemMapper(problems)
        transfer_examples = cross_lang_mapper.generate_transfer_examples(
            max_examples_per_pair=self.config.max_transfer_examples_per_pair
        )
        
        # Store processed data
        self.polyglot_data = {
            "problems": problems,
            "diff_examples": diff_examples,
            "transfer_examples": transfer_examples,
            "extraction_stats": self.extractor.language_stats,
            "diff_stats": diff_generator.get_statistics(),
            "transfer_stats": cross_lang_mapper.get_cross_language_statistics()
        }
        
        logger.info("Data extraction and processing complete")
        return self.polyglot_data
    
    def create_training_datasets(self, train_split: float = 0.8) -> Tuple[PolyglotCodeDataset, PolyglotCodeDataset]:
        """Create training and validation datasets"""
        if not self.polyglot_data:
            raise ValueError("Must call extract_and_process_data() first")
        
        logger.info("Creating training datasets...")
        
        # Create full dataset
        full_dataset = PolyglotCodeDataset(
            self.polyglot_data,
            self.input_processor,
            self.config
        )
        
        # Split into train/validation
        total_samples = len(full_dataset)
        train_size = int(train_split * total_samples)
        val_size = total_samples - train_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_samples))
        
        # Create subset datasets
        train_samples = [full_dataset.samples[i] for i in train_indices]
        val_samples = [full_dataset.samples[i] for i in val_indices]
        
        self.training_dataset = PolyglotCodeDataset(
            {
                "problems": self.polyglot_data["problems"],
                "diff_examples": [ex for ex in self.polyglot_data["diff_examples"] 
                                if any(s.get("problem_id") == getattr(ex, "problem_id", None) 
                                      for s in train_samples)],
                "transfer_examples": [ex for ex in self.polyglot_data["transfer_examples"]
                                    if any(s.get("problem_id") == ex.problem_id 
                                          for s in train_samples)]
            },
            self.input_processor,
            self.config
        )
        
        self.validation_dataset = PolyglotCodeDataset(
            {
                "problems": self.polyglot_data["problems"],
                "diff_examples": [ex for ex in self.polyglot_data["diff_examples"]
                                if any(s.get("problem_id") == getattr(ex, "problem_id", None)
                                      for s in val_samples)],
                "transfer_examples": [ex for ex in self.polyglot_data["transfer_examples"]
                                    if any(s.get("problem_id") == ex.problem_id
                                          for s in val_samples)]
            },
            self.input_processor,
            self.config
        )
        
        logger.info(f"Created datasets: Train({len(self.training_dataset)}), Val({len(self.validation_dataset)})")
        
        return self.training_dataset, self.validation_dataset
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        if self.training_dataset is None or self.validation_dataset is None:
            raise ValueError("Must call create_training_datasets() first")
        
        def collate_fn(batch):
            """Custom collate function for variable-length sequences"""
            # Stack tensors
            input_tokens = torch.stack([item["input_tokens"] for item in batch])
            language_ids = torch.stack([item["language_id"] for item in batch])
            task_ids = torch.stack([item["task_id"] for item in batch])
            attention_masks = torch.stack([item["attention_mask"] for item in batch])
            puzzle_identifiers = torch.stack([item["puzzle_identifier"] for item in batch])
            
            # Collect other data
            target_codes = [item["target_code"] for item in batch]
            sample_types = [item["sample_type"] for item in batch]
            complexities = torch.tensor([item["complexity"] for item in batch])
            
            return {
                "input_tokens": input_tokens,
                "language_ids": language_ids,
                "task_ids": task_ids,
                "attention_masks": attention_masks,
                "puzzle_identifiers": puzzle_identifiers,
                "target_codes": target_codes,
                "sample_types": sample_types,
                "complexities": complexities
            }
        
        train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders: Train({len(train_loader)} batches), Val({len(val_loader)} batches)")
        
        return train_loader, val_loader
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive statistics about the training data"""
        if not self.polyglot_data:
            return {}
        
        stats = {
            "total_problems": len(self.polyglot_data["problems"]),
            "total_diff_examples": len(self.polyglot_data["diff_examples"]),
            "total_transfer_examples": len(self.polyglot_data["transfer_examples"]),
            "extraction_stats": self.polyglot_data.get("extraction_stats", {}),
            "diff_stats": self.polyglot_data.get("diff_stats", {}),
            "transfer_stats": self.polyglot_data.get("transfer_stats", {})
        }
        
        if self.training_dataset:
            stats["training_samples"] = len(self.training_dataset)
            
            # Sample type distribution
            sample_types = {}
            for sample in self.training_dataset.samples:
                sample_type = sample["type"]
                sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
            stats["training_sample_types"] = sample_types
            
            # Language distribution
            languages = {}
            for sample in self.training_dataset.samples:
                lang = sample["language"]
                languages[lang] = languages.get(lang, 0) + 1
            stats["training_languages"] = languages
            
            # Complexity distribution
            complexities = [sample["complexity"] for sample in self.training_dataset.samples]
            stats["complexity_stats"] = {
                "mean": np.mean(complexities),
                "std": np.std(complexities),
                "min": np.min(complexities),
                "max": np.max(complexities)
            }
        
        return stats
    
    def export_training_data(self, output_dir: str):
        """Export processed training data for external use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting training data to {output_path}")
        
        # Export raw problems data
        problems_file = output_path / "polyglot_problems.json"
        self.extractor.export_to_json(str(problems_file))
        
        # Export processed training samples
        if self.training_dataset:
            training_file = output_path / "training_samples.json"
            training_data = {
                "config": asdict(self.config),
                "samples": self.training_dataset.samples,
                "statistics": self.get_data_statistics()
            }
            
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Export validation samples
        if self.validation_dataset:
            validation_file = output_path / "validation_samples.json"
            validation_data = {
                "samples": self.validation_dataset.samples,
                "count": len(self.validation_dataset)
            }
            
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Export comprehensive statistics
        stats_file = output_path / "data_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_data_statistics(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Export complete: {output_path}")
    
    def print_data_summary(self):
        """Print a comprehensive summary of the training data"""
        stats = self.get_data_statistics()
        
        print("\n" + "="*60)
        print("POLYGLOT HRM TRAINING DATA SUMMARY")
        print("="*60)
        
        print(f"Total Problems: {stats.get('total_problems', 0)}")
        print(f"Total Diff Examples: {stats.get('total_diff_examples', 0)}")
        print(f"Total Transfer Examples: {stats.get('total_transfer_examples', 0)}")
        
        if "training_samples" in stats:
            print(f"\nTraining Samples: {stats['training_samples']}")
            
            print("\nSample Type Distribution:")
            for sample_type, count in stats.get("training_sample_types", {}).items():
                percentage = count / stats['training_samples'] * 100
                print(f"  {sample_type}: {count} ({percentage:.1f}%)")
            
            print("\nLanguage Distribution:")
            for lang, count in stats.get("training_languages", {}).items():
                percentage = count / stats['training_samples'] * 100
                print(f"  {lang}: {count} ({percentage:.1f}%)")
            
            complexity_stats = stats.get("complexity_stats", {})
            if complexity_stats:
                print(f"\nComplexity Statistics:")
                print(f"  Mean: {complexity_stats.get('mean', 0):.3f}")
                print(f"  Std: {complexity_stats.get('std', 0):.3f}")
                print(f"  Range: {complexity_stats.get('min', 0):.3f} - {complexity_stats.get('max', 0):.3f}")
        
        extraction_stats = stats.get("extraction_stats", {})
        if extraction_stats:
            print(f"\nPer-Language Extraction Statistics:")
            for lang, lang_stats in extraction_stats.items():
                print(f"  {lang.upper()}: {lang_stats.get('problems', 0)} problems, "
                      f"{lang_stats.get('files', 0)} files, {lang_stats.get('total_lines', 0)} lines")

def main():
    """Main function for testing the integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate Polyglot benchmark with HRM")
    parser.add_argument("--benchmark-root", required=True,
                       help="Path to polyglot-benchmark directory")
    parser.add_argument("--output-dir", default="polyglot_training_data",
                       help="Output directory for processed data")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--max-problems", type=int, default=50,
                       help="Maximum problems per language")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PolyglotTrainingConfig(
        max_problems_per_language=args.max_problems,
        batch_size=args.batch_size
    )
    
    # Create mock HRM model for testing
    class MockHRMModel:
        pass
    
    mock_model = MockHRMModel()
    
    # Initialize pipeline
    pipeline = PolyglotTrainingPipeline(
        args.benchmark_root,
        mock_model,
        config
    )
    
    # Extract and process data
    polyglot_data = pipeline.extract_and_process_data()
    
    # Create datasets
    train_dataset, val_dataset = pipeline.create_training_datasets()
    
    # Create data loaders
    train_loader, val_loader = pipeline.create_data_loaders()
    
    # Print summary
    pipeline.print_data_summary()
    
    # Export data
    pipeline.export_training_data(args.output_dir)
    
    # Test a few batches
    print(f"\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Test first 2 batches
            break
        
        print(f"Batch {i+1}:")
        print(f"  Input tokens shape: {batch['input_tokens'].shape}")
        print(f"  Language IDs: {batch['language_ids'].flatten().tolist()}")
        print(f"  Sample types: {batch['sample_types']}")
        print(f"  Complexities: {batch['complexities'].tolist()}")

if __name__ == "__main__":
    main()