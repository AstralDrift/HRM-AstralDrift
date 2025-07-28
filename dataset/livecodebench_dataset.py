"""
LiveCodeBench Dataset Module for HRM Training

This module provides PyTorch Dataset classes for efficient loading and batching
of LiveCodeBench data for HRM training. It supports multi-scenario training,
dynamic batching, and efficient memory management.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizer

from models.code_generation.input_processor import (
    CodeGenerationInput,
    CodeGenerationInputProcessor,
    CodeGenerationTask,
    ProgrammingLanguage
)

@dataclass
class LiveCodeBenchSample:
    """Single training sample from LiveCodeBench"""
    problem_id: str
    scenario: str
    input_text: str
    target_text: str
    language: ProgrammingLanguage
    task_type: CodeGenerationTask
    complexity_score: float
    metadata: Dict

class LiveCodeBenchDataset(Dataset):
    """PyTorch Dataset for LiveCodeBench training data"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 max_seq_len: int = 2048,
                 vocab_size: int = 40000,
                 scenario_weights: Optional[Dict[str, float]] = None,
                 augment_prob: float = 0.3):
        """
        Initialize LiveCodeBench dataset
        
        Args:
            data_dir: Directory containing processed dataset
            split: Dataset split (train/val/test)
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            scenario_weights: Weights for different scenarios
            augment_prob: Probability of applying augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.augment_prob = augment_prob
        
        # Default scenario weights
        self.scenario_weights = scenario_weights or {
            "codegeneration": 1.0,
            "selfrepair": 0.8,
            "testoutputprediction": 0.6,
            "codeexecution": 0.5
        }
        
        # Initialize input processor
        self.input_processor = CodeGenerationInputProcessor(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        
        # Load dataset
        self.samples = self._load_samples()
        self.scenario_indices = self._build_scenario_indices()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_dataset_stats()
    
    def _load_samples(self) -> List[LiveCodeBenchSample]:
        """Load samples from processed dataset files"""
        samples = []
        
        # Load the main dataset files
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load inputs and labels
        inputs_file = split_dir / "inputs.npy"
        labels_file = split_dir / "labels.npy"
        metadata_file = split_dir / "metadata.json"
        
        if not all(f.exists() for f in [inputs_file, labels_file, metadata_file]):
            raise FileNotFoundError("Required dataset files not found")
        
        # Load numpy arrays
        inputs = np.load(inputs_file)
        labels = np.load(labels_file)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
        
        # Create samples
        for i, (input_tokens, label_tokens, metadata) in enumerate(zip(inputs, labels, metadata_list)):
            sample = LiveCodeBenchSample(
                problem_id=metadata.get("problem_id", f"sample_{i}"),
                scenario=metadata.get("scenario", "codegeneration"),
                input_text=metadata.get("input_text", ""),
                target_text=metadata.get("target_text", ""),
                language=ProgrammingLanguage(metadata.get("language", "python")),
                task_type=CodeGenerationTask(metadata.get("task_type", "generation")),
                complexity_score=metadata.get("complexity_score", 0.5),
                metadata=metadata
            )
            samples.append(sample)
        
        return samples
    
    def _build_scenario_indices(self) -> Dict[str, List[int]]:
        """Build indices for each scenario for balanced sampling"""
        scenario_indices = {}
        
        for i, sample in enumerate(self.samples):
            scenario = sample.scenario
            if scenario not in scenario_indices:
                scenario_indices[scenario] = []
            scenario_indices[scenario].append(i)
        
        return scenario_indices
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"Dataset Statistics for {self.split}:")
        
        # Scenario distribution
        scenario_counts = {}
        language_counts = {}
        task_counts = {}
        
        for sample in self.samples:
            scenario = sample.scenario
            language = sample.language.value
            task = sample.task_type.value
            
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            language_counts[language] = language_counts.get(language, 0) + 1
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print("  Scenarios:")
        for scenario, count in sorted(scenario_counts.items()):
            print(f"    {scenario}: {count} ({count/len(self.samples)*100:.1f}%)")
        
        print("  Languages:")
        for language, count in sorted(language_counts.items()):
            print(f"    {language}: {count} ({count/len(self.samples)*100:.1f}%)")
        
        print("  Task Types:")
        for task, count in sorted(task_counts.items()):
            print(f"    {task}: {count} ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        sample = self.samples[idx]
        
        # Apply dynamic augmentation during training
        if self.split == "train" and random.random() < self.augment_prob:
            sample = self._augment_sample(sample)
        
        # Create CodeGenerationInput
        code_input = CodeGenerationInput(
            problem_description=sample.input_text,
            language=sample.language,
            task_type=sample.task_type,
            existing_code=sample.metadata.get("existing_code"),
            test_cases=sample.metadata.get("test_cases"),
            error_message=sample.metadata.get("error_message")
        )
        
        # Process through input processor
        processed = self.input_processor.process_input(code_input)
        
        # Tokenize target
        target_tokens = self._tokenize_target(sample.target_text)
        
        # Apply scenario weighting
        scenario_weight = self.scenario_weights.get(sample.scenario, 1.0)
        
        return {
            "inputs": processed.input_tokens,
            "targets": target_tokens,
            "attention_mask": processed.attention_mask,
            "puzzle_identifier": processed.puzzle_identifier,
            "language_id": processed.language_id,
            "task_id": processed.task_id,
            "complexity_score": torch.tensor([processed.metadata["complexity_score"]], dtype=torch.float),
            "scenario_weight": torch.tensor([scenario_weight], dtype=torch.float),
            "scenario": sample.scenario,
            "problem_id": sample.problem_id
        }
    
    def _tokenize_target(self, target_text: str) -> torch.Tensor:
        """Tokenize target text for training"""
        tokens = self.input_processor.tokenizer.base_tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len
        )
        
        # Pad or truncate to match max_seq_len
        input_ids = tokens["input_ids"].squeeze(0)
        if input_ids.size(0) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            pad_length = self.max_seq_len - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
        
        return input_ids
    
    def _augment_sample(self, sample: LiveCodeBenchSample) -> LiveCodeBenchSample:
        """Apply dynamic augmentation to a sample"""
        # Simple augmentations that preserve meaning
        augmented_input = sample.input_text
        
        # Text variations
        replacements = [
            ("Write a function", "Implement a function"),
            ("Create a function", "Write a function"),
            ("that returns", "which returns"),
            ("array", "list"),
            ("string", "text"),
        ]
        
        # Apply random subset of replacements
        for old, new in random.sample(replacements, min(2, len(replacements))):
            if old in augmented_input:
                augmented_input = augmented_input.replace(old, new)
        
        # Create augmented sample
        return LiveCodeBenchSample(
            problem_id=f"{sample.problem_id}_aug",
            scenario=sample.scenario,
            input_text=augmented_input,
            target_text=sample.target_text,
            language=sample.language,
            task_type=sample.task_type,
            complexity_score=sample.complexity_score,
            metadata={**sample.metadata, "is_augmented": True}
        )
    
    def get_scenario_sample(self, scenario: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get a random sample from a specific scenario"""
        if scenario not in self.scenario_indices:
            return None
        
        idx = random.choice(self.scenario_indices[scenario])
        return self[idx]
    
    def get_complexity_sample(self, min_complexity: float = 0.0, max_complexity: float = 1.0) -> Optional[Dict[str, torch.Tensor]]:
        """Get a random sample within complexity range"""
        valid_indices = [
            i for i, sample in enumerate(self.samples)
            if min_complexity <= sample.complexity_score <= max_complexity
        ]
        
        if not valid_indices:
            return None
        
        idx = random.choice(valid_indices)
        return self[idx]

class BalancedScenarioSampler(Sampler):
    """Sampler that ensures balanced sampling across scenarios"""
    
    def __init__(self, dataset: LiveCodeBenchDataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.scenario_indices = dataset.scenario_indices
        self.scenarios = list(self.scenario_indices.keys())
        
        # Calculate number of samples per scenario per batch
        self.samples_per_scenario = max(1, batch_size // len(self.scenarios))
        
    def __iter__(self) -> Iterator[int]:
        # Create balanced batches
        all_indices = []
        
        # Calculate total batches needed
        total_samples = len(self.dataset)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Sample from each scenario
            for scenario in self.scenarios:
                scenario_pool = self.scenario_indices[scenario]
                sampled = random.sample(scenario_pool, 
                                      min(self.samples_per_scenario, len(scenario_pool)))
                batch_indices.extend(sampled)
            
            # Fill remaining slots if needed
            while len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                # Sample from all scenarios
                all_scenario_indices = []
                for indices in self.scenario_indices.values():
                    all_scenario_indices.extend(indices)
                
                extra_samples = random.sample(all_scenario_indices, 
                                            min(remaining, len(all_scenario_indices)))
                batch_indices.extend(extra_samples)
            
            # Shuffle batch and truncate to exact batch size
            random.shuffle(batch_indices)
            all_indices.extend(batch_indices[:self.batch_size])
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return len(self.dataset)

class LiveCodeBenchDataLoader:
    """Specialized DataLoader for LiveCodeBench training"""
    
    def __init__(self,
                 dataset: LiveCodeBenchDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 balanced_sampling: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Choose sampler
        if balanced_sampling and dataset.split == "train":
            sampler = BalancedScenarioSampler(dataset, batch_size)
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        # Create DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        # Stack tensors
        collated = {}
        
        # Handle tensor fields
        tensor_fields = [
            "inputs", "targets", "attention_mask", "puzzle_identifier",
            "language_id", "task_id", "complexity_score", "scenario_weight"
        ]
        
        for field in tensor_fields:
            if field in batch[0]:
                collated[field] = torch.stack([item[field] for item in batch])
        
        # Handle string fields
        string_fields = ["scenario", "problem_id"]
        for field in string_fields:
            if field in batch[0]:
                collated[field] = [item[field] for item in batch]
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

def create_livecodebench_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    vocab_size: int = 40000,
    scenario_weights: Optional[Dict[str, float]] = None,
    num_workers: int = 4,
    balanced_sampling: bool = True
) -> Tuple[LiveCodeBenchDataLoader, LiveCodeBenchDataLoader]:
    """
    Create train and validation dataloaders for LiveCodeBench
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = LiveCodeBenchDataset(
        data_dir=data_dir,
        split="train",
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        scenario_weights=scenario_weights,
        augment_prob=0.3
    )
    
    val_dataset = LiveCodeBenchDataset(
        data_dir=data_dir,
        split="test",  # LiveCodeBench uses test split for validation
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        scenario_weights=scenario_weights,
        augment_prob=0.0  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = LiveCodeBenchDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        balanced_sampling=balanced_sampling,
        num_workers=num_workers
    )
    
    val_loader = LiveCodeBenchDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        balanced_sampling=False,  # No balanced sampling for validation
        num_workers=num_workers
    )
    
    return train_loader, val_loader

class AdaptiveComplexitySampler(Sampler):
    """Sampler that adaptively samples based on training progress and complexity"""
    
    def __init__(self, 
                 dataset: LiveCodeBenchDataset, 
                 batch_size: int = 32,
                 complexity_schedule: str = "linear"):  # linear, exponential, curriculum
        self.dataset = dataset
        self.batch_size = batch_size
        self.complexity_schedule = complexity_schedule
        self.epoch = 0
        
        # Group samples by complexity
        self.complexity_groups = self._build_complexity_groups()
    
    def _build_complexity_groups(self) -> Dict[str, List[int]]:
        """Group samples by complexity level"""
        groups = {"easy": [], "medium": [], "hard": []}
        
        for i, sample in enumerate(self.dataset.samples):
            complexity = sample.complexity_score
            if complexity < 0.33:
                groups["easy"].append(i)
            elif complexity < 0.67:
                groups["medium"].append(i)
            else:
                groups["hard"].append(i)
        
        return groups
    
    def set_epoch(self, epoch: int):
        """Set current epoch for complexity scheduling"""
        self.epoch = epoch
    
    def _get_complexity_weights(self) -> Dict[str, float]:
        """Get complexity weights based on current epoch"""
        if self.complexity_schedule == "linear":
            # Gradually increase hard problem weight
            progress = min(1.0, self.epoch / 100.0)  # Full weight at epoch 100
            return {
                "easy": 1.0 - 0.3 * progress,
                "medium": 1.0,
                "hard": 0.5 + 0.5 * progress
            }
        elif self.complexity_schedule == "exponential":
            # Exponential increase in hard problems
            progress = min(1.0, self.epoch / 50.0)
            hard_weight = np.exp(progress) / np.exp(1.0)
            return {
                "easy": 1.0 - 0.4 * progress,
                "medium": 1.0,
                "hard": 0.3 + 0.7 * hard_weight
            }
        else:  # curriculum
            # Start with easy, gradually add harder
            if self.epoch < 20:
                return {"easy": 1.0, "medium": 0.3, "hard": 0.1}
            elif self.epoch < 50:
                return {"easy": 0.8, "medium": 1.0, "hard": 0.4}
            else:
                return {"easy": 0.6, "medium": 1.0, "hard": 1.0}
    
    def __iter__(self) -> Iterator[int]:
        complexity_weights = self._get_complexity_weights()
        
        # Calculate samples per complexity level
        total_samples = len(self.dataset)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        all_indices = []
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Sample based on complexity weights
            for complexity, weight in complexity_weights.items():
                group_indices = self.complexity_groups[complexity]
                if not group_indices:
                    continue
                
                num_samples = int(self.batch_size * weight / sum(complexity_weights.values()))
                num_samples = min(num_samples, len(group_indices))
                
                if num_samples > 0:
                    sampled = random.sample(group_indices, num_samples)
                    batch_indices.extend(sampled)
            
            # Fill remaining slots
            while len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                all_group_indices = []
                for indices in self.complexity_groups.values():
                    all_group_indices.extend(indices)
                
                extra_samples = random.sample(all_group_indices,
                                            min(remaining, len(all_group_indices)))
                batch_indices.extend(extra_samples)
            
            random.shuffle(batch_indices)
            all_indices.extend(batch_indices[:self.batch_size])
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return len(self.dataset)

if __name__ == "__main__":
    # Test the dataset and dataloader
    print("Testing LiveCodeBench Dataset...")
    
    # Create test dataset (assuming data exists)
    data_dir = "data/livecodebench-hrm"
    
    try:
        # Create dataloaders
        train_loader, val_loader = create_livecodebench_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            balanced_sampling=True
        )
        
        print(f"Created train loader with {len(train_loader)} batches")
        print(f"Created val loader with {len(val_loader)} batches")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Input shape: {batch['inputs'].shape}")
            print(f"Target shape: {batch['targets'].shape}")
            print(f"Scenarios in batch: {set(batch['scenario'])}")
            break
            
        print("Dataset test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Run build_livecodebench_dataset.py first to create the dataset")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()