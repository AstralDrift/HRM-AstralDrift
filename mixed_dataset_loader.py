#!/usr/bin/env python3
"""
Mixed Dataset Loader for HRM Code Generation

Combines SWE-Smith and LiveCodeBench datasets with configurable mixing ratios.
Supports proper sampling, validation, and efficient loading for training.

Default ratio: 70% SWE-Smith (real-world software engineering) + 30% LiveCodeBench (algorithmic)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
import pydantic
from dataclasses import dataclass
import logging

from utils.error_handling import robust_error_handler, DatasetError, validate_dataset_path


@dataclass
class MixedDatasetConfig:
    """Configuration for mixed dataset loading"""
    swe_smith_path: str = "data/swe-smith-1k"
    livecodebench_path: str = "data/livecodebench_real"
    swe_smith_ratio: float = 0.7  # 70% SWE-Smith
    livecodebench_ratio: float = 0.3  # 30% LiveCodeBench
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    max_input_length: int = 1024
    max_output_length: int = 512
    batch_size: int = 6
    shuffle: bool = True
    validation_split: float = 0.1
    seed: int = 42


class MixedCodeGenerationDataset(Dataset):
    """Mixed dataset combining SWE-Smith and LiveCodeBench"""
    
    def __init__(self, config: MixedDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load both datasets
        self.swe_smith_data = self._load_swe_smith_data()
        self.livecodebench_data = self._load_livecodebench_data()
        
        # Create mixed dataset with proper ratios
        self.mixed_instances = self._create_mixed_dataset()
        
        # Split train/validation
        if config.validation_split > 0:
            self.train_instances, self.val_instances = self._split_dataset()
        else:
            self.train_instances = self.mixed_instances
            self.val_instances = []
            
        # Select appropriate split
        if split == "train":
            self.instances = self.train_instances
        elif split == "validation" or split == "val":
            self.instances = self.val_instances
        else:
            raise ValueError(f"Unknown split: {split}")
            
        self._log_dataset_stats()
        
    @robust_error_handler(max_retries=2)
    def _load_swe_smith_data(self) -> List[Dict[str, Any]]:
        """Load SWE-Smith dataset with error handling"""
        try:
            swe_path = Path(self.config.swe_smith_path) / "instances.json"
            
            if not swe_path.exists():
                logging.warning(f"⚠️  SWE-Smith dataset not found at {swe_path}")
                return []
            
            # Validate file before loading
            if swe_path.stat().st_size == 0:
                raise DatasetError(f"SWE-Smith dataset file is empty: {swe_path}")
            
            with open(swe_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise DatasetError(f"SWE-Smith dataset must be a list, got {type(data)}")
            
            # Validate data structure
            if data and not isinstance(data[0], dict):
                raise DatasetError("SWE-Smith dataset items must be dictionaries")
            
            logging.info(f"✅ Loaded {len(data)} SWE-Smith instances")
            return data
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in SWE-Smith dataset: {e}") from e
        except Exception as e:
            raise DatasetError(f"Failed to load SWE-Smith dataset: {e}") from e
        
    @robust_error_handler(max_retries=2)
    def _load_livecodebench_data(self) -> List[Dict[str, Any]]:
        """Load LiveCodeBench dataset with error handling"""
        try:
            lcb_path = Path(self.config.livecodebench_path) / "livecodebench_real.json"
            
            if not lcb_path.exists():
                logging.warning(f"⚠️  LiveCodeBench dataset not found at {lcb_path}")
                return []
            
            # Validate file before loading
            if lcb_path.stat().st_size == 0:
                raise DatasetError(f"LiveCodeBench dataset file is empty: {lcb_path}")
            
            with open(lcb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise DatasetError(f"LiveCodeBench dataset must be a list, got {type(data)}")
            
            # Validate data structure
            if data and not isinstance(data[0], dict):
                raise DatasetError("LiveCodeBench dataset items must be dictionaries")
            
            logging.info(f"✅ Loaded {len(data)} LiveCodeBench instances")
            return data
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in LiveCodeBench dataset: {e}") from e
        except Exception as e:
            raise DatasetError(f"Failed to load LiveCodeBench dataset: {e}") from e
        
    def _create_mixed_dataset(self) -> List[Dict[str, Any]]:
        """Create mixed dataset with specified ratios"""
        total_swe = len(self.swe_smith_data)
        total_lcb = len(self.livecodebench_data)
        
        if total_swe == 0 and total_lcb == 0:
            raise ValueError("No data found in either dataset")
        
        # Calculate target counts based on smaller dataset
        if total_swe == 0:
            swe_count = 0
            lcb_count = total_lcb
        elif total_lcb == 0:
            swe_count = total_swe
            lcb_count = 0
        else:
            # Calculate balanced mixing
            total_target = min(
                int(total_swe / self.config.swe_smith_ratio),
                int(total_lcb / self.config.livecodebench_ratio)
            )
            swe_count = int(total_target * self.config.swe_smith_ratio)
            lcb_count = int(total_target * self.config.livecodebench_ratio)
        
        # Sample from datasets
        mixed_data = []
        
        if swe_count > 0:
            swe_sample = random.sample(self.swe_smith_data, min(swe_count, total_swe))
            # Normalize SWE-Smith format
            for item in swe_sample:
                normalized = self._normalize_swe_smith_instance(item)
                mixed_data.append(normalized)
                
        if lcb_count > 0:
            lcb_sample = random.sample(self.livecodebench_data, min(lcb_count, total_lcb))
            # Normalize LiveCodeBench format
            for item in lcb_sample:
                normalized = self._normalize_livecodebench_instance(item)
                mixed_data.append(normalized)
        
        # Shuffle the mixed dataset
        if self.config.shuffle:
            random.shuffle(mixed_data)
            
        print(f"✅ Created mixed dataset: {swe_count} SWE-Smith + {lcb_count} LiveCodeBench = {len(mixed_data)} total")
        return mixed_data
        
    def _normalize_swe_smith_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize SWE-Smith instance to common format"""
        return {
            "instance_id": instance.get("instance_id", "unknown"),
            "dataset_source": "swe_smith",
            "input_text": instance.get("input_text", ""),
            "target_text": instance.get("target_text", ""),
            "domain": "software_engineering",
            "language": "python",  # Most SWE-Smith is Python
            "complexity": 0.8,  # Real-world software engineering is complex
            "metadata": {
                "coordination_target": instance.get("coordination_target", {}),
                "tool_workflow": instance.get("tool_workflow", {}),
                "original_source": "swe_smith"
            }
        }
        
    def _normalize_livecodebench_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LiveCodeBench instance to common format"""
        return {
            "instance_id": instance.get("instance_id", "unknown"),
            "dataset_source": "livecodebench",
            "input_text": instance.get("input_text", ""),
            "target_text": instance.get("target_text", ""),
            "domain": instance.get("domain", "algorithms"),
            "language": instance.get("language", "python"),
            "complexity": instance.get("complexity", 0.5),
            "metadata": {
                "original_source": "livecodebench",
                **instance.get("metadata", {})
            }
        }
        
    def _split_dataset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/validation"""
        total_size = len(self.mixed_instances)
        val_size = int(total_size * self.config.validation_split)
        
        # Ensure balanced splits
        indices = list(range(total_size))
        random.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_data = [self.mixed_instances[i] for i in train_indices]
        val_data = [self.mixed_instances[i] for i in val_indices]
        
        return train_data, val_data
        
    def _log_dataset_stats(self):
        """Log dataset statistics"""
        if not self.instances:
            print("⚠️  No instances in current split")
            return
            
        # Count by source
        swe_count = sum(1 for item in self.instances if item["dataset_source"] == "swe_smith")
        lcb_count = sum(1 for item in self.instances if item["dataset_source"] == "livecodebench")
        
        # Count by domain
        domains = {}
        languages = {}
        
        for item in self.instances:
            domain = item.get("domain", "unknown")
            language = item.get("language", "unknown")
            
            domains[domain] = domains.get(domain, 0) + 1
            languages[language] = languages.get(language, 0) + 1
            
        print(f"📊 {self.split.upper()} Dataset Statistics:")
        print(f"   Total instances: {len(self.instances)}")
        print(f"   SWE-Smith: {swe_count} ({swe_count/len(self.instances)*100:.1f}%)")
        print(f"   LiveCodeBench: {lcb_count} ({lcb_count/len(self.instances)*100:.1f}%)")
        print(f"   Domains: {domains}")
        print(f"   Languages: {languages}")
        
    def __len__(self) -> int:
        return len(self.instances)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single instance"""
        instance = self.instances[idx]
        
        # Tokenize input and target
        input_tokens = self.tokenizer.encode(
            instance["input_text"],
            max_length=self.config.max_input_length,
            truncation=True,
            padding=False
        )
        
        target_tokens = self.tokenizer.encode(
            instance["target_text"],
            max_length=self.config.max_output_length,
            truncation=True,
            padding=False
        )
        
        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "instance_id": instance["instance_id"],
            "dataset_source": instance["dataset_source"],
            "domain": instance["domain"],
            "language": instance["language"],
            "complexity": instance["complexity"],
            "metadata": instance["metadata"]
        }


def create_mixed_dataloader(config: MixedDatasetConfig, split: str = "train", model_seq_len: int = 1024) -> DataLoader:
    """Create a DataLoader for the mixed dataset"""
    
    # Create dataset first to get tokenizer
    dataset = MixedCodeGenerationDataset(config, split)
    tokenizer = dataset.tokenizer
    
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        # Use fixed model sequence length for padding instead of max in batch
        # This ensures all batches have the same sequence length as expected by the model
        
        # Pad sequences to fixed model sequence length
        combined_sequences = []
        labels = []
        puzzle_identifiers = []
        
        for i, item in enumerate(batch):
            input_ids = item["input_ids"]
            target_ids = item["target_ids"]
            
            # Combine input and target first
            combined_seq = torch.cat([input_ids, target_ids])
            
            # Truncate or pad to model sequence length
            if len(combined_seq) > model_seq_len:
                # Truncate - keep input portion and truncate target
                input_len = len(input_ids)
                if input_len >= model_seq_len:
                    # Input is too long, truncate input and no target
                    combined_seq = input_ids[:model_seq_len]
                    label_seq = torch.full((model_seq_len,), -100, dtype=torch.long)  # All ignored
                else:
                    # Keep full input, truncate target
                    target_len = model_seq_len - input_len
                    combined_seq = torch.cat([input_ids, target_ids[:target_len]])
                    label_seq = torch.full((model_seq_len,), -100, dtype=torch.long)
                    label_seq[input_len:] = target_ids[:target_len].clone()
            else:
                # Pad to model sequence length
                padding_len = model_seq_len - len(combined_seq)
                combined_seq = torch.cat([combined_seq, torch.full((padding_len,), tokenizer.pad_token_id)])
                
                # Create labels - only compute loss on target portion
                label_seq = torch.full((model_seq_len,), -100, dtype=torch.long)  # Ignore input portion
                input_len = len(input_ids)
                if input_len < model_seq_len:
                    target_end = min(input_len + len(target_ids), model_seq_len)
                    label_seq[input_len:target_end] = target_ids[:target_end-input_len].clone()
            
            # Ignore padding tokens in labels
            label_seq[combined_seq == tokenizer.pad_token_id] = -100
            
            combined_sequences.append(combined_seq)
            labels.append(label_seq)
            
            # Create puzzle identifiers (use batch index as ID)
            puzzle_identifiers.append(torch.tensor([i], dtype=torch.long))
        
        return {
            "inputs": torch.stack(combined_sequences),  # HRM model expects "inputs"
            "labels": torch.stack(labels),  # HRM model expects "labels" 
            "puzzle_identifiers": torch.stack(puzzle_identifiers),  # HRM model expects "puzzle_identifiers"
            # Keep metadata for debugging
            "metadata": [item["metadata"] for item in batch],
            "instance_ids": [item["instance_id"] for item in batch],
            "dataset_sources": [item["dataset_source"] for item in batch],
            "domains": [item["domain"] for item in batch],
            "languages": [item["language"] for item in batch],
            "complexities": [item["complexity"] for item in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle if split == "train" else False,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=torch.cuda.is_available()
    )


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Mixed Dataset Loader")
    
    config = MixedDatasetConfig(
        swe_smith_ratio=0.7,
        livecodebench_ratio=0.3,
        batch_size=4,
        validation_split=0.2
    )
    
    # Test train dataloader
    train_loader = create_mixed_dataloader(config, "train")
    print(f"✅ Train loader created with {len(train_loader.dataset)} instances")
    
    # Test validation dataloader
    val_loader = create_mixed_dataloader(config, "validation")
    print(f"✅ Validation loader created with {len(val_loader.dataset)} instances")
    
    # Test a batch
    try:
        batch = next(iter(train_loader))
        print(f"✅ Sample batch shape: input {batch['input_ids'].shape}, target {batch['target_ids'].shape}")
        print(f"   Batch sources: {batch['dataset_sources']}")
        print(f"   Batch domains: {batch['domains']}")
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        
    print("🎉 Mixed dataset loader test complete!")