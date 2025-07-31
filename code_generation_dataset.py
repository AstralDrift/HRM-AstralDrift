"""
Code Generation Dataset for HRM

This dataset handles real code generation tasks with sophisticated analysis:
- Real GitHub issues and fixes
- Multi-agent coordination 
- Tool workflow planning
- Domain-specific complexity
- Repository context

Unlike PuzzleDataset, this preserves the rich structure of real software engineering tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
import pydantic
from torch.utils.data import ConcatDataset


class CodeGenerationDatasetConfig(pydantic.BaseModel):
    dataset_path: str
    split: str = "train"
    tokenizer_name: str = "microsoft/CodeBERT-base"
    max_input_length: int = 1024
    max_output_length: int = 512
    include_coordination_data: bool = True
    include_metadata: bool = True


class CodeGenerationDatasetMetadata(pydantic.BaseModel):
    """Metadata for code generation dataset"""
    vocab_size: int
    pad_token_id: int
    ignore_label_id: int
    max_input_length: int
    max_output_length: int
    num_instances: int
    num_domains: int
    num_languages: int
    average_complexity: float
    

class CodeGenerationDataset(Dataset):
    """Dataset for real code generation tasks"""
    
    def __init__(self, config: CodeGenerationDatasetConfig):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.split = config.split
        
        # Initialize tokenizer for code
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load the data
        self.instances = self._load_instances()
        self.metadata = self._create_metadata()
        
        print(f"✅ Loaded {len(self.instances)} code generation instances from {self.dataset_path}")
    
    def _load_instances(self) -> List[Dict[str, Any]]:
        """Load code generation instances"""
        # Handle both directory structure and direct JSON file
        if self.dataset_path.is_file() and self.dataset_path.suffix == '.json':
            # Direct JSON file
            instances_file = self.dataset_path
        else:
            # Directory structure with instances.json
            instances_file = self.dataset_path / "instances.json"
        
        if not instances_file.exists():
            raise FileNotFoundError(f"Instances file not found: {instances_file}")
        
        with open(instances_file, 'r') as f:
            instances = json.load(f)
            
        # Filter by split if needed (for now, we'll use all data as train)
        # In a real implementation, we'd have proper train/test splits
        return instances
    
    def _create_metadata(self) -> CodeGenerationDatasetMetadata:
        """Create metadata from the loaded instances"""
        
        # Analyze the dataset
        domains = set()
        languages = set()
        complexities = []
        
        for instance in self.instances:
            metadata = instance.get('metadata', {})
            
            domain = metadata.get('domain', 'unknown')
            domains.add(domain)
            
            instance_languages = metadata.get('languages', ['python'])
            languages.update(instance_languages)
            
            complexity = metadata.get('complexity', 0.5)
            complexities.append(complexity)
        
        return CodeGenerationDatasetMetadata(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            ignore_label_id=-100,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            num_instances=len(self.instances),
            num_domains=len(domains),
            num_languages=len(languages),
            average_complexity=np.mean(complexities) if complexities else 0.5
        )
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        
        instance = self.instances[idx]
        
        # Extract text data
        input_text = instance['input_text']
        target_text = instance['target_text']
        
        # Tokenize input (problem description, repo context, etc.)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.config.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (patch/code fix)
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.config.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Combine input and target for sequence-to-sequence format
        # This matches how HRM expects the data
        combined_sequence = torch.cat([
            input_encoding['input_ids'].squeeze(0)[:512],  # First half: input
            target_encoding['input_ids'].squeeze(0)[:512]  # Second half: target
        ])
        
        # Create labels - only compute loss on target portion
        labels = torch.full_like(combined_sequence, -100)  # Ignore input portion
        labels[512:] = combined_sequence[512:].clone()  # Only target portion
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding
        
        result = {
            'inputs': combined_sequence,
            'labels': labels,
            'puzzle_identifiers': torch.tensor([hash(instance.get('instance_id', f'instance_{idx}')) % 10000], dtype=torch.long)
        }
        
        # Add coordination data if requested
        if self.config.include_coordination_data:
            coord_target = instance.get('coordination_target', {})
            agent_assignments = coord_target.get('agent_assignments', [])
            
            # Encode agent information as additional context
            num_agents = len(agent_assignments)
            coordination_complexity = coord_target.get('coordination_complexity', 0.5)
            
            result.update({
                'num_agents': torch.tensor([num_agents], dtype=torch.long),
                'coordination_complexity': torch.tensor([coordination_complexity], dtype=torch.float)
            })
        
        # Add metadata if requested
        if self.config.include_metadata:
            metadata = instance.get('metadata', {})
            
            result.update({
                'complexity': torch.tensor([metadata.get('complexity', 0.5)], dtype=torch.float),
                'agents_needed': torch.tensor([metadata.get('agents_needed', 1)], dtype=torch.long),
                'hierarchical_depth': torch.tensor([metadata.get('hierarchical_depth', 2)], dtype=torch.long)
            })
        
        return result
    
    def get_metadata(self) -> CodeGenerationDatasetMetadata:
        """Get dataset metadata"""
        return self.metadata
    
    def get_sample_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = {}
        
        for idx in indices:
            sample = self[idx]
            for key, value in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)
        
        # Stack tensors
        for key, values in batch.items():
            batch[key] = torch.stack(values, dim=0)
        
        return batch


def create_code_generation_dataloader(
    config: CodeGenerationDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create a dataloader for code generation dataset"""
    
    # Handle mixed training mode
    if config.dataset_path == 'mixed':
        return create_mixed_dataloader(batch_size, shuffle, num_workers)
    
    dataset = CodeGenerationDataset(config)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    return dataloader, dataset.get_metadata()


def create_mixed_dataloader(
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, CodeGenerationDatasetMetadata]:
    """Create a mixed dataloader combining LiveCodeBench and SWE-Smith-1k"""
    
    # Create datasets for both sources
    lcb_config = CodeGenerationDatasetConfig(
        dataset_path='data/livecodebench_real/livecodebench_real.json',
        tokenizer_name="microsoft/CodeBERT-base",
        max_input_length=1024,
        max_output_length=512
    )
    
    swe_config = CodeGenerationDatasetConfig(
        dataset_path='data/swe-smith-1k',
        tokenizer_name="microsoft/CodeBERT-base", 
        max_input_length=1024,
        max_output_length=512
    )
    
    try:
        lcb_dataset = CodeGenerationDataset(lcb_config)
        print(f"✅ Loaded LiveCodeBench: {len(lcb_dataset)} instances")
    except Exception as e:
        print(f"⚠️ Failed to load LiveCodeBench: {e}")
        lcb_dataset = None
    
    try:
        swe_dataset = CodeGenerationDataset(swe_config)
        print(f"✅ Loaded SWE-Smith-1k: {len(swe_dataset)} instances")
    except Exception as e:
        print(f"⚠️ Failed to load SWE-Smith-1k: {e}")
        swe_dataset = None
    
    # Combine available datasets
    datasets = [d for d in [lcb_dataset, swe_dataset] if d is not None]
    
    if not datasets:
        raise ValueError("No datasets could be loaded for mixed training")
    elif len(datasets) == 1:
        print(f"ℹ️ Mixed training with only one dataset available")
        combined_dataset = datasets[0]
        metadata = datasets[0].get_metadata()
    else:
        print(f"🔄 Mixed training with {len(datasets)} datasets")
        combined_dataset = ConcatDataset(datasets)
        
        # Combine metadata from all datasets
        metadatas = [d.get_metadata() for d in datasets]
        metadata = CodeGenerationDatasetMetadata(
            vocab_size=metadatas[0].vocab_size,
            pad_token_id=metadatas[0].pad_token_id,
            ignore_label_id=metadatas[0].ignore_label_id,
            max_input_length=max(m.max_input_length for m in metadatas),
            max_output_length=max(m.max_output_length for m in metadatas),
            num_instances=sum(m.num_instances for m in metadatas),
            num_domains=sum(m.num_domains for m in metadatas),
            num_languages=max(m.num_languages for m in metadatas),
            average_complexity=sum(m.average_complexity * m.num_instances for m in metadatas) / sum(m.num_instances for m in metadatas)
        )
    
    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Add tokenizer attribute for compatibility
    if hasattr(datasets[0], 'tokenizer'):
        dataloader.dataset.tokenizer = datasets[0].tokenizer
    
    return dataloader, metadata


# Test functionality
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Test the dataset
    config = CodeGenerationDatasetConfig(
        dataset_path="data/swe-smith-1k",
        split="train",
        max_input_length=512,
        max_output_length=256
    )
    
    try:
        dataset = CodeGenerationDataset(config)
        print(f"Dataset loaded successfully: {len(dataset)} instances")
        
        # Test a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Input shape: {sample['inputs'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
        # Test metadata
        metadata = dataset.get_metadata()
        print(f"Metadata: {metadata}")
        
        # Test batch
        batch = dataset.get_sample_batch(2)
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch input shape: {batch['inputs'].shape}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()