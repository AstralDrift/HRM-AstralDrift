#!/usr/bin/env python3
"""
Convert SWE-smith dataset to standard HRM puzzle dataset format

This converts our sophisticated SWE-smith instances to the format expected by PuzzleDataset.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def convert_instances_to_hrm_format(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert SWE-smith instances to HRM format"""
    
    hrm_instances = []
    
    for instance in instances:
        # Extract the input and target text
        input_text = instance['input_text']
        target_text = instance['target_text']
        
        # Create a simple tokenization (in real use, we'd use a proper tokenizer)
        # For now, split on whitespace and assign numeric IDs
        input_tokens = input_text.split()
        target_tokens = target_text.split()
        
        # Create a simple vocabulary mapping (in practice, this would be more sophisticated)
        all_tokens = set(input_tokens + target_tokens)
        vocab = {token: i+1 for i, token in enumerate(sorted(all_tokens))}  # 0 reserved for padding
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = len(vocab)
        
        # Convert to token IDs
        input_ids = [vocab.get(token, vocab['<UNK>']) for token in input_tokens]
        target_ids = [vocab.get(token, vocab['<UNK>']) for token in target_tokens]
        
        # Truncate or pad to reasonable lengths
        max_input_len = 1024
        max_target_len = 512
        
        input_ids = input_ids[:max_input_len]
        target_ids = target_ids[:max_target_len]
        
        # Pad to consistent length
        input_ids.extend([0] * (max_input_len - len(input_ids)))
        target_ids.extend([0] * (max_target_len - len(target_ids)))
        
        hrm_instance = {
            'inputs': input_ids,
            'labels': target_ids,
            'puzzle_identifier': instance['puzzle_id']
        }
        
        hrm_instances.append(hrm_instance)
    
    return hrm_instances, len(vocab)


def create_hrm_dataset_structure(input_dir: str, output_dir: str):
    """Convert SWE-smith dataset to HRM format"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load the instances
    instances_file = input_path / 'instances.json'
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    
    print(f"Converting {len(instances)} instances...")
    
    # Convert to HRM format
    hrm_instances, vocab_size = convert_instances_to_hrm_format(instances)
    
    # Split into train/test (80/20)
    split_idx = int(len(hrm_instances) * 0.8)
    train_instances = hrm_instances[:split_idx]
    test_instances = hrm_instances[split_idx:]
    
    print(f"Train: {len(train_instances)}, Test: {len(test_instances)}")
    
    # Create directory structure
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        "pad_id": 0,
        "ignore_label_id": -100,
        "blank_identifier_id": 0,
        "vocab_size": vocab_size,
        "seq_len": 1536,  # 1024 input + 512 output
        "num_puzzle_identifiers": len(set(inst['puzzle_identifier'] for inst in hrm_instances)),
        "total_groups": len(hrm_instances),
        "mean_puzzle_examples": 1.0,
        "sets": ["train", "test"]
    }
    
    # Save train dataset
    with open(train_dir / 'dataset.json', 'w') as f:
        json.dump(metadata, f)
    
    with open(train_dir / 'data.json', 'w') as f:
        json.dump(train_instances, f)
    
    # Save test dataset  
    with open(test_dir / 'dataset.json', 'w') as f:
        json.dump(metadata, f)
    
    with open(test_dir / 'data.json', 'w') as f:
        json.dump(test_instances, f)
    
    # Save identifiers file
    with open(output_path / 'identifiers.json', 'w') as f:
        json.dump({
            'vocab_size': vocab_size,
            'num_puzzle_identifiers': metadata['num_puzzle_identifiers'],
            'groups': {
                'train': {'size': len(train_instances)},
                'test': {'size': len(test_instances)}
            }
        }, f, indent=2)
    
    print(f"✅ Converted dataset saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SWE-smith dataset to HRM format")
    parser.add_argument('--input-dir', default='data/swe-smith-1k', help='Input SWE-smith dataset directory')
    parser.add_argument('--output-dir', default='data/swe-smith-1k-hrm', help='Output HRM dataset directory')
    
    args = parser.parse_args()
    
    print(f"🔄 Converting {args.input_dir} to HRM format...")
    success = create_hrm_dataset_structure(args.input_dir, args.output_dir)
    
    if success:
        print("✅ Conversion completed successfully!")
        print(f"Use data_path={args.output_dir} in your training config")
        return 0
    else:
        print("❌ Conversion failed")
        return 1


if __name__ == "__main__":
    exit(main())