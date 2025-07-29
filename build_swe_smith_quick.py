#!/usr/bin/env python3
"""
Quick SWE-smith Dataset Builder

Build a training dataset from the existing SWE-smith test examples.
This uses the real GitHub issues and trajectories we already have.
"""

import json
import os
from pathlib import Path
import shutil
from typing import Dict, List
import argparse

def print_info(msg):
    print(f"ℹ️  {msg}")

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def extract_issue_data(logs_dir: Path) -> List[Dict]:
    """Extract training examples from SWE-smith test logs"""
    examples = []
    
    # Get trajectory files
    traj_dir = logs_dir / "trajectories"
    eval_dir = logs_dir / "run_evaluation"
    
    if not traj_dir.exists():
        print_error(f"Trajectories directory not found: {traj_dir}")
        return examples
    
    for issue_dir in traj_dir.iterdir():
        if not issue_dir.is_dir():
            continue
            
        issue_name = issue_dir.name
        traj_file = issue_dir / f"{issue_name}.traj"
        
        if not traj_file.exists():
            continue
            
        # Read trajectory
        try:
            with open(traj_file, 'r') as f:
                trajectory = f.read()
        except Exception as e:
            print_error(f"Error reading {traj_file}: {e}")
            continue
        
        # Get evaluation results if available
        eval_results = None
        eval_issue_dir = eval_dir / issue_name
        if eval_issue_dir.exists():
            report_file = eval_issue_dir / "report.json"
            if report_file.exists():
                try:
                    with open(report_file, 'r') as f:
                        eval_results = json.load(f)
                except Exception as e:
                    print_error(f"Error reading {report_file}: {e}")
        
        # Extract issue type and repo
        parts = issue_name.split('.')
        if len(parts) >= 2:
            repo = parts[0].replace('__', '/')
            issue_id = parts[1]
        else:
            repo = "unknown"
            issue_id = issue_name
        
        example = {
            'issue_name': issue_name,
            'repo': repo,
            'issue_id': issue_id,
            'trajectory': trajectory,
            'eval_results': eval_results,
            'type': 'github_issue_fix'
        }
        
        examples.append(example)
        print_success(f"Extracted: {issue_name}")
    
    return examples

def extract_code_files(files_dir: Path) -> List[Dict]:
    """Extract code examples from multi-language files"""
    examples = []
    
    if not files_dir.exists():
        print_error(f"Files directory not found: {files_dir}")
        return examples
    
    for lang_dir in files_dir.iterdir():
        if not lang_dir.is_dir():
            continue
            
        language = lang_dir.name
        
        for code_file in lang_dir.rglob("*"):
            if code_file.is_file() and code_file.suffix in ['.py', '.js', '.go', '.java', '.rs', '.c', '.cs', '.php', '.rb']:
                try:
                    with open(code_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    example = {
                        'filename': code_file.name,
                        'language': language,
                        'content': content,
                        'type': 'code_example'
                    }
                    
                    examples.append(example)
                    print_success(f"Extracted code: {language}/{code_file.name}")
                    
                except Exception as e:
                    print_error(f"Error reading {code_file}: {e}")
    
    return examples

def create_hrm_dataset(examples: List[Dict], output_dir: Path):
    """Create HRM-compatible dataset from examples"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train/test splits
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Split examples (80% train, 20% test)
    split_idx = int(len(examples) * 0.8)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    # Create dataset files
    train_data = []
    test_data = []
    
    for i, example in enumerate(train_examples):
        if example['type'] == 'github_issue_fix':
            # Format as instruction-following task
            inputs = f"Fix the following GitHub issue:\nRepo: {example['repo']}\nIssue ID: {example['issue_id']}\n\nRequired actions:"
            labels = example['trajectory'][:512]  # Truncate long trajectories
        else:
            # Code example
            inputs = f"Analyze and explain this {example['language']} code:\n{example['filename']}\n\nCode:"
            labels = example['content'][:512]
        
        train_data.append({
            'inputs': inputs,
            'labels': labels,
            'puzzle_identifier': 0  # HRM requires this
        })
    
    for i, example in enumerate(test_examples):
        if example['type'] == 'github_issue_fix':
            inputs = f"Fix the following GitHub issue:\nRepo: {example['repo']}\nIssue ID: {example['issue_id']}\n\nRequired actions:"
            labels = example['trajectory'][:512]
        else:
            inputs = f"Analyze and explain this {example['language']} code:\n{example['filename']}\n\nCode:"
            labels = example['content'][:512]
        
        test_data.append({
            'inputs': inputs,
            'labels': labels,
            'puzzle_identifier': 0
        })
    
    # Save dataset files
    with open(train_dir / "data.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_dir / "data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create metadata
    metadata = {
        'vocab_size': 50000,  # Standard vocab size
        'num_puzzle_identifiers': 1,
        'groups': {
            'train': {'size': len(train_data)},
            'test': {'size': len(test_data)}
        }
    }
    
    with open(output_dir / "identifiers.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_success(f"Created HRM dataset: {len(train_data)} train, {len(test_data)} test examples")
    return True

def main():
    parser = argparse.ArgumentParser(description="Build SWE-smith dataset quickly")
    parser.add_argument('--output-dir', default='data/swe-smith-small',
                       help='Output directory for dataset')
    args = parser.parse_args()
    
    print("🚀 SWE-smith Quick Dataset Builder")
    
    # Paths
    swe_smith_dir = Path("SWE-smith")
    logs_dir = swe_smith_dir / "tests" / "test_logs"
    files_dir = logs_dir / "files"
    output_dir = Path(args.output_dir)
    
    if not logs_dir.exists():
        print_error(f"SWE-smith test logs not found: {logs_dir}")
        return 1
    
    # Extract examples
    print_info("Extracting GitHub issue examples...")
    issue_examples = extract_issue_data(logs_dir)
    
    print_info("Extracting code file examples...")
    code_examples = extract_code_files(files_dir)
    
    all_examples = issue_examples + code_examples
    
    if not all_examples:
        print_error("No examples found!")
        return 1
    
    print_success(f"Found {len(all_examples)} total examples ({len(issue_examples)} issues, {len(code_examples)} code files)")
    
    # Create dataset
    print_info("Creating HRM-compatible dataset...")
    success = create_hrm_dataset(all_examples, output_dir)
    
    if success:
        print_success(f"✅ SWE-smith dataset created at: {output_dir}")
        print_info("Ready for training!")
        return 0
    else:
        print_error("Failed to create dataset")
        return 1

if __name__ == '__main__':
    exit(main())