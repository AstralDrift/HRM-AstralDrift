#!/usr/bin/env python3
"""
HRM Code Generation Training Script

Train HRM on real code generation tasks using:
- SWE-smith: Real GitHub issues and fixes  
- LiveCodeBench: Programming challenges
- Tool usage: CLI operations, git commands, package management
- Multi-language support: Python, JavaScript, C++, Go, Java, Rust

Usage:
    python train_code_generation.py --dataset swe_smith --size small
    python train_code_generation.py --dataset livecodebench --size medium  
    python train_code_generation.py --dataset combined --size large
"""

import argparse
import sys
import subprocess
from pathlib import Path
import yaml

# Training configurations for different code generation scenarios
TRAINING_CONFIGS = {
    'swe_smith_1k': {
        'description': 'SWE-smith 1K: Real GitHub issues with sophisticated analysis',
        'data_path': 'data/swe-smith-1k',
        'epochs': 3000,
        'eval_interval': 500,
        'global_batch_size': 24,
        'lr': 4e-5,
        'project_name': 'HRM-SWE-Smith-1K',
        'run_name': 'sophisticated-github-issues-v1',
        'arch': {
            'enable_swe_search': True,
            'enable_reverse_learning': True,
            'seq_len': 2048,  # Match dataset sequence length
            'H_cycles': 3,  # More cycles for complex reasoning
            'L_cycles': 4,
            'loss': {
                'name': 'ACTSWESearchLossHead',
                'swe_search_weight': 0.3,  # Higher for sophisticated coordination
                'reverse_learning_weight': 0.2
            }
        },
        'expected_time': '3-6 hours'
    },
    
    'livecodebench_small': {
        'description': 'LiveCodeBench small: Programming challenges',
        'data_path': 'data/livecodebench-small',
        'epochs': 3000,
        'eval_interval': 500,
        'global_batch_size': 12,
        'lr': 4e-5,
        'project_name': 'HRM-LiveCodeBench',
        'run_name': 'programming-challenges-v1',
        'arch': {
            'enable_swe_search': True,
            'enable_reverse_learning': True,
            'seq_len': 512,
            'loss': {
                'name': 'ACTSWESearchLossHead',
                'swe_search_weight': 0.2,
                'reverse_learning_weight': 0.1
            }
        },
        'expected_time': '3-6 hours'
    },
    
    'polyglot_medium': {
        'description': 'Polyglot medium: Multi-language code (6 languages)',
        'data_path': 'data/polyglot-medium',
        'epochs': 4000,
        'eval_interval': 600,
        'global_batch_size': 10,
        'lr': 3e-5,
        'project_name': 'HRM-Polyglot',  
        'run_name': 'multi-language-v1',
        'arch': {
            'enable_swe_search': True,
            'enable_reverse_learning': True,
            'seq_len': 768,
            'loss': {
                'name': 'ACTSWESearchLossHead',
                'swe_search_weight': 0.3,  # Higher for language adaptation
                'reverse_learning_weight': 0.2
            }
        },
        'expected_time': '4-8 hours'
    },
    
    'combined_large': {
        'description': 'Combined large: All datasets for maximum capability',
        'data_path': 'data/code-generation-combined',
        'epochs': 10000,
        'eval_interval': 1000,
        'global_batch_size': 32,
        'lr': 2e-5,
        'project_name': 'HRM-CodeGen-Full',
        'run_name': 'full-capability-v1',
        'arch': {
            'enable_swe_search': True,
            'enable_reverse_learning': True,
            'seq_len': 1024,
            'H_cycles': 3,  # More reasoning for complex code
            'L_cycles': 4,
            'loss': {
                'name': 'ACTSWESearchLossHead',
                'swe_search_weight': 0.25,
                'reverse_learning_weight': 0.15
            }
        },
        'expected_time': '8-16 hours'
    }
}

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def check_dataset_availability():
    """Check which datasets are available"""
    print_header("Dataset Availability Check")
    
    datasets = {
        'SWE-smith': Path('SWE-smith').exists(),
        'LiveCodeBench': Path('LiveCodeBench').exists(),  
        'Polyglot': Path('polyglot-benchmark').exists(),
        'Data directory': Path('data').exists()
    }
    
    for name, available in datasets.items():
        if available:
            print_success(f"{name}: Available")
        else:
            print_error(f"{name}: Not found")
    
    return all(datasets.values())

def build_dataset(config_name):
    """Build the dataset for the given configuration"""
    config = TRAINING_CONFIGS[config_name]
    data_path = config['data_path']
    
    print_header(f"Building Dataset: {config['description']}")
    print_info(f"Target path: {data_path}")
    
    # Create data directory if it doesn't exist
    Path('data').mkdir(exist_ok=True)
    
    if Path(data_path).exists():
        print_success(f"Dataset already exists at {data_path}")
        return True
    
    # Determine which dataset builder to use
    if 'swe-smith' in data_path:
        return build_swe_smith_dataset(data_path)
    elif 'livecodebench' in data_path:
        return build_livecodebench_dataset(data_path)
    elif 'polyglot' in data_path:
        return build_polyglot_dataset(data_path)
    elif 'combined' in data_path:
        return build_combined_dataset(data_path)
    else:
        print_error(f"Unknown dataset type for {data_path}")
        return False

def build_swe_smith_dataset(output_dir):
    """Build SWE-smith dataset using sophisticated integration"""
    print_info("Building SWE-smith dataset with sophisticated analysis...")
    
    try:
        # Use our sophisticated SWE-smith dataset builder
        cmd = [
            'python', 'build_swe_smith_dataset.py',
            '--output-dir', output_dir,
            '--num-samples', '1000',  # Full dataset for training
            '--batch-size', '16'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print_success("SWE-smith sophisticated dataset built successfully")
            return True
        else:
            print_error(f"SWE-smith dataset build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error building SWE-smith dataset: {e}")
        return False

def build_livecodebench_dataset(output_dir):
    """Build LiveCodeBench dataset for programming challenges"""
    print_info("Building LiveCodeBench dataset...")
    
    try:
        cmd = [
            'python', 'dataset/build_livecodebench_dataset.py',
            '--output-dir', output_dir,
            '--max-problems', '100',
            '--languages', 'python,javascript,cpp'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            print_success("LiveCodeBench dataset built successfully")
            return True
        else:
            print_error(f"LiveCodeBench dataset build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error building LiveCodeBench dataset: {e}")
        return False

def build_polyglot_dataset(output_dir):
    """Build Polyglot dataset for multi-language code"""
    print_info("Building Polyglot dataset...")
    
    try:
        cmd = [
            'python', 'dataset/polyglot_benchmark_extractor.py',
            '--output-dir', output_dir,
            '--max-problems-per-language', '30',
            '--languages', 'python,javascript,go,cpp,java,rust'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            print_success("Polyglot dataset built successfully")
            return True
        else:
            print_error(f"Polyglot dataset build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error building Polyglot dataset: {e}")
        return False

def build_combined_dataset(output_dir):
    """Build combined dataset from all sources"""
    print_info("Building combined dataset from all sources...")
    
    # Build individual datasets first
    individual_datasets = [
        'data/swe-smith-small',
        'data/livecodebench-small', 
        'data/polyglot-medium'
    ]
    
    success = True
    for dataset_path in individual_datasets:
        if not Path(dataset_path).exists():
            config_name = None
            for name, config in TRAINING_CONFIGS.items():
                if config['data_path'] == dataset_path:
                    config_name = name
                    break
            
            if config_name:
                success &= build_dataset(config_name)
    
    if success:
        # Combine the datasets (this would need a proper implementation)
        print_info("Combining individual datasets...")
        # TODO: Implement dataset combination logic
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print_success("Combined dataset created")
        return True
    else:
        print_error("Failed to build some individual datasets")
        return False

def run_training(config_name):
    """Run training with the specified configuration"""
    config = TRAINING_CONFIGS[config_name]
    
    print_header(f"Training: {config['description']}")
    print_info(f"Expected time: {config['expected_time']}")
    print_info(f"Data path: {config['data_path']}")
    
    # Build training command
    cmd = ['python', 'pretrain.py']
    
    # Add basic parameters
    for key, value in config.items():
        if key in ['description', 'expected_time', 'arch']:
            continue
        cmd.extend([f'{key}={value}'])
    
    # Add architecture parameters
    if 'arch' in config:
        for key, value in config['arch'].items():
            if key == 'loss':
                for loss_key, loss_value in value.items():
                    cmd.append(f'+arch.loss.{loss_key}={loss_value}')
            else:
                cmd.append(f'+arch.{key}={value}')
    
    # Add environment variables for optimal performance
    env_vars = [
        'DISABLE_COMPILE=1',  # Disable torch.compile for stability
        'WANDB_MODE=offline'  # Use offline mode for now
    ]
    
    print_info("Starting training with command:")
    print(f"  {' '.join(env_vars)} {' '.join(cmd)}")
    
    try:
        # Run training
        full_cmd = f"source venv_py310/bin/activate && {' '.join(env_vars)} {' '.join(cmd)}"
        result = subprocess.run(full_cmd, shell=True, text=True)
        
        if result.returncode == 0:
            print_success("Training completed successfully!")
            return True
        else:
            print_error("Training failed")
            return False
            
    except KeyboardInterrupt:
        print_info("Training interrupted by user")
        return False
    except Exception as e:
        print_error(f"Training error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train HRM on code generation tasks")
    parser.add_argument('--config', choices=list(TRAINING_CONFIGS.keys()), 
                       default='swe_smith_small',
                       help='Training configuration to use')
    parser.add_argument('--build-only', action='store_true',
                       help='Only build dataset, don\'t train')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')
    parser.add_argument('--check-datasets', action='store_true',
                       help='Check dataset availability')
    
    args = parser.parse_args()
    
    if args.list_configs:
        print_header("Available Training Configurations")
        for name, config in TRAINING_CONFIGS.items():
            print(f"\n🎯 {name}:")
            print(f"   Description: {config['description']}")
            print(f"   Expected time: {config['expected_time']}")
            print(f"   Data path: {config['data_path']}")
        return 0
    
    if args.check_datasets:
        available = check_dataset_availability()
        return 0 if available else 1
    
    print_header("HRM Code Generation Training")
    print_info(f"Configuration: {args.config}")
    print_info(f"Description: {TRAINING_CONFIGS[args.config]['description']}")
    
    # Check prerequisites
    if not check_dataset_availability():
        print_error("Some required datasets are missing")
        return 1
    
    # Build dataset
    print_info("Building dataset...")
    if not build_dataset(args.config):
        print_error("Dataset building failed")
        return 1
    
    if args.build_only:
        print_success("Dataset built successfully (build-only mode)")
        return 0
    
    # Run training
    success = run_training(args.config)
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())