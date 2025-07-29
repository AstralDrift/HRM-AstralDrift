#!/usr/bin/env python3
"""
Quick Dataset Builder for HRM-AstralDrift

This script helps you build training datasets quickly for different training phases.
Run with: python build_datasets.py [preset_name]

Available presets:
- micro: Tiny dataset for 5-minute verification
- small: Small dataset for 1-2 hour training
- medium: Medium dataset for 4-8 hour training  
- large: Large dataset for production training
"""

import argparse
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"📦 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_success(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed")
        print(f"Error: {e.stderr}")
        return False

def build_micro_datasets():
    """Build micro datasets for quick verification"""
    print_header("Building Micro Datasets (5-minute verification)")
    
    commands = [
        (
            ["python", "dataset/build_sudoku_dataset.py", 
             "--output-dir", "data/micro-test", 
             "--subsample-size", "10", 
             "--num-aug", "10"],
            "Building micro Sudoku dataset"
        )
    ]
    
    success = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            success = False
    
    if success:
        print_success("Micro datasets ready! You can now run:")
        print("  python pretrain.py data_path=data/micro-test epochs=50 global_batch_size=2")
    
    return success

def build_small_datasets():
    """Build small datasets for 1-2 hour training"""
    print_header("Building Small Datasets (1-2 hour training)")
    
    commands = [
        (
            ["python", "dataset/build_sudoku_dataset.py",
             "--output-dir", "data/small-sudoku",
             "--subsample-size", "100",
             "--num-aug", "100"],
            "Building small Sudoku dataset"
        )
    ]
    
    # Add code generation if LiveCodeBench is available
    if Path("LiveCodeBench").exists():
        commands.append((
            ["python", "dataset/build_livecodebench_dataset.py",
             "--output-dir", "data/livecodebench-small",
             "--max-problems", "50",
             "--languages", "python,javascript"],
            "Building small LiveCodeBench dataset"
        ))
    
    success = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            success = False
    
    if success:
        print_success("Small datasets ready! You can now run:")
        print("  python pretrain.py data_path=data/small-sudoku epochs=1000")
        if Path("data/livecodebench-small").exists():
            print("  python pretrain.py data_path=data/livecodebench-small epochs=2000")
    
    return success

def build_medium_datasets():
    """Build medium datasets for 4-8 hour training"""
    print_header("Building Medium Datasets (4-8 hour training)")
    
    commands = [
        (
            ["python", "dataset/build_sudoku_dataset.py",
             "--output-dir", "data/medium-sudoku", 
             "--subsample-size", "500",
             "--num-aug", "200"],
            "Building medium Sudoku dataset"
        )
    ]
    
    # Add code generation datasets if available
    if Path("LiveCodeBench").exists():
        commands.append((
            ["python", "dataset/build_livecodebench_dataset.py",
             "--output-dir", "data/livecodebench-medium",
             "--max-problems", "200",
             "--languages", "python,javascript,cpp"],
            "Building medium LiveCodeBench dataset"
        ))
    
    if Path("polyglot-benchmark").exists():
        commands.append((
            ["python", "dataset/polyglot_benchmark_extractor.py",
             "--output-dir", "data/polyglot-medium",
             "--max-problems-per-language", "50",
             "--languages", "python,javascript,go"],
            "Building medium Polyglot dataset"
        ))
    
    success = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            success = False
            
    if success:
        print_success("Medium datasets ready! You can now run:")
        print("  python pretrain.py data_path=data/medium-sudoku epochs=5000")
        if Path("data/livecodebench-medium").exists():
            print("  python pretrain.py data_path=data/livecodebench-medium epochs=5000 arch.enable_swe_search=true")
        if Path("data/polyglot-medium").exists():
            print("  python pretrain.py data_path=data/polyglot-medium epochs=8000 arch.enable_reverse_learning=true")
    
    return success

def build_large_datasets():
    """Build large datasets for production training"""
    print_header("Building Large Datasets (Production scale)")
    
    print("⚠️  Warning: Large datasets will take significant time and storage!")
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return False
    
    commands = [
        (
            ["python", "dataset/build_sudoku_dataset.py",
             "--output-dir", "data/large-sudoku",
             "--subsample-size", "2000", 
             "--num-aug", "500"],
            "Building large Sudoku dataset"
        )
    ]
    
    if Path("LiveCodeBench").exists():
        commands.append((
            ["python", "dataset/build_livecodebench_dataset.py",
             "--output-dir", "data/livecodebench-full",
             "--max-problems", "1000"],
            "Building full LiveCodeBench dataset"
        ))
    
    if Path("polyglot-benchmark").exists():
        commands.append((
            ["python", "dataset/polyglot_benchmark_extractor.py",
             "--output-dir", "data/polyglot-full",
             "--max-problems-per-language", "200"],
            "Building full Polyglot dataset"
        ))
    
    success = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            success = False
    
    if success:
        print_success("Large datasets ready! You can now run:")
        print("  OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/large-sudoku")
        print("  OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/livecodebench-full")
    
    return success

def build_all_datasets():
    """Build all dataset sizes"""
    print_header("Building All Datasets")
    
    builders = [
        ("micro", build_micro_datasets),
        ("small", build_small_datasets), 
        ("medium", build_medium_datasets),
        ("large", build_large_datasets)
    ]
    
    results = []
    for name, builder in builders:
        print(f"\n🔄 Building {name} datasets...")
        result = builder()
        results.append((name, result))
        
        if not result:
            print_error(f"Failed to build {name} datasets")
            break
    
    print_header("Dataset Building Summary")
    for name, result in results:
        if result:
            print_success(f"{name.capitalize()} datasets: Ready")
        else:
            print_error(f"{name.capitalize()} datasets: Failed")

def check_prerequisites():
    """Check if required files exist"""
    required_files = [
        "dataset/build_sudoku_dataset.py",
        "models/hrm/hrm_act_v1.py", 
        "pretrain.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_error("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Build training datasets for HRM-AstralDrift")
    parser.add_argument("preset", nargs="?", default="micro",
                       choices=["micro", "small", "medium", "large", "all"],
                       help="Dataset preset to build (default: micro)")
    parser.add_argument("--list", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available dataset presets:")
        print("  micro:  Tiny datasets for 5-minute verification")
        print("  small:  Small datasets for 1-2 hour training")
        print("  medium: Medium datasets for 4-8 hour training")
        print("  large:  Large datasets for production training")
        print("  all:    Build all dataset sizes sequentially")
        return 0
    
    print("🚀 HRM-AstralDrift Dataset Builder")
    
    # Check prerequisites
    if not check_prerequisites():
        print_error("Prerequisites not met. Make sure you're in the HRM-AstralDrift directory.")
        return 1
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Build requested datasets
    builders = {
        "micro": build_micro_datasets,
        "small": build_small_datasets,
        "medium": build_medium_datasets, 
        "large": build_large_datasets,
        "all": build_all_datasets
    }
    
    builder = builders[args.preset]
    success = builder()
    
    if success:
        print_header("Next Steps")
        print("✅ Datasets built successfully!")
        print("\nRecommended commands:")
        print("  python verify_system.py          # Verify everything works")
        print("  python pretrain.py --help       # See training options")
        print("  python pretrain.py data_path=data/micro-test epochs=50  # Quick test")
        return 0
    else:
        print_error("Dataset building failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())