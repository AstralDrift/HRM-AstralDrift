#!/usr/bin/env python3
"""
HRM-AstralDrift System Verification Script

This script performs a complete system check to ensure everything is working
before you start training. Run this first to catch any issues early.

Usage: python verify_system.py
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import torch

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def check_python_version():
    """Check Python version"""
    print_header("Python Environment Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print_success("Python version is compatible (3.10+)")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} is too old. Need 3.10+")
        return False

def check_dependencies():
    """Check required dependencies"""
    print_header("Dependency Check")
    
    required_packages = [
        'torch',
        'numpy', 
        'pydantic',
        'omegaconf',
        'hydra',
        'tqdm',
        'wandb',
        'einops'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_pytorch_cuda():
    """Check PyTorch and CUDA setup"""
    print_header("PyTorch & CUDA Check")
    
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA is available")
            print_success(f"CUDA version: {torch.version.cuda}")
            print_success(f"Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_success(f"  GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            print_warning("CUDA is not available - will use CPU (slower)")
            
        return True
    except Exception as e:
        print_error(f"PyTorch check failed: {e}")
        return False

def check_project_structure():
    """Check project file structure"""
    print_header("Project Structure Check")
    
    required_files = [
        'models/hrm/hrm_act_v1.py',
        'models/hrm/swe_search_integration.py', 
        'models/hrm/reverse_learning.py',
        'models/losses.py',
        'train_hrm_optimized.py',
        'evaluate.py',
        'code_generation_dataset.py',
        'test_swe_search_integration.py',
        'test_reverse_learning_integration.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} is missing")
            missing_files.append(file_path)
    
    if missing_files:
        print_error(f"Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_framework_imports():
    """Check if HRM frameworks can be imported"""
    print_header("Framework Import Check")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1
        print_success("HRM architecture imports successful")
        
        from models.hrm.swe_search_integration import SWESearchController
        print_success("SWE-Search framework imports successful")
        
        from models.hrm.reverse_learning import ReverseLearningModule
        print_success("Reverse Learning framework imports successful")
        
        from models.losses import ACTSWESearchLossHead
        print_success("Enhanced loss function imports successful")
        
        return True
        
    except Exception as e:
        print_error(f"Framework import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_creation():
    """Test basic model creation"""
    print_header("Model Creation Test")
    
    try:
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1
        
        # Create test config
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=2,
            seq_len=32,
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=1,
            vocab_size=100,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=64,
            expansion=2.0,
            num_heads=2,
            pos_encodings="learned",
            halt_max_steps=2,
            halt_exploration_prob=0.1,
            enable_swe_search=True,
            enable_reverse_learning=True
        )
        
        # Create model
        model = HierarchicalReasoningModel_ACTV1(config.__dict__)
        
        # Move model to correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model = model.to(device)
        print_success("Base model creation successful")
        
        # Test forward pass
        # Use same device as model
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        batch = {
            'inputs': torch.randint(0, 100, (2, 32), device=device),
            'labels': torch.randint(0, 100, (2, 32), device=device),
            'puzzle_identifiers': torch.zeros((2,), dtype=torch.long, device=device)
        }
        
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        print_success("Model forward pass successful")
        print_success(f"Output shape: {outputs['logits'].shape}")
        
        # Check advanced features
        if hasattr(model.inner, 'swe_search_controller'):
            print_success("SWE-Search integration active")
        
        if hasattr(model.inner, 'reverse_learning'):
            print_success("Reverse Learning integration active")
        
        return True
        
    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_integration_tests():
    """Run the integration test scripts"""
    print_header("Integration Tests")
    
    test_files = [
        'test_swe_search_integration.py',
        'test_reverse_learning_integration.py'
    ]
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print_error(f"{test_file} not found")
            continue
            
        try:
            print(f"\nRunning {test_file}...")
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print_success(f"{test_file} passed")
                # Count passed tests
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Tests passed:' in line:
                        print_success(f"  {line.strip()}")
                        break
            else:
                print_error(f"{test_file} failed")
                print(f"Error output: {result.stderr[-500:]}")  # Last 500 chars
                
        except subprocess.TimeoutExpired:
            print_error(f"{test_file} timed out")
        except Exception as e:
            print_error(f"Error running {test_file}: {e}")

def check_dataset_builders():
    """Check if dataset builders are working"""
    print_header("Dataset Builder Check")
    
    builders = [
        'dataset/build_sudoku_dataset.py',
        'dataset/build_livecodebench_dataset.py',
        'dataset/polyglot_benchmark_extractor.py'
    ]
    
    all_exist = True
    for builder in builders:
        if Path(builder).exists():
            print_success(f"{builder} exists")
        else:
            print_warning(f"{builder} not found")
            all_exist = False
    
    return all_exist

def suggest_next_steps(all_checks_passed):
    """Suggest next steps based on results"""
    print_header("Next Steps")
    
    if all_checks_passed:
        print_success("🎉 All system checks passed! You're ready to train.")
        print("\nRecommended next steps:")
        print("1. Follow DEPLOYMENT_GUIDE.md Phase 1 (15-minute verification)")
        print("2. Run: python dataset/build_sudoku_dataset.py --output-dir data/micro-test --subsample-size 10 --num-aug 10")
        print("3. Run: python pretrain.py data_path=data/micro-test epochs=50 global_batch_size=2")
        print("4. Check W&B dashboard for training progress")
        
    else:
        print_error("❌ Some system checks failed.")
        print("\nPlease fix the issues above before proceeding.")
        print("Common solutions:")
        print("- Install missing packages: pip install torch wandb einops pydantic omegaconf hydra-core tqdm")
        print("- Check Python version: python --version (need 3.10+)")
        print("- Verify you're in the right directory: ls -la (should see models/, dataset/, etc.)")

def main():
    """Run all system checks"""
    print("🚀 HRM-AstralDrift System Verification")
    print("This will check if your system is ready for training...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("PyTorch & CUDA", check_pytorch_cuda),
        ("Project Structure", check_project_structure),
        ("Framework Imports", check_framework_imports),
        ("Model Creation", check_model_creation),
        ("Dataset Builders", check_dataset_builders)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print_error(f"{check_name} check crashed: {e}")
            results.append(False)
    
    # Run integration tests separately (they take longer)
    print_warning("Running integration tests (this may take a few minutes)...")
    run_quick_integration_tests()
    
    # Summary
    passed_checks = sum(results)
    total_checks = len(results)
    all_passed = passed_checks == total_checks
    
    print_header("Verification Summary")
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if all_passed:
        print_success("🎉 System verification completed successfully!")
    else:
        print_error(f"❌ {total_checks - passed_checks} checks failed")
    
    suggest_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())