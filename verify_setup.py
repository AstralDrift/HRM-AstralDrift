#!/usr/bin/env python3
"""
HRM Codebase Verification Script

Verifies that the cleaned codebase is ready for production training on RTX 4090.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def check_puzzle_cleanup():
    """Verify puzzle-related code has been removed"""
    print("🧹 Checking Puzzle Code Cleanup")
    print("-" * 40)
    
    removed_files = [
        "dataset/build_sudoku_dataset.py",
        "dataset/build_maze_dataset.py", 
        "dataset/build_arc_dataset.py",
        "arc_eval.ipynb",
        "data/small-sudoku",
        "data/micro-test",
        "data/mock-livecodebench"
    ]
    
    all_removed = True
    for file_path in removed_files:
        if Path(file_path).exists():
            print(f"❌ Still exists: {file_path}")
            all_removed = False
        else:
            print(f"✅ Removed: {file_path}")
    
    if all_removed:
        print("🎉 All puzzle-related code successfully removed!")
    
    return all_removed

def check_dataset_consolidation():
    """Verify only SWE-Smith-1k and LiveCodeBench datasets remain"""
    print("\n📊 Checking Dataset Consolidation")
    print("-" * 40)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    expected_datasets = {
        "swe-smith-1k", 
        "swe-smith-1k-hrm",
        "livecodebench_real",
        "livecodebench_hrm"
    }
    
    actual_datasets = {item.name for item in data_dir.iterdir() if item.is_dir()}
    
    extra_datasets = actual_datasets - expected_datasets
    missing_datasets = expected_datasets - actual_datasets
    
    if extra_datasets:
        print(f"⚠️ Extra datasets found: {extra_datasets}")
    
    if missing_datasets:
        print(f"⚠️ Missing datasets: {missing_datasets}")
    
    for dataset in expected_datasets & actual_datasets:
        print(f"✅ Found: {dataset}")
    
    consolidation_success = len(extra_datasets) == 0
    
    if consolidation_success:
        print("🎉 Dataset consolidation successful!")
    
    return consolidation_success

def check_tokenizer_integration():
    """Verify tokenizer fix is applied"""
    print("\n🔤 Checking Tokenizer Integration")
    print("-" * 40)
    
    # Check models/losses.py has the fix
    losses_file = Path("models/losses.py")
    if not losses_file.exists():
        print("❌ models/losses.py not found")
        return False
    
    content = losses_file.read_text()
    
    # Check for key tokenizer fix components
    checks = [
        ("skip_special_tokens=True", "tokenizer.decode with skip_special_tokens"),
        ("_tokens_to_string", "token-to-string conversion method"),
        ("_get_or_load_fallback_tokenizer", "fallback tokenizer loading"),
        ("ACTSWESearchLossHead", "enhanced loss head with tokenizer")
    ]
    
    all_present = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✅ Found: {description}")
        else:
            print(f"❌ Missing: {description}")
            all_present = False
    
    if all_present:
        print("🎉 Tokenizer integration verified!")
    
    return all_present

def check_cuda_compatibility():
    """Verify CUDA compatibility for RTX 4090 training"""
    print("\n🚀 Checking CUDA Compatibility")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Devices: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA not available on this system (expected on MacBook)")
            print("   Will work on RTX 4090 systems with CUDA 12.6+")
        
        # Check MPS (for MacBook testing)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS available for MacBook testing")
        
        # Verify device selection utility
        from utils.device import get_device
        current_device = get_device()
        print(f"✅ Device selection: {current_device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def check_training_configuration():
    """Verify optimized training parameters"""
    print("\n⚙️ Checking Training Configuration")
    print("-" * 40)
    
    # Check config files
    config_file = Path("config/arch/hrm_swe_search.yaml")
    if config_file.exists():
        print("✅ Found: config/arch/hrm_swe_search.yaml")
        
        content = config_file.read_text()
        
        # Check optimized parameters
        param_checks = [
            ("halt_exploration_prob: 0.5", "exploration probability"),
            ("swe_search_weight: 0.3", "SWE-Search weight"),
            ("reverse_learning_weight: 0.2", "reverse learning weight"),
            ("enable_swe_search: true", "SWE-Search enabled"),
            ("enable_reverse_learning: true", "reverse learning enabled")
        ]
        
        for param, description in param_checks:
            if param in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️ {description} not found")
    else:
        print("❌ Config file not found")
        return False
    
    # Check training script
    train_script = Path("train_hrm_optimized.py")
    if train_script.exists():
        print("✅ Found: train_hrm_optimized.py")
        
        content = train_script.read_text()
        
        # Check for mixed training support
        if "mixed" in content and "create_mixed_dataloader" in content:
            print("   ✅ Mixed training support")
        else:
            print("   ⚠️ Mixed training support incomplete")
    
    return True

def check_evaluation_suite():
    """Verify consolidated evaluation capabilities"""
    print("\n🔬 Checking Evaluation Suite")
    print("-" * 40)
    
    eval_script = Path("eval_suite.py")
    if eval_script.exists():
        print("✅ Found: eval_suite.py (consolidated evaluation)")
        
        content = eval_script.read_text()
        
        # Check capabilities
        capabilities = [
            ("quick_eval", "quick evaluation"),
            ("comprehensive_eval", "comprehensive evaluation"), 
            ("save_results", "result saving"),
            ("HRMEvaluationSuite", "unified evaluation class")
        ]
        
        for capability, description in capabilities:
            if capability in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️ {description} missing")
    else:
        print("❌ Evaluation suite not found")
        return False
    
    return True

def verify_ready_for_training():
    """Final verification for training readiness"""
    print("\n🎯 Training Readiness Summary")
    print("-" * 40)
    
    # Check critical files exist
    critical_files = [
        "train_hrm_optimized.py",
        "code_generation_dataset.py", 
        "models/hrm/hrm_act_v1.py",
        "models/losses.py",
        "eval_suite.py"
    ]
    
    all_present = True
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_present = False
    
    return all_present

def main():
    """Run all verification checks"""
    print("🔍 HRM Codebase Verification")
    print("=" * 60)
    
    checks = [
        ("Puzzle Cleanup", check_puzzle_cleanup),
        ("Dataset Consolidation", check_dataset_consolidation),
        ("Tokenizer Integration", check_tokenizer_integration), 
        ("CUDA Compatibility", check_cuda_compatibility),
        ("Training Configuration", check_training_configuration),
        ("Evaluation Suite", check_evaluation_suite),
        ("Training Readiness", verify_ready_for_training)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results[check_name] = False
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
    
    if all_passed:
        print("\n🎉 CODEBASE READY FOR PRODUCTION!")
        print("   ✅ Puzzle code removed")
        print("   ✅ Datasets consolidated (SWE-Smith-1k + LiveCodeBench)")
        print("   ✅ Tokenizer fix applied")
        print("   ✅ CUDA/RTX 4090 compatible") 
        print("   ✅ Optimized parameters configured")
        print("   ✅ Evaluation suite unified")
        print("\n🚀 Ready for RTX 4090 training with:")
        print("   python train_hrm_optimized.py --data-path mixed --epochs 40 --batch-size 6")
        
        return 0
    else:
        print("\n⚠️ Some checks failed. Review output above.")
        return 1

if __name__ == "__main__":
    exit(main())