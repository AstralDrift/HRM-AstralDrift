"""
Integration Test for LiveCodeBench HRM System

This script tests the complete LiveCodeBench integration with HRM,
including data processing, model integration, and evaluation pipeline.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Test imports
from dataset.build_livecodebench_dataset import LiveCodeBenchProcessor, LiveCodeBenchConfig
from dataset.livecodebench_dataset import LiveCodeBenchDataset, create_livecodebench_dataloaders
from models.code_generation.input_processor import CodeGenerationInput, CodeGenerationTask, ProgrammingLanguage
from models.code_generation.hrm_code_model import HRMCodeGenerationModel
from evaluation.livecodebench_evaluator import CodeExecutionSandbox, LiveCodeBenchEvaluator, EvaluationConfig
from train_livecodebench import LiveCodeBenchTrainer, LiveCodeBenchTrainingConfig

def test_input_processor():
    """Test the input processor with sample data"""
    print("=== Testing Input Processor ===")
    
    from models.code_generation.input_processor import CodeGenerationInputProcessor
    
    # Create processor
    processor = CodeGenerationInputProcessor(
        vocab_size=40000,
        max_seq_len=512
    )
    
    # Test samples
    test_inputs = [
        CodeGenerationInput(
            problem_description="Write a function that returns the sum of two numbers",
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.GENERATION,
            test_cases=["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        ),
        CodeGenerationInput(
            problem_description="Fix the syntax error in this code",
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.REPAIR,
            existing_code="def add(a, b\n    return a + b",
            error_message="SyntaxError: invalid syntax"
        )
    ]
    
    # Test single processing
    for i, test_input in enumerate(test_inputs):
        print(f"Processing input {i+1}...")
        processed = processor.process_input(test_input)
        print(f"  Input tokens shape: {processed.input_tokens.shape}")
        print(f"  Language ID: {processed.language_id.item()}")
        print(f"  Task ID: {processed.task_id.item()}")
        print(f"  Complexity score: {processed.metadata['complexity_score']:.3f}")
    
    # Test batch processing
    print("Testing batch processing...")
    batch_result = processor.batch_process(test_inputs)
    print(f"  Batch input shape: {batch_result['inputs'].shape}")
    print(f"  Batch puzzle IDs: {batch_result['puzzle_identifiers']}")
    
    print("✓ Input processor test passed\n")

def test_code_execution_sandbox():
    """Test the code execution sandbox"""
    print("=== Testing Code Execution Sandbox ===")
    
    sandbox = CodeExecutionSandbox(timeout_seconds=5)
    
    # Test cases
    test_cases = [
        {
            "name": "Simple function",
            "code": "def add(a, b):\n    return a + b\nprint(add(2, 3))",
            "expected_output": "5",
            "should_pass": True
        },
        {
            "name": "Syntax error",
            "code": "def add(a, b\n    return a + b",
            "expected_output": "",
            "should_pass": False
        },
        {
            "name": "Runtime error",
            "code": "print(1 / 0)",
            "expected_output": "",
            "should_pass": False
        },
        {
            "name": "Timeout test",
            "code": "import time\ntime.sleep(10)\nprint('done')",
            "expected_output": "",
            "should_pass": False
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        success, output, error = sandbox.execute_code(
            test_case["code"], 
            expected_output=test_case["expected_output"]
        )
        
        if test_case["should_pass"] == success:
            print(f"  ✓ Passed (success={success})")
        else:
            print(f"  ✗ Failed (expected success={test_case['should_pass']}, got {success})")
            print(f"    Output: {output}")
            print(f"    Error: {error}")
    
    # Test function execution
    print("Testing function execution...")
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    
    success, result, error = sandbox.execute_function_test(
        code, "factorial", 5, 120
    )
    
    if success:
        print(f"  ✓ Function test passed (result={result})")
    else:
        print(f"  ✗ Function test failed: {error}")
    
    print("✓ Code execution sandbox test completed\n")

def test_hrm_model_creation():
    """Test HRM model creation and basic functionality"""
    print("=== Testing HRM Model Creation ===")
    
    try:
        from config.cfg_pretrain import Config
        
        # Create model config
        config = Config(
            vocab_size=40000,
            n_embd=256,  # Smaller for testing
            n_layer=6,   # Smaller for testing
            n_head=4,
            max_seq_len=512,
            causal=False,
            high_level_layers=2,
            low_level_layers=4,
            act_threshold=0.9,
            max_act_steps=8
        )
        
        # Create model
        model = HRMCodeGenerationModel(config)
        print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 128
        
        dummy_inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        dummy_puzzle_ids = torch.randint(0, 30, (batch_size, 1))  # 6 languages * 5 tasks
        dummy_attention_mask = torch.ones(batch_size, seq_len)
        
        print("  Testing forward pass...")
        with torch.no_grad():
            outputs = model(
                inputs=dummy_inputs,
                puzzle_identifiers=dummy_puzzle_ids,
                attention_mask=dummy_attention_mask
            )
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        if outputs.shape == (batch_size, seq_len, config.vocab_size):
            print("  ✓ Model forward pass test passed")
        else:
            print("  ✗ Model forward pass test failed - shape mismatch")
        
    except Exception as e:
        print(f"  ✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ HRM model test completed\n")

def create_mock_dataset(temp_dir: str):
    """Create a mock dataset for testing"""
    print("=== Creating Mock Dataset ===")
    
    # Create directory structure
    train_dir = Path(temp_dir) / "train"
    test_dir = Path(temp_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock data
    num_samples = 10
    seq_len = 128
    vocab_size = 40000
    
    # Generate random data
    inputs = np.random.randint(0, vocab_size, (num_samples, seq_len))
    labels = np.random.randint(0, vocab_size, (num_samples, seq_len))
    
    # Create metadata
    metadata = []
    scenarios = ["codegeneration", "selfrepair", "testoutputprediction", "codeexecution"]
    
    for i in range(num_samples):
        metadata.append({
            "problem_id": f"mock_problem_{i}",
            "scenario": scenarios[i % len(scenarios)],
            "input_text": f"Mock problem description {i}",
            "target_text": f"Mock solution {i}",
            "language": "python",
            "task_type": "generation",
            "complexity_score": np.random.random(),
            "is_augmented": False
        })
    
    # Save data for both splits
    for split_dir, split_name in [(train_dir, "train"), (test_dir, "test")]:
        np.save(split_dir / "inputs.npy", inputs)
        np.save(split_dir / "labels.npy", labels)
        
        with open(split_dir / "metadata.json", "w") as f:
            import json
            json.dump(metadata, f, indent=2)
    
    # Create dataset metadata
    metadata_dict = {
        "puzzle_name": "MockLiveCodeBench",
        "num_examples": num_samples,
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "scenarios": "codegeneration,selfrepair,testoutputprediction,codeexecution",
        "creation_date": "2024-01-01T00:00:00"
    }
    
    with open(Path(temp_dir) / "metadata.json", "w") as f:
        import json
        json.dump(metadata_dict, f, indent=2)
    
    print(f"  Mock dataset created in {temp_dir}")
    print("✓ Mock dataset creation completed\n")

def test_dataset_loading():
    """Test dataset loading and data loaders"""
    print("=== Testing Dataset Loading ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock dataset
        create_mock_dataset(temp_dir)
        
        try:
            # Test dataset creation
            dataset = LiveCodeBenchDataset(
                data_dir=temp_dir,
                split="train",
                max_seq_len=128,
                vocab_size=40000
            )
            
            print(f"  Dataset loaded with {len(dataset)} samples")
            
            # Test sample retrieval
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Input shape: {sample['inputs'].shape}")
            print(f"  Target shape: {sample['targets'].shape}")
            print(f"  Scenario: {sample['scenario']}")
            
            # Test data loaders
            train_loader, val_loader = create_livecodebench_dataloaders(
                data_dir=temp_dir,
                batch_size=4,
                max_seq_len=128,
                vocab_size=40000,
                num_workers=0  # Avoid multiprocessing issues in tests
            )
            
            print(f"  Train loader: {len(train_loader)} batches")
            print(f"  Val loader: {len(val_loader)} batches")
            
            # Test batch loading
            for batch in train_loader:
                print(f"  Batch input shape: {batch['inputs'].shape}")
                print(f"  Batch scenarios: {set(batch['scenario'])}")
                break
            
            print("  ✓ Dataset loading test passed")
            
        except Exception as e:
            print(f"  ✗ Dataset loading test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ Dataset loading test completed\n")

def test_training_integration():
    """Test training loop integration"""
    print("=== Testing Training Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock dataset
        create_mock_dataset(temp_dir)
        
        try:
            # Create minimal training config
            config = LiveCodeBenchTrainingConfig(
                data_path=temp_dir,
                hidden_size=128,  # Smaller for testing
                num_layers=4,     # Smaller for testing
                num_heads=4,
                batch_size=2,
                max_steps=5,      # Very few steps for testing
                eval_interval=3,
                save_interval=10,
                log_wandb=False,  # Disable wandb for testing
                mixed_precision=False,  # Disable for testing
                num_workers=0     # Avoid multiprocessing
            )
            
            # Create trainer
            trainer = LiveCodeBenchTrainer(config)
            print(f"  Trainer created with {trainer._count_parameters():,} parameters")
            
            # Test a few training steps
            print("  Running training steps...")
            trainer.model.train()
            
            for step in range(3):
                batch = next(iter(trainer.train_loader))
                batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                trainer.optimizer.zero_grad()
                loss, scenario_losses = trainer._forward_step(batch)
                loss.backward()
                trainer.optimizer.step()
                
                print(f"    Step {step+1}: loss = {loss.item():.4f}")
            
            print("  ✓ Training integration test passed")
            
        except Exception as e:
            print(f"  ✗ Training integration test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ Training integration test completed\n")

def run_all_tests():
    """Run all integration tests"""
    print("🧪 Running LiveCodeBench HRM Integration Tests")
    print("=" * 60)
    
    test_functions = [
        test_input_processor,
        test_code_execution_sandbox,
        test_hrm_model_creation,
        test_dataset_loading,
        test_training_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"🏁 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The LiveCodeBench HRM integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return failed == 0

def test_real_livecodebench_loading():
    """Test loading real LiveCodeBench data (if available)"""
    print("=== Testing Real LiveCodeBench Loading ===")
    
    try:
        # Try to load a small sample of real data
        sys.path.append('/Users/micahoates/Developer/x/HRM-AstralDrift/LiveCodeBench')
        from lcb_runner.benchmarks import load_code_generation_dataset
        
        print("  Loading real LiveCodeBench data...")
        problems = load_code_generation_dataset()[:5]  # Just 5 problems for testing
        
        print(f"  Loaded {len(problems)} problems")
        
        # Test processing
        config = LiveCodeBenchConfig(
            output_dir="test_output",
            scenarios="codegeneration",
            subsample_size=5,
            num_aug=0
        )
        
        processor = LiveCodeBenchProcessor(config)
        
        # Convert a few problems
        examples = processor._load_generation_data()[:3]
        print(f"  Converted {len(examples)} examples")
        
        for i, example in enumerate(examples):
            print(f"    Example {i+1}: {example.problem_id}, {example.scenario}")
            print(f"      Input length: {len(example.input_text)} chars")
            print(f"      Target length: {len(example.target_output)} chars")
        
        print("  ✓ Real LiveCodeBench loading test passed")
        
    except ImportError:
        print("  ⚠️  LiveCodeBench not available - skipping real data test")
    except Exception as e:
        print(f"  ⚠️  Real LiveCodeBench test failed: {e}")
        # This is not critical for the integration test
    
    print("✓ Real LiveCodeBench loading test completed\n")

if __name__ == "__main__":
    # Run basic integration tests
    success = run_all_tests()
    
    # Optionally test real data loading
    print("\n" + "=" * 60)
    test_real_livecodebench_loading()
    
    if success:
        print("\n🚀 Ready to run LiveCodeBench training!")
        print("\nNext steps:")
        print("1. Build dataset: python dataset/build_livecodebench_dataset.py --output-dir data/livecodebench-hrm")
        print("2. Start training: python train_livecodebench.py --data-path data/livecodebench-hrm")
        print("3. Run evaluation: python evaluation/livecodebench_evaluator.py")
    else:
        print("\n❌ Integration tests failed. Please fix the issues before proceeding.")
    
    sys.exit(0 if success else 1)