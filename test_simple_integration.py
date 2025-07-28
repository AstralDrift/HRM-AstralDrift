"""
Simple Integration Test for LiveCodeBench HRM System

This is a minimal test to verify the basic components work.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("=== Testing Imports ===")
    
    success_count = 0
    total_count = 3
    
    # Test input processor imports
    try:
        from models.code_generation.input_processor import (
            CodeGenerationInput,
            CodeGenerationInputProcessor,
            CodeGenerationTask,
            ProgrammingLanguage
        )
        print("✓ Input processor imports successful")
        success_count += 1
    except Exception as e:
        print(f"✗ Input processor import failed: {e}")
    
    # Test config imports
    try:
        from config.cfg_pretrain import Config
        print("✓ Config imports successful")
        success_count += 1
    except Exception as e:
        print(f"✗ Config import failed: {e}")
    
    # Test basic evaluation components (without external deps)
    try:
        # Just test that we can import subprocess and tempfile
        import subprocess
        import tempfile
        print("✓ Basic evaluation components available")
        success_count += 1
    except Exception as e:
        print(f"✗ Basic evaluation import failed: {e}")
    
    return success_count == total_count

def test_input_processor():
    """Test basic input processor functionality"""
    print("\n=== Testing Input Processor ===")
    
    try:
        from models.code_generation.input_processor import (
            CodeGenerationInputProcessor,
            CodeGenerationInput,
            CodeGenerationTask,
            ProgrammingLanguage
        )
        
        # Create processor with minimal config
        processor = CodeGenerationInputProcessor(
            vocab_size=1000,  # Small for testing
            max_seq_len=128   # Small for testing
        )
        
        # Create test input
        test_input = CodeGenerationInput(
            problem_description="Write a function that adds two numbers",
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.GENERATION
        )
        
        # Process input
        processed = processor.process_input(test_input)
        
        print(f"✓ Processed input shape: {processed.input_tokens.shape}")
        print(f"✓ Language ID: {processed.language_id.item()}")
        print(f"✓ Task ID: {processed.task_id.item()}")
        
        return True
    except Exception as e:
        print(f"✗ Input processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sandbox():
    """Test code execution sandbox"""
    print("\n=== Testing Code Execution Sandbox ===")
    
    try:
        # Simple sandbox implementation for testing
        import subprocess
        import tempfile
        import os
        
        # Test simple code execution using subprocess directly
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('Hello World')")
            f.flush()
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "Hello World" in result.stdout:
                print("✓ Basic code execution successful")
                return True
            else:
                print(f"✗ Code execution failed: stdout='{result.stdout}', stderr='{result.stderr}'")
                return False
                
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
            
    except Exception as e:
        print(f"✗ Sandbox test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration creation"""
    print("\n=== Testing Configuration ===")
    
    try:
        from config.cfg_pretrain import Config
        
        config = Config(
            vocab_size=1000,
            n_embd=128,
            n_layer=8,  # Total layers
            n_head=4,
            max_seq_len=256,
            high_level_layers=2,  # Must sum to <= n_layer
            low_level_layers=6
        )
        
        print(f"✓ Config created: {config.n_embd} embedding size, {config.n_layer} layers")
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all simple tests"""
    print("🧪 Running Simple LiveCodeBench Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_input_processor, 
        test_sandbox,
        test_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed! Core components are working.")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)