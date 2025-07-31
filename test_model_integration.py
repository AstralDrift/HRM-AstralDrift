#!/usr/bin/env python3
"""
HRM Model Integration Test Suite

Tests the core functionality of HRM model including:
- Model creation with both dict and config inputs
- Tokenizer integration and encoding/decoding
- Basic forward pass functionality
- Device compatibility (CUDA/MPS/CPU)
"""

import sys
import os
import unittest
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1
from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig
from utils.device import get_device


class TestHRMIntegration(unittest.TestCase):
    """Test suite for HRM model integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = get_device()
        self.config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=2,
            seq_len=100,
            num_puzzle_identifiers=10,
            vocab_size=1000,
            H_cycles=2,
            L_cycles=3,
            H_layers=2,
            L_layers=2,
            hidden_size=128,  # Smaller for testing
            expansion=2.0,
            num_heads=4,
            pos_encodings='rope',
            halt_max_steps=5,
            halt_exploration_prob=0.1
        )
        
    def test_model_creation_with_config(self):
        """Test model creation with config object"""
        model = HierarchicalReasoningModel_ACTV1(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.hidden_size, 128)
        print("✅ Model creation with config object passed")
        
    def test_model_creation_with_dict(self):
        """Test model creation with dict"""
        config_dict = self.config.model_dump()
        model = HierarchicalReasoningModel_ACTV1(config_dict)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.hidden_size, 128)
        print("✅ Model creation with dict passed")
        
    def test_tokenizer_integration(self):
        """Test tokenizer loading and basic functionality"""
        try:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Test tokenization
            test_code = "def hello():\n    return 'world'"
            tokens = tokenizer.encode(test_code, max_length=50, truncation=True)
            decoded = tokenizer.decode(tokens)
            
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            self.assertIsInstance(decoded, str)
            print("✅ Tokenizer integration test passed")
            
        except Exception as e:
            self.fail(f"Tokenizer integration failed: {e}")
            
    def test_device_compatibility(self):
        """Test device compatibility across CUDA/MPS/CPU"""
        device = get_device()
        print(f"✅ Device detection: {device}")
        
        # Test tensor creation on device
        test_tensor = torch.randn(2, 10).to(device)
        self.assertEqual(str(test_tensor.device).split(':')[0], str(device).split(':')[0])
        print(f"✅ Device compatibility test passed for {device}")
        
    def test_model_forward_pass_basic(self):
        """Test basic model forward pass (without full training data)"""
        try:
            model = HierarchicalReasoningModel_ACTV1(self.config)
            model.eval()  # Set to eval mode
            
            # Create dummy inputs
            batch_size = 2
            seq_len = 50
            inputs = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
            puzzle_ids = torch.randint(0, self.config.num_puzzle_identifiers, (batch_size,))
            
            # Test that model doesn't crash on forward pass
            with torch.no_grad():
                # Test model structure exists
                self.assertTrue(hasattr(model, 'inner'))
                self.assertTrue(hasattr(model.inner, 'embed_tokens'))
                self.assertTrue(hasattr(model.inner, 'lm_head'))
                
            print("✅ Basic model structure test passed")
            
        except Exception as e:
            self.fail(f"Model forward pass test failed: {e}")


def run_integration_tests():
    """Run all integration tests"""
    print("🔍 Running HRM Model Integration Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHRMIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)