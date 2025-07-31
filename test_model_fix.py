#!/usr/bin/env python3
"""
Test script to validate the NotImplementedError fix in HRM model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config

def test_model_initialization():
    """Test that the model can be initialized with different positional encodings"""
    
    print("🧪 Testing Model Initialization Fix")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {"name": "rope", "pos_encodings": "rope"},
        {"name": "rotary", "pos_encodings": "rotary"}, 
        {"name": "learned", "pos_encodings": "learned"},
    ]
    
    base_config = {
        'vocab_size': 50265,
        'batch_size': 2,
        'seq_len': 128,
        'puzzle_emb_ndim': 0,  # No puzzle embeddings for code generation
        'num_puzzle_identifiers': 100,
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 2.0,
        'num_heads': 4,
        'halt_max_steps': 3,
        'halt_exploration_prob': 0.1,
        'causal': False,
        'head_size': 64,
        'act_threshold': 0.99,
        'rope_theta': 10000.0,
        'rms_norm_eps': 1e-5,
        'forward_dtype': 'bfloat16',
        'puzzle_emb_vocab_size': 1000,
        'puzzle_emb_dim': 256,
        'max_seq_len': 128
    }
    
    results = {}
    
    for test_config in test_configs:
        config_name = test_config["name"]
        pos_encoding = test_config["pos_encodings"]
        
        print(f"\n📋 Testing {config_name} positional encoding...")
        
        try:
            # Create config
            model_config = {**base_config, 'pos_encodings': pos_encoding}
            config = HierarchicalReasoningModel_ACTV1Config(**model_config)
            
            # Try to create model
            model = HierarchicalReasoningModel_ACTV1(config)
            
            print(f"   ✅ {config_name}: Model created successfully")
            print(f"      - Hidden size: {model.config.hidden_size}")
            print(f"      - Positional encoding: {model.config.pos_encodings}")
            
            # Check if the right components were created
            if pos_encoding in ["rope", "rotary"]:
                if hasattr(model.inner, 'rotary_emb'):
                    print(f"      - ✅ Rotary embedding initialized")
                else:
                    print(f"      - ❌ Rotary embedding missing")
            elif pos_encoding == "learned":
                if hasattr(model.inner, 'embed_pos'):
                    print(f"      - ✅ Learned position embedding initialized")
                else:
                    print(f"      - ❌ Learned position embedding missing")
            
            results[config_name] = True
            
        except Exception as e:
            print(f"   ❌ {config_name}: Failed with error: {e}")
            results[config_name] = False
    
    # Test invalid positional encoding
    print(f"\n📋 Testing invalid positional encoding...")
    try:
        invalid_config = {**base_config, 'pos_encodings': 'invalid'}
        config = HierarchicalReasoningModel_ACTV1Config(**invalid_config)
        model = HierarchicalReasoningModel_ACTV1(config)
        print(f"   ❌ Should have failed but didn't!")
        results["invalid_test"] = False
    except ValueError as e:
        print(f"   ✅ Correctly failed with ValueError: {e}")
        results["invalid_test"] = True
    except Exception as e:
        print(f"   ⚠️ Failed with unexpected error: {e}")
        results["invalid_test"] = False
    
    # Summary
    print(f"\n📊 Results Summary:")
    print("=" * 30)
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! NotImplementedError fix is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = test_model_initialization()
    sys.exit(0 if success else 1)