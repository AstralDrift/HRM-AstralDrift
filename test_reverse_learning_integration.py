#!/usr/bin/env python3
"""
Reverse Learning Integration Test

This script tests the complete reverse learning framework integration including:
- Gradient flow validation for bidirectional feedback
- ReverseLearningModule functionality  
- HRM architecture integration
- Training pipeline with reverse learning loss
- Performance and stability validation

Run with: python test_reverse_learning_integration.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config, HierarchicalReasoningModel_ACTV1
from models.hrm.reverse_learning import ReverseLearningModule, ReverseLearningMetrics
from models.losses import ACTSWESearchLossHead


def create_test_config():
    """Create a test configuration for HRM with Reverse Learning enabled"""
    return HierarchicalReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=64,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=1000,
        H_cycles=2,
        L_cycles=3,
        H_layers=2,
        L_layers=2,
        hidden_size=128,
        expansion=2.0,
        num_heads=4,
        pos_encodings="learned",
        halt_max_steps=3,
        halt_exploration_prob=0.1,
        # Reverse Learning configuration
        enable_reverse_learning=True,
        reverse_feedback_weight=0.1,
        reverse_learning_consistency_weight=0.5
    )


def create_test_batch(config):
    """Create a test batch for model evaluation"""
    batch_size = config.batch_size
    seq_len = config.seq_len
    vocab_size = config.vocab_size
    
    # Use optimal device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    return {
        'inputs': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),  
        'puzzle_identifiers': torch.zeros((batch_size,), dtype=torch.long, device=device)
    }


def test_reverse_learning_module():
    """Test ReverseLearningModule functionality"""
    print("🧪 Testing Reverse Learning Module...")
    
    try:
        config = create_test_config()
        module = ReverseLearningModule(config)
        
        # Test basic initialization
        assert hasattr(module, 'insight_extractor')
        assert hasattr(module, 'reverse_projector')
        assert hasattr(module, 'insight_gate')
        assert hasattr(module, 'pattern_memory')
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        hidden_size = config.hidden_size
        forward_dtype = getattr(torch, config.forward_dtype)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size, dtype=forward_dtype)
        z_L = torch.randn(batch_size, seq_len, hidden_size, dtype=forward_dtype)
        
        # Test insight extraction
        insights, extract_metrics = module.extract_implementation_insights(z_L)
        
        # Validate insight extraction outputs
        assert insights.shape == (batch_size, seq_len, hidden_size // 4)
        assert isinstance(extract_metrics, ReverseLearningMetrics)
        assert extract_metrics.insight_strength >= 0
        assert 0 <= extract_metrics.feedback_magnitude <= 1
        
        # Test feedback integration
        z_H_refined, final_metrics = module.integrate_reverse_feedback(z_H, insights, extract_metrics)
        
        # Validate integration outputs
        assert z_H_refined.shape == z_H.shape
        assert isinstance(final_metrics, ReverseLearningMetrics)
        assert 0 <= final_metrics.integration_gate_value <= 1
        assert final_metrics.planning_refinement_score >= 0
        
        # Test complete forward pass
        z_H_final, complete_metrics = module(z_H, z_L)
        assert z_H_final.shape == z_H.shape
        assert isinstance(complete_metrics, ReverseLearningMetrics)
        
        # Test statistics collection
        stats = module.get_reverse_learning_statistics()
        print(f"   Statistics: {list(stats.keys()) if stats else 'No stats yet'}")
        
        print("   ✅ Reverse Learning Module tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Reverse Learning Module test failed: {e}")
        traceback.print_exc()
        return False


def test_hrm_integration():
    """Test HRM model integration with Reverse Learning"""
    print("🧪 Testing HRM-Reverse Learning Integration...")
    
    try:
        config = create_test_config()
        model = HierarchicalReasoningModel_ACTV1(config.__dict__)
        # Move to correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model = model.to(device)
        
        # Test model has reverse learning components
        assert hasattr(model.inner, 'reverse_learning')
        assert hasattr(model.inner, '_reverse_metrics')
        
        # Test batch processing
        batch = create_test_batch(config)
        carry = model.initial_carry(batch)
        
        # Forward pass
        new_carry, outputs = model(carry, batch)
        
        # Validate outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == config.batch_size
        assert outputs['logits'].shape[-1] == config.vocab_size
        
        # Test reverse learning metrics retrieval
        reverse_metrics = model.get_latest_reverse_metrics()
        print(f"   Reverse metrics collected: {len(reverse_metrics)}")
        
        # Test reverse learning statistics
        stats = model.get_reverse_learning_statistics()
        print(f"   Reverse learning statistics: {list(stats.keys())}")
        
        print("   ✅ HRM-Reverse Learning integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ HRM-Reverse Learning integration test failed: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow_validation():
    """Test gradient flow with bidirectional feedback"""
    print("🧪 Testing Gradient Flow Validation...")
    
    try:
        config = create_test_config()
        model = HierarchicalReasoningModel_ACTV1(config.__dict__)
        # Move to correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model = model.to(device)
        
        # Create loss head with reverse learning
        loss_head = ACTSWESearchLossHead(model, 'softmax_cross_entropy', 
                                       swe_search_weight=0.0,  # Disable SWE-Search to isolate reverse learning
                                       reverse_learning_weight=0.2)
        
        # Test batch
        batch = create_test_batch(config)
        carry = loss_head.initial_carry(batch)
        
        # Enable gradient computation
        model.train()
        
        # Forward pass through loss head
        new_carry, loss, metrics, outputs, all_halted = loss_head(
            return_keys=['logits'],
            carry=carry,
            batch=batch
        )
        
        # Test gradient computation
        loss.backward()
        
        # Validate gradient flow through reverse learning components
        reverse_learning = model.inner.reverse_learning
        
        # Check gradients exist for key components
        assert reverse_learning.insight_extractor[0].weight.grad is not None, "Insight extractor gradients missing"
        assert reverse_learning.reverse_projector[0].weight.grad is not None, "Reverse projector gradients missing"
        assert reverse_learning.insight_gate[0].weight.grad is not None, "Insight gate gradients missing"
        
        # Check gradient magnitudes are reasonable
        insight_grad_norm = reverse_learning.insight_extractor[0].weight.grad.norm().item()
        projector_grad_norm = reverse_learning.reverse_projector[0].weight.grad.norm().item()
        gate_grad_norm = reverse_learning.insight_gate[0].weight.grad.norm().item()
        
        print(f"   Gradient norms - Insight: {insight_grad_norm:.6f}, Projector: {projector_grad_norm:.6f}, Gate: {gate_grad_norm:.6f}")
        
        # Validate gradients are not exploding or vanishing
        assert 1e-8 < insight_grad_norm < 100, f"Insight extractor gradient norm out of range: {insight_grad_norm}"
        assert 1e-8 < projector_grad_norm < 100, f"Reverse projector gradient norm out of range: {projector_grad_norm}"
        assert 1e-8 < gate_grad_norm < 100, f"Insight gate gradient norm out of range: {gate_grad_norm}"
        
        # Test main model gradients still flow properly
        main_grad_norm = model.inner.lm_head.weight.grad.norm().item()
        print(f"   Main model gradient norm: {main_grad_norm:.6f}")
        assert 1e-8 < main_grad_norm < 100, f"Main model gradient norm out of range: {main_grad_norm}"
        
        # Validate loss computation
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Validate metrics
        assert isinstance(metrics, dict)
        assert 'count' in metrics
        assert 'accuracy' in metrics
        assert 'lm_loss' in metrics
        
        # Check for reverse learning specific metrics
        if 'reverse_learning_loss' in metrics:
            print(f"   Reverse learning loss: {metrics['reverse_learning_loss']:.6f}")
        
        if any(k.startswith('reverse_') for k in metrics.keys()):
            reverse_metrics = {k: v for k, v in metrics.items() if k.startswith('reverse_')}
            print(f"   Reverse learning metrics: {reverse_metrics}")
        
        print("   ✅ Gradient flow validation tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Gradient flow validation test failed: {e}")
        traceback.print_exc()
        return False


def test_stability_validation():
    """Test training stability with reverse learning"""
    print("🧪 Testing Training Stability...")
    
    try:
        config = create_test_config()
        model = HierarchicalReasoningModel_ACTV1(config.__dict__)
        # Move to correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        model = model.to(device)
        
        # Create loss head
        loss_head = ACTSWESearchLossHead(model, 'softmax_cross_entropy', 
                                       reverse_learning_weight=0.1)
        
        # Track losses over multiple steps
        losses = []
        gradient_norms = []
        
        model.train()
        
        for step in range(10):  # Test 10 training steps
            # Create batch
            batch = create_test_batch(config)
            carry = loss_head.initial_carry(batch)
            
            # Zero gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Forward pass
            new_carry, loss, metrics, outputs, all_halted = loss_head(
                return_keys=['logits'],
                carry=carry,
                batch=batch
            )
            
            # Backward pass
            loss.backward()
            
            # Track metrics
            losses.append(loss.item())
            
            # Compute gradient norm
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            gradient_norms.append(total_grad_norm)
            
            # Check for instability
            assert not torch.isnan(loss), f"Loss became NaN at step {step}"
            assert not torch.isinf(loss), f"Loss became infinite at step {step}"
            assert total_grad_norm < 1000, f"Gradient explosion at step {step}: {total_grad_norm}"
            assert total_grad_norm > 1e-8, f"Gradient vanishing at step {step}: {total_grad_norm}"
        
        # Analyze stability
        loss_std = np.std(losses)
        grad_std = np.std(gradient_norms)
        
        print(f"   Loss stability - Mean: {np.mean(losses):.6f}, Std: {loss_std:.6f}")
        print(f"   Gradient stability - Mean: {np.mean(gradient_norms):.6f}, Std: {grad_std:.6f}")
        
        # Validate reasonable stability (losses shouldn't vary wildly)
        assert loss_std < 10.0, f"Loss too unstable: std={loss_std}"
        assert grad_std < 100.0, f"Gradients too unstable: std={grad_std}"
        
        print("   ✅ Training stability tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Training stability test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance and efficiency metrics"""
    print("🧪 Testing Performance Metrics...")
    
    try:
        config = create_test_config()
        
        # Test with and without reverse learning
        configs = [
            (config, "With Reverse Learning"),
            (HierarchicalReasoningModel_ACTV1Config(**{**config.__dict__, 'enable_reverse_learning': False}), "Without Reverse Learning")
        ]
        
        performance_results = {}
        
        for test_config, test_name in configs:
            model = HierarchicalReasoningModel_ACTV1(test_config.__dict__)
            
            # Memory usage test (if psutil available)
            memory_overhead = 0
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Create batch and run inference
                batch = create_test_batch(test_config)
                carry = model.initial_carry(batch)
                
                # Time inference
                import time
                start_time = time.time()
                
                new_carry, outputs = model(carry, batch)
                
                inference_time = time.time() - start_time
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_overhead = memory_after - memory_before
                
                performance_results[test_name] = {
                    'inference_time': inference_time,
                    'memory_usage': memory_after,
                    'memory_overhead': memory_overhead
                }
                
            except ImportError:
                print(f"   Memory tracking skipped for {test_name} (psutil not available)")
                
                # Still test inference time
                batch = create_test_batch(test_config)
                carry = model.initial_carry(batch)
                
                import time
                start_time = time.time()
                new_carry, outputs = model(carry, batch)
                inference_time = time.time() - start_time
                
                performance_results[test_name] = {
                    'inference_time': inference_time,
                    'memory_usage': 0,
                    'memory_overhead': 0
                }
        
        # Compare performance
        for name, results in performance_results.items():
            print(f"   {name}:")
            print(f"     Inference time: {results['inference_time']*1000:.2f}ms")
            if results['memory_usage'] > 0:
                print(f"     Memory usage: {results['memory_usage']:.1f}MB")
                print(f"     Memory overhead: {results['memory_overhead']:.1f}MB")
        
        # Validate reverse learning doesn't add excessive overhead
        if len(performance_results) == 2:
            with_rl = performance_results["With Reverse Learning"]
            without_rl = performance_results["Without Reverse Learning"]
            
            time_overhead = (with_rl['inference_time'] - without_rl['inference_time']) / without_rl['inference_time']
            
            print(f"   Time overhead from reverse learning: {time_overhead*100:.1f}%")
            
            # Validate overhead is reasonable (<30% for verification)
            assert time_overhead < 0.3, f"Reverse learning adds too much time overhead: {time_overhead*100:.1f}%"
        
        print("   ✅ Performance metrics tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance metrics test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all reverse learning integration tests"""
    print("🚀 Reverse Learning Framework Integration Tests")
    print("=" * 60)
    
    tests = [
        test_reverse_learning_module,
        test_hrm_integration,
        test_gradient_flow_validation,
        test_stability_validation,
        test_performance_metrics
    ]
    
    results = []
    for test_func in tests:
        print()
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Reverse Learning framework is ready.")
        print("\nNext steps:")
        print("1. Configure training with enable_reverse_learning=True")
        print("2. Use ACTSWESearchLossHead with reverse_learning_weight parameter")
        print("3. Monitor reverse learning metrics during training")
        print("4. Evaluate code quality improvements on test data")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())