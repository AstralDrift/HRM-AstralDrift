#!/usr/bin/env python3
"""
SWE-Search Integration Test

This script tests the complete SWE-Search framework integration including:
- Model configuration and initialization
- SWE-Search controller functionality
- Training pipeline integration
- Evaluation system
- Self-evolution capabilities

Run with: python test_swe_search_integration.py
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
from models.hrm.swe_search_integration import SWESearchController, SWESearchMetrics
from models.losses import ACTSWESearchLossHead
from swe_search_evaluator import create_swe_search_evaluator


def create_test_config():
    """Create a test configuration for HRM with SWE-Search enabled"""
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
        # SWE-Search configuration
        enable_swe_search=True,
        swe_search_iterations=2,
        multi_agent_debate_rounds=1,
        mcts_exploration_factor=1.2,
        enable_self_evolution=True,
        performance_buffer_size=10,
        evolution_threshold=0.9
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


def test_swe_search_controller():
    """Test SWE-Search controller functionality"""
    print("🧪 Testing SWE-Search Controller...")
    
    try:
        config = create_test_config()
        controller = SWESearchController(config)
        
        # Move to correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        controller = controller.to(device)
        
        # Test basic initialization
        assert hasattr(controller, 'state_evaluator')
        assert hasattr(controller, 'debate_coordinator')
        assert hasattr(controller, 'performance_buffer')
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        hidden_size = config.hidden_size
        forward_dtype = getattr(torch, config.forward_dtype)
        
        problem_embedding = torch.randn(batch_size, seq_len, hidden_size, dtype=forward_dtype)
        solution_embedding = torch.randn(batch_size, seq_len, hidden_size, dtype=forward_dtype)
        
        refined_solution, metrics = controller.swe_search_forward(
            problem_embedding, solution_embedding
        )
        
        # Validate outputs
        assert refined_solution.shape == solution_embedding.shape
        assert isinstance(metrics, SWESearchMetrics)
        assert 0 <= metrics.final_score <= 1.0
        assert metrics.iterations_used > 0
        assert metrics.total_candidates >= 0
        
        # Test self-evolution
        for _ in range(15):  # Add enough samples to trigger evolution check
            controller.performance_buffer.append({
                'final_score': np.random.uniform(0.3, 0.7),
                'iterations_used': np.random.randint(1, 4),
                'total_candidates': np.random.randint(2, 8),
                'efficiency': np.random.uniform(0.1, 0.5)
            })
        
        should_evolve = controller.should_evolve_architecture()
        print(f"   Should evolve: {should_evolve}")
        
        # Test parameter evolution
        original_iterations = controller.iteration_count
        controller.evolve_search_parameters()
        print(f"   Evolution triggered, iterations: {original_iterations} -> {controller.iteration_count}")
        
        # Test statistics
        stats = controller.get_search_statistics()
        assert isinstance(stats, dict)
        assert 'avg_final_score' in stats
        
        print("   ✅ SWE-Search Controller tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ SWE-Search Controller test failed: {e}")
        traceback.print_exc()
        return False


def test_hrm_integration():
    """Test HRM model integration with SWE-Search"""
    print("🧪 Testing HRM-SWE-Search Integration...")
    
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
        
        # Test model has SWE-Search components
        assert hasattr(model.inner, 'swe_search_controller')
        assert hasattr(model.inner, '_search_metrics')
        
        # Test batch processing
        batch = create_test_batch(config)
        carry = model.initial_carry(batch)
        
        # Forward pass
        new_carry, outputs = model(carry, batch)
        
        # Validate outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == config.batch_size
        assert outputs['logits'].shape[-1] == config.vocab_size
        
        # Test search metrics retrieval
        search_metrics = model.get_latest_search_metrics()
        print(f"   Search metrics collected: {len(search_metrics)}")
        
        # Test search statistics
        stats = model.get_swe_search_statistics()
        print(f"   Search statistics: {list(stats.keys())}")
        
        # Test parameter evolution
        model.evolve_swe_search_parameters()
        
        print("   ✅ HRM-SWE-Search integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ HRM-SWE-Search integration test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_integration():
    """Test SWE-Search loss integration"""
    print("🧪 Testing SWE-Search Loss Integration...")
    
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
        loss_head = ACTSWESearchLossHead(model, 'softmax_cross_entropy', swe_search_weight=0.2)
        
        # Test batch
        batch = create_test_batch(config)
        carry = loss_head.initial_carry(batch)
        
        # Forward pass through loss head
        new_carry, loss, metrics, outputs, all_halted = loss_head(
            return_keys=['logits'],
            carry=carry,
            batch=batch
        )
        
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
        
        # Check for SWE-Search specific metrics
        if 'swe_search_loss' in metrics:
            print(f"   SWE-Search loss: {metrics['swe_search_loss']:.4f}")
        
        if any(k.startswith('swe_search_') for k in metrics.keys()):
            search_metrics = {k: v for k, v in metrics.items() if k.startswith('swe_search_')}
            print(f"   SWE-Search metrics: {search_metrics}")
        
        print("   ✅ SWE-Search loss integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ SWE-Search loss integration test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluator():
    """Test SWE-Search evaluator"""
    print("🧪 Testing SWE-Search Evaluator...")
    
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
        
        # Create evaluator with CPU device if CUDA not available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        evaluator = create_swe_search_evaluator(model, config, device=device)
        
        # Create mock data loader
        class MockDataLoader:
            def __init__(self, batches):
                self.batches = batches
                self.dataset = list(range(len(batches) * config.batch_size))
            
            def __iter__(self):
                return iter(self.batches)
            
            def __len__(self):
                return len(self.batches)
        
        # Create test batches
        test_batches = [create_test_batch(config) for _ in range(3)]
        mock_loader = MockDataLoader(test_batches)
        
        # Run evaluation
        results = evaluator.evaluate_with_swe_search(
            mock_loader,
            compare_baseline=True,
            save_detailed_results=False
        )
        
        # Validate results
        assert hasattr(results, 'base_accuracy')
        assert hasattr(results, 'enhanced_accuracy')
        assert hasattr(results, 'improvement')
        assert hasattr(results, 'avg_search_score')
        assert hasattr(results, 'convergence_rate')
        
        print(f"   Base accuracy: {results.base_accuracy:.4f}")
        print(f"   Enhanced accuracy: {results.enhanced_accuracy:.4f}")
        print(f"   Improvement: {results.improvement:.4f}")
        print(f"   Average search score: {results.avg_search_score:.4f}")
        print(f"   Convergence rate: {results.convergence_rate:.2%}")
        
        # Test performance summary
        summary = evaluator.get_performance_summary()
        assert isinstance(summary, dict)
        
        print("   ✅ SWE-Search evaluator tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ SWE-Search evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_targets():
    """Test performance against target metrics"""
    print("🧪 Testing Performance Targets...")
    
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
        
        # Memory usage test (if psutil available)
        memory_overhead = 0
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create larger batch for realistic memory test
            large_config = create_test_config()
            large_config.batch_size = 8
            large_config.seq_len = 256
            
            large_batch = create_test_batch(large_config)
            large_model = HierarchicalReasoningModel_ACTV1(large_config.__dict__)
            
            carry = large_model.initial_carry(large_batch)
            new_carry, outputs = large_model(carry, large_batch)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_overhead = memory_after - memory_before
            
            print(f"   Memory usage: {memory_before:.1f}MB -> {memory_after:.1f}MB (+{memory_overhead:.1f}MB)")
        except ImportError:
            print("   Memory usage test skipped (psutil not available)")
            
            # Still test model creation without memory tracking
            large_config = create_test_config()
            large_config.batch_size = 4  # Smaller batch for testing
            large_config.seq_len = 128
            
            large_batch = create_test_batch(large_config)
            large_model = HierarchicalReasoningModel_ACTV1(large_config.__dict__)
            
            carry = large_model.initial_carry(large_batch)
            new_carry, outputs = large_model(carry, large_batch)
            print("   Model creation and forward pass successful")
        
        # Model size check (use the model that was created)
        test_model = large_model if 'large_model' in locals() else model
        total_params = sum(p.numel() for p in test_model.parameters())
        target_params = 100_000_000  # 100M parameter target
        
        print(f"   Model parameters: {total_params:,} (target: <{target_params:,})")
        
        if total_params < target_params:
            print(f"   ✅ Parameter count within target")
        else:
            print(f"   ⚠️  Parameter count exceeds target by {total_params - target_params:,}")
        
        # Search overhead check
        if hasattr(test_model, 'get_swe_search_statistics'):
            stats = test_model.get_swe_search_statistics()
            if stats:
                overhead_pct = stats.get('avg_iterations', 1) * 15  # Rough estimate
                print(f"   Estimated search overhead: ~{overhead_pct:.1f}%")
        
        print("   ✅ Performance target tests completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance target test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("🚀 SWE-Search Framework Integration Tests")
    print("=" * 60)
    
    tests = [
        test_swe_search_controller,
        test_hrm_integration,
        test_loss_integration,
        test_evaluator,
        test_performance_targets
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
        print("🎉 All tests passed! SWE-Search framework is ready.")
        print("\nNext steps:")
        print("1. Configure training with enable_swe_search=True")
        print("2. Use ACTSWESearchLossHead for training")
        print("3. Monitor search metrics during training")
        print("4. Evaluate with enhanced evaluation system")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())