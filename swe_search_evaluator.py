"""
SWE-Search Enhanced Evaluation System

This module provides comprehensive evaluation capabilities for the SWE-Search framework,
including search-specific metrics, performance analysis, and comparison against baseline.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import json
import time
from pathlib import Path

from models.hrm.swe_search_integration import SWESearchMetrics


@dataclass
class SWESearchEvaluationResults:
    """Comprehensive evaluation results for SWE-Search"""
    
    # Standard metrics
    base_accuracy: float
    enhanced_accuracy: float
    improvement: float
    
    # Search-specific metrics
    avg_search_score: float
    avg_iterations: float
    avg_candidates: float
    convergence_rate: float
    search_efficiency: float
    
    # Performance metrics
    inference_time_base: float
    inference_time_enhanced: float
    memory_overhead: float
    
    # Detailed results
    search_metrics_history: List[SWESearchMetrics]
    per_sample_improvements: List[float]


class SWESearchEvaluator:
    """Enhanced evaluator for SWE-Search framework with comprehensive metrics"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.results_history = []
        
    def evaluate_with_swe_search(self, 
                                test_data: torch.utils.data.DataLoader,
                                compare_baseline: bool = True,
                                save_detailed_results: bool = True) -> SWESearchEvaluationResults:
        """
        Comprehensive evaluation with SWE-Search enhancements
        
        Args:
            test_data: Test data loader
            compare_baseline: Whether to compare against baseline (non-search) performance
            save_detailed_results: Whether to save detailed per-sample results
            
        Returns:
            Comprehensive evaluation results
        """
        self.model.eval()
        
        # Metrics tracking
        base_correct = 0
        enhanced_correct = 0
        total_samples = 0
        
        search_metrics_history = []
        per_sample_improvements = []
        
        # Performance tracking
        base_inference_times = []
        enhanced_inference_times = []
        memory_usage_base = []
        memory_usage_enhanced = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['inputs'].size(0)
                
                # Initialize carry
                carry = self.model.initial_carry(batch)
                
                if compare_baseline:
                    # Baseline evaluation (disable SWE-Search temporarily)
                    original_swe_search_enabled = getattr(self.model.config, 'enable_swe_search', False)
                    self.model.config.enable_swe_search = False
                    
                    start_time = time.time()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    base_carry, base_outputs = self.model(carry, batch)
                    
                    base_time = time.time() - start_time
                    memory_after_base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    base_inference_times.append(base_time)
                    memory_usage_base.append(memory_after_base - memory_before)
                    
                    # Restore SWE-Search setting
                    self.model.config.enable_swe_search = original_swe_search_enabled
                else:
                    base_outputs = None
                    base_time = 0
                
                # Enhanced evaluation (with SWE-Search)
                if getattr(self.model.config, 'enable_swe_search', False):
                    carry = self.model.initial_carry(batch)  # Reset carry
                    
                    start_time = time.time()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    enhanced_carry, enhanced_outputs = self.model(carry, batch)
                    
                    enhanced_time = time.time() - start_time
                    memory_after_enhanced = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    enhanced_inference_times.append(enhanced_time)
                    memory_usage_enhanced.append(memory_after_enhanced - memory_before)
                    
                    # Get search metrics
                    if hasattr(self.model, 'get_latest_search_metrics'):
                        batch_search_metrics = self.model.get_latest_search_metrics()
                        search_metrics_history.extend(batch_search_metrics)
                else:
                    enhanced_outputs = base_outputs
                    enhanced_time = base_time
                
                # Compute accuracy
                labels = batch['labels']
                
                if base_outputs is not None:
                    base_predictions = torch.argmax(base_outputs['logits'], dim=-1)
                    base_batch_correct = self._compute_accuracy(base_predictions, labels)
                    base_correct += base_batch_correct
                else:
                    base_batch_correct = 0
                
                if enhanced_outputs is not None:
                    enhanced_predictions = torch.argmax(enhanced_outputs['logits'], dim=-1)
                    enhanced_batch_correct = self._compute_accuracy(enhanced_predictions, labels)
                    enhanced_correct += enhanced_batch_correct
                else:
                    enhanced_batch_correct = base_batch_correct
                
                # Track per-sample improvements
                if compare_baseline:
                    sample_improvement = (enhanced_batch_correct - base_batch_correct) / batch_size
                    per_sample_improvements.append(sample_improvement)
                
                total_samples += batch_size
                
                # Progress logging
                if batch_idx % 50 == 0:
                    print(f"Evaluated {batch_idx * batch_size}/{len(test_data.dataset)} samples")
        
        # Compute final metrics
        base_accuracy = base_correct / total_samples if total_samples > 0 else 0.0
        enhanced_accuracy = enhanced_correct / total_samples if total_samples > 0 else 0.0
        improvement = enhanced_accuracy - base_accuracy
        
        # Search-specific metrics
        search_stats = self._compute_search_statistics(search_metrics_history)
        
        # Performance metrics
        avg_base_time = np.mean(base_inference_times) if base_inference_times else 0.0
        avg_enhanced_time = np.mean(enhanced_inference_times) if enhanced_inference_times else 0.0
        
        avg_base_memory = np.mean(memory_usage_base) if memory_usage_base else 0.0
        avg_enhanced_memory = np.mean(memory_usage_enhanced) if memory_usage_enhanced else 0.0
        memory_overhead = (avg_enhanced_memory - avg_base_memory) / max(avg_base_memory, 1.0)
        
        # Create results
        results = SWESearchEvaluationResults(
            base_accuracy=base_accuracy,
            enhanced_accuracy=enhanced_accuracy,
            improvement=improvement,
            avg_search_score=search_stats['avg_search_score'],
            avg_iterations=search_stats['avg_iterations'],
            avg_candidates=search_stats['avg_candidates'],
            convergence_rate=search_stats['convergence_rate'],
            search_efficiency=search_stats['search_efficiency'],
            inference_time_base=avg_base_time,
            inference_time_enhanced=avg_enhanced_time,
            memory_overhead=memory_overhead,
            search_metrics_history=search_metrics_history,
            per_sample_improvements=per_sample_improvements
        )
        
        self.results_history.append(results)
        
        if save_detailed_results:
            self._save_detailed_results(results)
        
        return results
    
    def _compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> int:
        """Compute accuracy for given predictions and labels"""
        # Handle ignore labels
        mask = labels != -100
        if not mask.any():
            return 0
        
        correct = (predictions == labels) & mask
        return correct.sum().item()
    
    def _compute_search_statistics(self, search_metrics: List[SWESearchMetrics]) -> Dict[str, float]:
        """Compute aggregated search statistics"""
        if not search_metrics:
            return {
                'avg_search_score': 0.0,
                'avg_iterations': 0.0,
                'avg_candidates': 0.0,
                'convergence_rate': 0.0,
                'search_efficiency': 0.0
            }
        
        avg_search_score = np.mean([m.final_score for m in search_metrics])
        avg_iterations = np.mean([m.iterations_used for m in search_metrics])
        avg_candidates = np.mean([m.total_candidates for m in search_metrics])
        convergence_rate = np.mean([m.convergence_achieved for m in search_metrics])
        search_efficiency = np.mean([m.search_efficiency for m in search_metrics])
        
        return {
            'avg_search_score': avg_search_score,
            'avg_iterations': avg_iterations,
            'avg_candidates': avg_candidates,
            'convergence_rate': convergence_rate,
            'search_efficiency': search_efficiency
        }
    
    def _save_detailed_results(self, results: SWESearchEvaluationResults):
        """Save detailed evaluation results to file"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save summary results
        summary = {
            'base_accuracy': results.base_accuracy,
            'enhanced_accuracy': results.enhanced_accuracy,
            'improvement': results.improvement,
            'improvement_percentage': results.improvement / max(results.base_accuracy, 0.01) * 100,
            'avg_search_score': results.avg_search_score,
            'avg_iterations': results.avg_iterations,
            'avg_candidates': results.avg_candidates,
            'convergence_rate': results.convergence_rate,
            'search_efficiency': results.search_efficiency,
            'inference_time_base': results.inference_time_base,
            'inference_time_enhanced': results.inference_time_enhanced,
            'time_overhead_percentage': (results.inference_time_enhanced - results.inference_time_base) / max(results.inference_time_base, 0.001) * 100,
            'memory_overhead_percentage': results.memory_overhead * 100
        }
        
        timestamp = int(time.time())
        summary_file = results_dir / f"swe_search_evaluation_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Detailed results saved to {summary_file}")
        
        # Print summary
        self._print_results_summary(results)
    
    def _print_results_summary(self, results: SWESearchEvaluationResults):
        """Print a comprehensive results summary"""
        print("\n" + "="*80)
        print("SWE-SEARCH EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Base Accuracy:      {results.base_accuracy:.4f}")
        print(f"   Enhanced Accuracy:  {results.enhanced_accuracy:.4f}")
        print(f"   Improvement:        {results.improvement:.4f} ({results.improvement/max(results.base_accuracy, 0.01)*100:.2f}%)")
        
        print(f"\n🔍 SEARCH METRICS:")
        print(f"   Average Score:      {results.avg_search_score:.4f}")
        print(f"   Average Iterations: {results.avg_iterations:.2f}")
        print(f"   Average Candidates: {results.avg_candidates:.1f}")
        print(f"   Convergence Rate:   {results.convergence_rate:.2%}")
        print(f"   Search Efficiency:  {results.search_efficiency:.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Base Inference:     {results.inference_time_base*1000:.2f}ms")
        print(f"   Enhanced Inference: {results.inference_time_enhanced*1000:.2f}ms")
        time_overhead = (results.inference_time_enhanced - results.inference_time_base) / max(results.inference_time_base, 0.001) * 100
        print(f"   Time Overhead:      {time_overhead:.1f}%")
        print(f"   Memory Overhead:    {results.memory_overhead*100:.1f}%")
        
        print(f"\n🎯 SUCCESS CRITERIA CHECK:")
        target_improvement = 0.23  # 23% target improvement
        achieved_improvement = results.improvement / max(results.base_accuracy, 0.01)
        
        if achieved_improvement >= target_improvement:
            print(f"   ✅ Target improvement achieved: {achieved_improvement:.2%} >= {target_improvement:.2%}")
        else:
            print(f"   ❌ Target improvement not met: {achieved_improvement:.2%} < {target_improvement:.2%}")
        
        if results.memory_overhead <= 0.15:  # 15% memory overhead target
            print(f"   ✅ Memory overhead within target: {results.memory_overhead:.2%} <= 15%")
        else:
            print(f"   ⚠️  Memory overhead exceeds target: {results.memory_overhead:.2%} > 15%")
        
        if results.convergence_rate >= 0.8:  # 80% convergence rate target
            print(f"   ✅ Good convergence rate: {results.convergence_rate:.2%} >= 80%")
        else:
            print(f"   ⚠️  Low convergence rate: {results.convergence_rate:.2%} < 80%")
        
        print("="*80)
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of all evaluation runs"""
        if not self.results_history:
            return {}
        
        latest = self.results_history[-1]
        return {
            'latest_improvement': latest.improvement,
            'latest_search_score': latest.avg_search_score,
            'latest_convergence_rate': latest.convergence_rate,
            'total_evaluations': len(self.results_history),
            'best_improvement': max(r.improvement for r in self.results_history),
            'avg_improvement': np.mean([r.improvement for r in self.results_history])
        }


def create_swe_search_evaluator(model, config, device='cuda') -> SWESearchEvaluator:
    """Factory function to create SWE-Search evaluator"""
    return SWESearchEvaluator(model, config, device)


# Utility functions for integration with existing evaluation pipeline
def enhanced_evaluate_function(config, train_state, eval_loader, eval_metadata, rank, world_size):
    """Enhanced evaluation function that can be used as drop-in replacement"""
    
    # Create SWE-Search evaluator
    evaluator = create_swe_search_evaluator(train_state.model, config)
    
    # Run enhanced evaluation
    results = evaluator.evaluate_with_swe_search(
        eval_loader, 
        compare_baseline=True,
        save_detailed_results=(rank == 0)  # Only save on rank 0
    )
    
    # Return metrics in format compatible with existing evaluation
    metrics = {
        'eval/accuracy': results.enhanced_accuracy,
        'eval/base_accuracy': results.base_accuracy,
        'eval/improvement': results.improvement,
        'eval/swe_search_score': results.avg_search_score,
        'eval/search_convergence_rate': results.convergence_rate,
        'eval/search_efficiency': results.search_efficiency,
        'eval/memory_overhead': results.memory_overhead,
    }
    
    return metrics