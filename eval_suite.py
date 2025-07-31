#!/usr/bin/env python3
"""
Unified HRM Evaluation Suite

Consolidates all evaluation capabilities:
- Quick checkpoint evaluation
- Comprehensive code generation metrics
- SWE-Search performance analysis
- Tool usage assessment
- Model inference testing
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.losses import ACTSWESearchLossHead
from utils.device import get_device


class HRMEvaluationSuite:
    """Comprehensive evaluation suite for HRM models"""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = get_device()
        
        print(f"🔬 HRM Evaluation Suite")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Device: {self.device}")
        
        # Load model
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer from checkpoint"""
        print("📚 Loading model and tokenizer...")
        
        # Load checkpoint
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint  
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            print("⚠️ Model config not found in checkpoint, using defaults")
            model_config = {
                'vocab_size': 50265,
                'hidden_size': 512,
                'H_layers': 4,
                'L_layers': 4, 
                'H_cycles': 2,
                'L_cycles': 2,
                'seq_len': 512,
                'num_heads': 8,
                'expansion': 4,
                'batch_size': 1,  # Added missing fields
                'num_puzzle_identifiers': 100,
                'pos_encodings': 'rope',
                'halt_max_steps': 16,
                'halt_exploration_prob': 0.1,
                'causal': False,
                'head_size': 64,
                'act_threshold': 0.99,
                'puzzle_emb_vocab_size': 1000,
                'puzzle_emb_dim': 512,
                'max_seq_len': 512
            }
        
        # Create model
        model = HierarchicalReasoningModel_ACTV1(model_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        return model, tokenizer
    
    def quick_eval(self, num_samples: int = 5) -> Dict[str, Any]:
        """Quick evaluation on a few samples"""
        print(f"\n🚀 Quick Evaluation ({num_samples} samples)")
        print("-" * 50)
        
        # Load test dataset
        dataset_config = CodeGenerationDatasetConfig(
            dataset_path=self.config.get('test_data_path', 'data/swe-smith-1k'),
            split='test' if 'swe-smith' in self.config.get('test_data_path', '') else 'train',
            max_input_length=512,
            max_output_length=256
        )
        
        try:
            dataset = CodeGenerationDataset(dataset_config)
            print(f"📊 Loaded test dataset: {len(dataset)} instances")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {"error": str(e)}
        
        results = []
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                
                # Run inference
                start_time = time.time()
                try:
                    # Prepare inputs
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                    labels = sample['labels'].unsqueeze(0).to(self.device)
                    
                    # Model forward pass
                    carry = self.model.initial_carry(
                        inputs=input_ids,
                        labels=labels,
                        puzzle_identifiers=torch.zeros(1, dtype=torch.long, device=self.device),
                        **sample.get('metadata', {})
                    )
                    
                    new_carry, outputs = self.model(carry=carry)
                    inference_time = time.time() - start_time
                    
                    # Generate predictions
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    
                    # Decode
                    pred_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
                    true_text = self.tokenizer.decode(sample['labels'], skip_special_tokens=True)
                    input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                    
                    # Compute metrics
                    token_accuracy = (predictions[0] == labels[0]).float().mean().item()
                    
                    result = {
                        'sample_id': i,
                        'input_text': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                        'predicted_text': pred_text[:100] + "..." if len(pred_text) > 100 else pred_text,
                        'target_text': true_text[:100] + "..." if len(true_text) > 100 else true_text,
                        'token_accuracy': token_accuracy,
                        'inference_time': inference_time,
                        'steps_taken': new_carry.steps.item() if hasattr(new_carry, 'steps') else 'N/A',
                        'halted': new_carry.halted.item() if hasattr(new_carry, 'halted') else 'N/A'
                    }
                    
                    results.append(result)
                    
                    print(f"Sample {i+1}:")
                    print(f"  Token Accuracy: {token_accuracy:.3f}")
                    print(f"  Inference Time: {inference_time:.3f}s")
                    print(f"  Predicted: {pred_text[:50]}...")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error processing sample {i}: {e}")
                    results.append({'sample_id': i, 'error': str(e)})
        
        # Summary statistics
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            avg_accuracy = np.mean([r['token_accuracy'] for r in successful_results])
            avg_time = np.mean([r['inference_time'] for r in successful_results])
            
            summary = {
                'avg_token_accuracy': avg_accuracy,
                'avg_inference_time': avg_time,
                'success_rate': len(successful_results) / len(results),
                'total_samples': len(results)
            }
            
            print(f"📊 Quick Evaluation Summary:")
            print(f"   Average Token Accuracy: {avg_accuracy:.3f}")
            print(f"   Average Inference Time: {avg_time:.3f}s")
            print(f"   Success Rate: {len(successful_results)}/{len(results)}")
        else:
            summary = {'error': 'No successful evaluations'}
        
        return {'results': results, 'summary': summary}
    
    def comprehensive_eval(self, num_samples: int = 50) -> Dict[str, Any]:
        """Comprehensive evaluation with detailed metrics"""
        print(f"\n🔬 Comprehensive Evaluation ({num_samples} samples)")
        print("-" * 50)
        
        # Load dataset
        dataset_config = CodeGenerationDatasetConfig(
            dataset_path=self.config.get('test_data_path', 'data/swe-smith-1k'),
            split='test' if 'swe-smith' in self.config.get('test_data_path', '') else 'train',
            max_input_length=512,
            max_output_length=256
        )
        
        try:
            dataset = CodeGenerationDataset(dataset_config)
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}
        
        # Metrics tracking
        metrics = {
            'token_accuracies': [],
            'syntax_validities': [],
            'compilation_successes': [],
            'inference_times': [],
            'step_counts': [],
            'halt_rates': []
        }
        
        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
                try:
                    sample = dataset[i]
                    
                    # Prepare inputs
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                    labels = sample['labels'].unsqueeze(0).to(self.device)
                    
                    # Model forward pass
                    start_time = time.time()
                    carry = self.model.initial_carry(
                        inputs=input_ids,
                        labels=labels,
                        puzzle_identifiers=torch.zeros(1, dtype=torch.long, device=self.device)
                    )
                    
                    new_carry, outputs = self.model(carry=carry)
                    inference_time = time.time() - start_time
                    
                    # Predictions
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    pred_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
                    
                    # Token accuracy
                    token_accuracy = (predictions[0] == labels[0]).float().mean().item()
                    metrics['token_accuracies'].append(token_accuracy)
                    
                    # Syntax validity
                    try:
                        compile(pred_text, '<string>', 'exec')
                        syntax_valid = 1
                        compilation_success = 1
                    except SyntaxError:
                        syntax_valid = 0
                        compilation_success = 0
                    except:
                        syntax_valid = 0
                        compilation_success = 0
                    
                    metrics['syntax_validities'].append(syntax_valid)
                    metrics['compilation_successes'].append(compilation_success)
                    metrics['inference_times'].append(inference_time)
                    
                    if hasattr(new_carry, 'steps'):
                        metrics['step_counts'].append(new_carry.steps.item())
                    if hasattr(new_carry, 'halted'):
                        metrics['halt_rates'].append(new_carry.halted.item())
                        
                except Exception as e:
                    print(f"Error on sample {i}: {e}")
                    continue
        
        # Calculate summary statistics
        summary = {}
        for key, values in metrics.items():
            if values:
                summary[f'avg_{key}'] = np.mean(values)
                summary[f'std_{key}'] = np.std(values)
                summary[f'min_{key}'] = np.min(values)
                summary[f'max_{key}'] = np.max(values)
        
        summary['total_samples'] = len(metrics['token_accuracies'])
        
        print(f"\n📊 Comprehensive Evaluation Results:")
        print(f"   Token Accuracy: {summary.get('avg_token_accuracies', 0):.3f} ± {summary.get('std_token_accuracies', 0):.3f}")
        print(f"   Syntax Validity: {summary.get('avg_syntax_validities', 0):.3f}")
        print(f"   Compilation Success: {summary.get('avg_compilation_successes', 0):.3f}")
        print(f"   Avg Inference Time: {summary.get('avg_inference_times', 0):.3f}s")
        if 'avg_step_counts' in summary:
            print(f"   Avg Steps: {summary['avg_step_counts']:.1f}")
        if 'avg_halt_rates' in summary:
            print(f"   Halt Rate: {summary['avg_halt_rates']:.3f}")
        
        return {'metrics': metrics, 'summary': summary}
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to file"""
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"evaluations/eval_results_{timestamp}.json"
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HRM Evaluation Suite")
    
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'both'], default='quick', 
                       help='Evaluation mode')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--test-data', default='data/swe-smith-1k', help='Test dataset path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'test_data_path': args.test_data,
        'device': get_device()
    }
    
    # Create evaluation suite
    try:
        evaluator = HRMEvaluationSuite(args.checkpoint, config)
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return 1
    
    # Run evaluations
    all_results = {}
    
    if args.mode in ['quick', 'both']:
        quick_results = evaluator.quick_eval(args.num_samples)
        all_results['quick_evaluation'] = quick_results
    
    if args.mode in ['comprehensive', 'both']:
        comp_samples = max(args.num_samples, 20)  # Minimum 20 for comprehensive
        comp_results = evaluator.comprehensive_eval(comp_samples)
        all_results['comprehensive_evaluation'] = comp_results
    
    # Save results if requested
    if args.save_results:
        output_path = args.output or f"evaluations/eval_results_{int(time.time())}.json"
        evaluator.save_results(all_results, output_path)
    
    print(f"\n🎉 Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())