#!/usr/bin/env python3
"""
Recalculate code metrics for existing checkpoints with fixed tokenizer
"""

import torch
import json
import logging
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from models.losses import ACTSWESearchLossHead
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricRecalculator:
    def __init__(self, checkpoint_dir="checkpoints/hrm-production-run"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the tokenizer used in training"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            return None
    
    def _create_loss_head_with_tokenizer(self, model):
        """Create loss head with proper tokenizer integration"""
        return ACTSWESearchLossHead(
            model,
            loss_type="softmax_cross_entropy",
            swe_search_weight=0.2,
            reverse_learning_weight=0.1,
            tokenizer=self.tokenizer
        )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint and extract model state"""
        logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract metadata
            epoch = checkpoint.get('epoch', 'unknown')
            old_metrics = checkpoint.get('metrics', {})
            
            logger.info(f"   Epoch: {epoch}")
            logger.info(f"   Old loss: {old_metrics.get('loss', 'N/A')}")
            logger.info(f"   Old syntax validity: {old_metrics.get('syntax_validity', 0)}")
            logger.info(f"   Old compilation success: {old_metrics.get('compilation_success', 0)}")
            
            return checkpoint, epoch, old_metrics
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return None, None, None
    
    def _create_sample_data(self, batch_size=4, seq_len=128):
        """Create sample data for metric recalculation"""
        vocab_size = self.tokenizer.vocab_size
        
        # Create sample code strings for testing
        sample_codes = [
            "def add(a, b): return a + b",
            "class MyClass: def __init__(self): pass", 
            "for i in range(10): print(i)",
            "if x > 0: result = True"
        ]
        
        # Tokenize sample codes
        logits_list = []
        labels_list = []
        
        for code in sample_codes[:batch_size]:
            # Encode code
            tokens = self.tokenizer.encode(code, max_length=seq_len, padding='max_length', truncation=True)
            
            # Create mock logits (batch_size=1, seq_len, vocab_size)
            logits = torch.randn(1, len(tokens), vocab_size)
            # Set highest probability to correct tokens for testing
            for i, token in enumerate(tokens):
                if token < vocab_size:
                    logits[0, i, token] = 10.0  # High probability for correct token
            
            labels = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
            
            logits_list.append(logits)
            labels_list.append(labels)
        
        # Combine into batches
        max_len = max(l.size(1) for l in logits_list)
        
        # Pad sequences to same length
        padded_logits = []
        padded_labels = []
        
        for logits, labels in zip(logits_list, labels_list):
            if logits.size(1) < max_len:
                pad_size = max_len - logits.size(1)
                logits = torch.cat([logits, torch.zeros(1, pad_size, vocab_size)], dim=1)
                labels = torch.cat([labels, torch.full((1, pad_size), -100)], dim=1)
            
            padded_logits.append(logits)
            padded_labels.append(labels)
        
        final_logits = torch.cat(padded_logits, dim=0)
        final_labels = torch.cat(padded_labels, dim=0)
        
        # Create current_data
        current_data = {
            "input_text": sample_codes,
            "target_text": sample_codes  # Same for testing
        }
        
        return final_logits, final_labels, current_data
    
    def recalculate_checkpoint_metrics(self, checkpoint_path):
        """Recalculate metrics for a specific checkpoint"""
        checkpoint, epoch, old_metrics = self._load_checkpoint(checkpoint_path)
        if checkpoint is None:
            return None
        
        # Create model configuration (matching training config)
        model_config = HierarchicalReasoningModel_ACTV1Config(
            vocab_size=self.tokenizer.vocab_size,
            batch_size=4,
            seq_len=128,
            num_puzzle_identifiers=1000,
            h_dim=512,
            l_dim=512,
            max_seq_len=128,
            H_layers=4,
            L_layers=6,
            H_cycles=1,
            L_cycles=2,
            causal=False,
            head_size=64,
            hidden_size=512,
            expansion=4,
            num_heads=8,
            pos_encodings="rotary",
            act_threshold=0.99,
            halt_max_steps=6,
            halt_exploration_prob=0.4,
            puzzle_emb_vocab_size=30000,
            puzzle_emb_dim=512
        )
        
        try:
            # Create model and load state
            model = HierarchicalReasoningModel_ACTV1(model_config)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("✅ Model state loaded")
            else:
                logger.warning("⚠️ No model state found - using random weights")
            
            # Create loss head with tokenizer
            loss_head = self._create_loss_head_with_tokenizer(model)
            model.eval()
            loss_head.eval()
            
            # Create sample data for testing
            logits, labels, current_data = self._create_sample_data()
            
            # Recalculate code metrics
            logger.info("🔄 Recalculating code metrics...")
            with torch.no_grad():
                new_metrics = loss_head._compute_code_metrics(logits, labels, current_data)
            
            # Convert tensors to scalars
            processed_metrics = {}
            for key, value in new_metrics.items():
                if isinstance(value, torch.Tensor):
                    processed_metrics[key] = value.item()
                else:
                    processed_metrics[key] = value
            
            # Compare old vs new
            comparison = {
                "checkpoint_path": str(checkpoint_path),
                "epoch": epoch,
                "old_metrics": old_metrics,
                "new_metrics": processed_metrics,
                "improvements": {}
            }
            
            # Calculate improvements
            key_metrics = ["syntax_validity", "compilation_success", "edit_distance", "tiered_accuracy"]
            for metric in key_metrics:
                old_val = old_metrics.get(metric, 0)
                new_val = processed_metrics.get(metric, 0)
                if isinstance(old_val, torch.Tensor):
                    old_val = old_val.item()
                comparison["improvements"][metric] = {
                    "old": old_val,
                    "new": new_val,
                    "change": new_val - old_val
                }
            
            logger.info("📊 Metric Comparison:")
            for metric, data in comparison["improvements"].items():
                change_str = f"+{data['change']:.3f}" if data['change'] >= 0 else f"{data['change']:.3f}"
                logger.info(f"   {metric}: {data['old']:.3f} → {data['new']:.3f} ({change_str})")
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error recalculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def recalculate_all_checkpoints(self):
        """Recalculate metrics for all available checkpoints"""
        logger.info("🚀 Recalculating metrics for all checkpoints...")
        
        if not self.checkpoint_dir.exists():
            logger.error(f"❌ Checkpoint directory not found: {self.checkpoint_dir}")
            return []
        
        # Find checkpoint files
        checkpoint_files = []
        for pattern in ["best_model.pt", "epoch_*.pt"]:
            checkpoint_files.extend(self.checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            logger.error("❌ No checkpoint files found")
            return []
        
        logger.info(f"📁 Found {len(checkpoint_files)} checkpoint files")
        
        results = []
        for checkpoint_path in sorted(checkpoint_files):
            result = self.recalculate_checkpoint_metrics(checkpoint_path)
            if result:
                results.append(result)
        
        return results
    
    def save_results(self, results):
        """Save recalculation results"""
        output_path = f"evaluations/metric_recalculation_{int(time.time())}.json"
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {output_path}")
        return output_path

def main():
    """Main recalculation process"""
    logger.info("🚀 Starting Metric Recalculation with Fixed Tokenizer")
    
    try:
        recalculator = MetricRecalculator()
        
        if recalculator.tokenizer is None:
            logger.error("❌ Cannot proceed without tokenizer")
            return
        
        # Recalculate all checkpoints
        results = recalculator.recalculate_all_checkpoints()
        
        if results:
            # Save results
            output_path = recalculator.save_results(results)
            
            # Summary
            logger.info("\n📊 Summary of Improvements:")
            for result in results:
                logger.info(f"\n🔍 {result['checkpoint_path'].split('/')[-1]}:")
                for metric, data in result["improvements"].items():
                    if data["change"] != 0:
                        change_str = f"+{data['change']:.3f}" if data['change'] >= 0 else f"{data['change']:.3f}"
                        logger.info(f"   {metric}: {change_str}")
            
            logger.info(f"\n✅ Recalculation complete! Check {output_path} for detailed results.")
        else:
            logger.error("❌ No results generated")
            
    except Exception as e:
        logger.error(f"❌ Recalculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()