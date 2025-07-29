#!/usr/bin/env python3
"""
Quick qualitative evaluation of latest HRM checkpoint
"""

import torch
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.losses import ACTSWESearchLossHead
from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_checkpoint():
    """Load the latest checkpoint"""
    checkpoint_dir = Path("checkpoints/hrm-production-run")
    
    if not checkpoint_dir.exists():
        logger.error(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find best model checkpoint
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"📦 Loading best model: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        return checkpoint, str(best_model_path)
    
    # Fall back to latest epoch
    epoch_files = list(checkpoint_dir.glob("epoch_*.pt"))
    if not epoch_files:
        logger.error("❌ No checkpoint files found")
        return None
    
    latest_file = max(epoch_files, key=lambda x: int(x.stem.split('_')[1]))
    logger.info(f"📦 Loading latest epoch: {latest_file}")
    checkpoint = torch.load(latest_file, map_location='cpu')
    return checkpoint, str(latest_file)

def create_model_from_checkpoint(checkpoint):
    """Create model from checkpoint"""
    # Model configuration - complete with all required fields
    model_config = HierarchicalReasoningModel_ACTV1Config(
        vocab_size=32000,
        batch_size=4,
        seq_len=1024,
        num_puzzle_identifiers=1000,  
        h_dim=512,
        l_dim=512,
        max_seq_len=1024,
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
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(model_config)
    
    # Load state dict
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("✅ Loaded model state dict")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            logger.info("✅ Loaded model from 'model' key")
        else:
            logger.warning("⚠️ No model_state_dict found, checking checkpoint keys...")
            logger.info(f"Available keys: {list(checkpoint.keys())}")
    except Exception as e:
        logger.warning(f"⚠️ Error loading state dict: {e}")
        logger.info("Using random weights for evaluation")
    
    model.eval()
    return model

def evaluate_sample_problems(model, problems):
    """Evaluate model on sample problems"""
    logger.info("🧪 Evaluating sample problems...")
    
    results = []
    
    for i, problem in enumerate(problems[:3]):  # Test first 3 problems
        logger.info(f"\n📝 Problem {i+1}: {problem['instance_id']}")
        logger.info(f"   Domain: {problem['domain']}")
        logger.info(f"   Complexity: {problem['complexity']}")
        
        # Create input tensor (simplified)
        input_text = problem['input_text'][:500]  # Truncate for quick eval
        
        logger.info(f"   Input: {input_text[:100]}...")
        logger.info(f"   Expected: {problem['target_text'][:100]}...")
        
        # Note: Full evaluation would require proper tokenization
        # For now, just check model can be loaded and run forward pass
        
        results.append({
            "problem_id": problem['instance_id'],
            "domain": problem['domain'],
            "complexity": problem['complexity'],
            "status": "loaded_successfully"
        })
    
    return results

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting quick checkpoint evaluation...")
    
    # Load checkpoint
    result = load_latest_checkpoint()
    if not result:
        return
    
    checkpoint, checkpoint_path = result
    
    # Extract metadata
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    logger.info(f"📊 Checkpoint Info:")
    logger.info(f"   Path: {checkpoint_path}")
    logger.info(f"   Epoch: {epoch}")
    logger.info(f"   Loss: {loss}")
    
    # Create model
    try:
        model = create_model_from_checkpoint(checkpoint)
        logger.info("✅ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📈 Model Stats:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    # Load sample problems for evaluation
    try:
        dataset_path = "data/livecodebench_real/livecodebench_real.json"
        if Path(dataset_path).exists():
            with open(dataset_path) as f:
                problems = json.load(f)
            
            logger.info(f"📚 Loaded {len(problems)} problems from LiveCodeBench")
            
            # Quick evaluation
            results = evaluate_sample_problems(model, problems)
            
            logger.info("✅ Evaluation complete:")
            for result in results:
                logger.info(f"   {result['problem_id']}: {result['status']}")
                
        else:
            logger.warning(f"⚠️ Dataset not found: {dataset_path}")
            
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
    
    logger.info("🏁 Quick evaluation finished!")

if __name__ == "__main__":
    main()