#!/usr/bin/env python3
"""
Quick evaluation script for any checkpoint - lightweight version
"""

import torch
import json
import time
from pathlib import Path
import sys

# Add project root to path  
sys.path.append(str(Path(__file__).parent.parent))

def quick_checkpoint_analysis(checkpoint_path: str):
    """Quick analysis of checkpoint without full model loading"""
    print(f"🔍 Quick Checkpoint Analysis: {checkpoint_path}")
    print("=" * 50)
    
    try:
        # Load checkpoint metadata only
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract key information
        epoch = checkpoint.get('epoch', 'Unknown')
        loss = checkpoint.get('loss', 'Unknown')
        metrics = checkpoint.get('metrics', {})
        optimizer_state = 'optimizer_state_dict' in checkpoint
        model_state = 'model_state_dict' in checkpoint
        
        print(f"📊 Checkpoint Info:")
        print(f"   Epoch: {epoch}")
        print(f"   Loss: {loss}")
        print(f"   Has model state: {'✅' if model_state else '❌'}")
        print(f"   Has optimizer state: {'✅' if optimizer_state else '❌'}")
        
        if metrics:
            print(f"\n📈 Training Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        # Model state analysis
        if model_state:
            model_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in model_dict.values() if isinstance(p, torch.Tensor))
            model_size_mb = sum(p.numel() * p.element_size() for p in model_dict.values() if isinstance(p, torch.Tensor)) / (1024**2)
            
            print(f"\n🏗️ Model Architecture:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Model size: {model_size_mb:.1f} MB")
            print(f"   Layer count: {len([k for k in model_dict.keys() if 'weight' in k])}")
        
        # File info
        file_size_mb = Path(checkpoint_path).stat().st_size / (1024**2)
        print(f"\n💾 File Info:")
        print(f"   Checkpoint size: {file_size_mb:.1f} MB")
        print(f"   Created: {time.ctime(Path(checkpoint_path).stat().st_mtime)}")
        
        return {
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "total_params": total_params if model_state else None,
            "model_size_mb": model_size_mb if model_state else None,
            "file_size_mb": file_size_mb
        }
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return None

def compare_checkpoints(checkpoint_paths: list):
    """Compare multiple checkpoints"""
    print("🔄 Comparing Checkpoints")
    print("=" * 30)
    
    results = []
    for path in checkpoint_paths:
        result = quick_checkpoint_analysis(path)
        if result:
            result['path'] = path
            results.append(result)
        print()
    
    if len(results) > 1:
        print("📊 Comparison Summary:")
        print("-" * 20)
        for i, result in enumerate(results):
            print(f"Checkpoint {i+1}: Epoch {result['epoch']}, Loss {result['loss']}")
        
        # Find best and worst
        if all('loss' in r and isinstance(r['loss'], (int, float)) for r in results):
            best = min(results, key=lambda x: x['loss'])
            worst = max(results, key=lambda x: x['loss'])
            print(f"\n🏆 Best: {Path(best['path']).name} (Loss: {best['loss']:.4f})")
            print(f"📉 Worst: {Path(worst['path']).name} (Loss: {worst['loss']:.4f})")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick checkpoint evaluation")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint file paths")
    parser.add_argument("--compare", action="store_true", help="Compare multiple checkpoints")
    
    args = parser.parse_args()
    
    if args.compare and len(args.checkpoints) > 1:
        compare_checkpoints(args.checkpoints)
    else:
        for checkpoint in args.checkpoints:
            quick_checkpoint_analysis(checkpoint)
            print()

if __name__ == "__main__":
    main()