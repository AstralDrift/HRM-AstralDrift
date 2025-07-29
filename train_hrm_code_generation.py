#!/usr/bin/env python3
"""
HRM Code Generation Training Script

Train HRM on real code generation tasks using our sophisticated dataset format
that preserves multi-agent coordination, tool workflows, and domain analysis.

This is completely different from puzzle-based training - we're training on
real-world software engineering tasks for disruptive code generation performance.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig, create_code_generation_dataloader
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.losses import ACTSWESearchLossHead
from utils.device import get_device


class CodeGenerationTrainer:
    """Trainer for HRM code generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        
        print(f"🚀 Initializing HRM Code Generation Trainer")
        print(f"   Device: {self.device}")
        print(f"   Dataset: {config['data_path']}")
        
        # Initialize dataset
        self.dataset_config = CodeGenerationDatasetConfig(
            dataset_path=config['data_path'],
            max_input_length=512,  # Half of total sequence
            max_output_length=512,  # Half of total sequence  
            include_coordination_data=True,
            include_metadata=True
        )
        
        # Create dataloaders
        self.train_loader, self.dataset_metadata = create_code_generation_dataloader(
            self.dataset_config,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        print(f"   Loaded {self.dataset_metadata.num_instances} instances")
        print(f"   Vocab size: {self.dataset_metadata.vocab_size}")
        print(f"   Domains: {self.dataset_metadata.num_domains}")
        print(f"   Languages: {self.dataset_metadata.num_languages}")
        print(f"   Avg complexity: {self.dataset_metadata.average_complexity:.3f}")
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = ACTSWESearchLossHead(
            self.model,
            loss_type=config.get('loss_type', 'softmax_cross_entropy'),
            swe_search_weight=config.get('swe_search_weight', 0.3),
            reverse_learning_weight=config.get('reverse_learning_weight', 0.2)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.1)
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Initialize W&B if enabled
        if config.get('use_wandb', False):
            self._init_wandb()
    
    def _create_model(self) -> nn.Module:
        """Create HRM model for code generation"""
        
        model_config = HierarchicalReasoningModel_ACTV1Config(
            # Dataset-specific config
            vocab_size=self.dataset_metadata.vocab_size,
            seq_len=1024,  # 512 input + 512 output
            batch_size=self.config['batch_size'],
            
            # Model architecture 
            hidden_size=self.config.get('hidden_size', 768),
            num_heads=self.config.get('num_heads', 12),
            H_layers=self.config.get('H_layers', 6),
            L_layers=self.config.get('L_layers', 6),
            H_cycles=self.config.get('H_cycles', 3),
            L_cycles=self.config.get('L_cycles', 4),
            
            # Code generation specific
            enable_swe_search=True,
            enable_reverse_learning=True,
            
            # ACT settings
            halt_max_steps=self.config.get('halt_max_steps', 20),
            halt_exploration_prob=self.config.get('halt_exploration_prob', 0.15),
            
            # Other settings
            puzzle_emb_ndim=0,  # No puzzle embeddings for code
            num_puzzle_identifiers=self.dataset_metadata.num_instances,
            pos_encodings=self.config.get('pos_encodings', 'rope'),
            expansion=self.config.get('expansion', 4.0)
        )
        
        model = HierarchicalReasoningModel_ACTV1(model_config.__dict__)
        
        print(f"   Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        return model
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('project_name', 'hrm-code-generation'),
            name=self.config.get('run_name', f'run-{int(time.time())}'),
            config=self.config,
            mode=self.config.get('wandb_mode', 'online')
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Initialize carry state
            carry = self.loss_fn.initial_carry(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            new_carry, loss, metrics, outputs, all_halted = self.loss_fn(
                return_keys=['logits'],
                carry=carry,
                batch=batch
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            epoch_losses.append(loss.item())
            
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'accuracy': f"{metrics.get('accuracy', 0):.3f}",
                'steps': f"{metrics.get('steps', 0):.1f}"
            })
            
            self.global_step += 1
            
            # Log to W&B
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/global_step': self.global_step
                }
                for key, value in metrics.items():
                    log_dict[f'train/{key}'] = value.item() if torch.is_tensor(value) else value
                
                wandb.log(log_dict)
        
        # Calculate epoch metrics
        epoch_results = {
            'loss': sum(epoch_losses) / len(epoch_losses),
        }
        
        for key, values in epoch_metrics.items():
            epoch_results[key] = sum(values) / len(values)
        
        return epoch_results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop"""
        
        print(f"\n🎯 Starting HRM Code Generation Training")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Device: {self.device}")
        
        # Create checkpoints directory
        checkpoint_dir = Path("checkpoints") / self.config.get('run_name', 'hrm-code-gen')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for epoch in range(self.config['epochs']):
                self.epoch = epoch
                
                # Train epoch
                epoch_results = self.train_epoch()
                
                # Print results
                print(f"\n📊 Epoch {epoch} Results:")
                for key, value in epoch_results.items():
                    print(f"   {key}: {value:.4f}")
                
                # Save checkpoint periodically
                if epoch % self.config.get('checkpoint_interval', 100) == 0:
                    checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
                    self.save_checkpoint(checkpoint_path)
                
                # Save best model
                if epoch_results['loss'] < self.best_loss:
                    self.best_loss = epoch_results['loss']
                    best_path = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(best_path)
                    print(f"🏆 New best model saved! Loss: {self.best_loss:.4f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            final_path = checkpoint_dir / "interrupted_model.pt" 
            self.save_checkpoint(final_path)
        
        print(f"\n✅ Training completed!")
        print(f"   Best loss: {self.best_loss:.4f}")
        print(f"   Checkpoints saved in: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train HRM on code generation tasks")
    
    parser.add_argument('--data-path', default='data/swe-smith-1k', help='Path to code generation dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=768, help='Model hidden size')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--run-name', default=None, help='Run name for logging')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'max_input_length': 1024,
        'max_output_length': 512,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name or f'hrm-code-gen-{int(time.time())}',
        'project_name': 'HRM-Code-Generation',
        'checkpoint_interval': 100,
        'weight_decay': 0.1,
        'swe_search_weight': 0.3,
        'reverse_learning_weight': 0.2,
        'H_cycles': 3,
        'L_cycles': 4,
        'halt_max_steps': 20
    }
    
    print("🚀 HRM Code Generation Training")
    print("=" * 50)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create trainer and start training
    trainer = CodeGenerationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()