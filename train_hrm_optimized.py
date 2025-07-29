#!/usr/bin/env python3
"""
Optimized HRM Code Generation Training Script

This addresses the performance and monitoring issues identified:
- Fixed MPS compatibility issues
- Added proper learning rate scheduling  
- Enhanced progress monitoring
- Optimized batch processing
- Better error handling and logging
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig, create_code_generation_dataloader
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.losses import ACTSWESearchLossHead
from utils.device import get_device


class OptimizedCodeGenerationTrainer:
    """Optimized trainer for HRM code generation with better monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"🚀 Initializing Optimized HRM Code Generation Trainer")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Dataset: {config['data_path']}")
        
        # Initialize dataset with MPS-optimized settings
        self.dataset_config = CodeGenerationDatasetConfig(
            dataset_path=config['data_path'],
            max_input_length=512,
            max_output_length=512,
            include_coordination_data=True,
            include_metadata=True
        )
        
        # Create dataloaders with device-specific optimizations
        pin_memory = self.device == 'cuda'  # Only use pin_memory for CUDA
        num_workers = 0 if self.device == 'mps' else 2  # MPS works better with single-threaded
        
        self.train_loader, self.dataset_metadata = create_code_generation_dataloader(
            self.dataset_config,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers
        )
        
        # Update dataloader to fix MPS issues
        self.train_loader.pin_memory = pin_memory
        
        self.logger.info(f"   Loaded {self.dataset_metadata.num_instances} instances")
        self.logger.info(f"   Vocab size: {self.dataset_metadata.vocab_size}")
        self.logger.info(f"   Domains: {self.dataset_metadata.num_domains}")
        self.logger.info(f"   Languages: {self.dataset_metadata.num_languages}")
        self.logger.info(f"   Avg complexity: {self.dataset_metadata.average_complexity:.3f}")
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"   Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
        
        # Initialize loss function
        self.loss_fn = ACTSWESearchLossHead(
            self.model,
            loss_type=config.get('loss_type', 'softmax_cross_entropy'),
            swe_search_weight=config.get('swe_search_weight', 0.3),
            reverse_learning_weight=config.get('reverse_learning_weight', 0.2)
        )
        
        # Initialize optimizer with better settings
        self.optimizer = torch.optim.AdamW(  # AdamW instead of Adam
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.1),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_t0', 100),
            T_mult=2,
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Performance monitoring
        self.batch_times = []
        self.loss_history = []
        
        # Initialize W&B if enabled
        if config.get('use_wandb', False):
            self._init_wandb()
    
    def _setup_logging(self):
        """Setup proper logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('hrm_training.log')
            ]
        )
        self.logger = logging.getLogger('HRMTrainer')
    
    def _create_model(self) -> nn.Module:
        """Create HRM model with optimized configuration"""
        
        model_config = HierarchicalReasoningModel_ACTV1Config(
            # Dataset-specific config
            vocab_size=self.dataset_metadata.vocab_size,
            seq_len=1024,  # 512 input + 512 output
            batch_size=self.config['batch_size'],
            
            # Optimized architecture for faster training
            hidden_size=self.config.get('hidden_size', 512),  # Smaller for faster training
            num_heads=self.config.get('num_heads', 8),
            H_layers=self.config.get('H_layers', 4),  # Reduced layers
            L_layers=self.config.get('L_layers', 4),
            H_cycles=self.config.get('H_cycles', 2),  # Reduced cycles initially
            L_cycles=self.config.get('L_cycles', 3),
            
            # Code generation specific
            enable_swe_search=True,
            enable_reverse_learning=True,
            
            # ACT settings optimized for speed
            halt_max_steps=self.config.get('halt_max_steps', 16),
            halt_exploration_prob=self.config.get('halt_exploration_prob', 0.1),
            
            # Other settings
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=self.dataset_metadata.num_instances,
            pos_encodings=self.config.get('pos_encodings', 'rope'),
            expansion=self.config.get('expansion', 2.0),  # Smaller expansion
            
            # Set forward dtype for MPS compatibility
            forward_dtype='float32'
        )
        
        model = HierarchicalReasoningModel_ACTV1(model_config.__dict__)
        return model
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('project_name', 'hrm-code-generation'),
            name=self.config.get('run_name', f'optimized-run-{int(time.time())}'),
            config=self.config,
            mode=self.config.get('wandb_mode', 'online')
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with enhanced monitoring"""
        
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        batch_times = []
        
        # Enhanced progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch}",
            ncols=120,  # Wider progress bar
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move batch to device efficiently
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Initialize carry state
            carry = self.loss_fn.initial_carry(batch)
            
            # Forward pass with gradient accumulation support
            self.optimizer.zero_grad()
            
            try:
                new_carry, loss, metrics, outputs, all_halted = self.loss_fn(
                    return_keys=['logits'],
                    carry=carry,
                    batch=batch
                )
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at step {self.global_step}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping with adaptive threshold
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                epoch_losses.append(loss.item())
                
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
                
                # Calculate running averages
                recent_losses = epoch_losses[-10:] if len(epoch_losses) >= 10 else epoch_losses
                recent_times = batch_times[-10:] if len(batch_times) >= 10 else batch_times
                
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_time = sum(recent_times) / len(recent_times)
                
                # Enhanced progress display
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'acc': f"{metrics.get('accuracy', 0):.3f}",
                    'grad': f"{grad_norm:.2f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'time': f"{avg_time:.1f}s"
                })
                
                self.global_step += 1
                
                # Enhanced logging
                if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/grad_norm': grad_norm,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/batch_time': batch_time,
                        'train/global_step': self.global_step
                    }
                    for key, value in metrics.items():
                        log_dict[f'train/{key}'] = value.item() if torch.is_tensor(value) else value
                    
                    wandb.log(log_dict)
                
                # Early stopping check for debugging
                if batch_idx >= self.config.get('max_batches_per_epoch', float('inf')):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        if epoch_losses:
            epoch_results = {
                'loss': sum(epoch_losses) / len(epoch_losses),
                'avg_batch_time': sum(batch_times) / len(batch_times),
                'total_batches': len(epoch_losses)
            }
            
            for key, values in epoch_metrics.items():
                if values:
                    epoch_results[key] = sum(values) / len(values)
        else:
            epoch_results = {'loss': float('inf'), 'avg_batch_time': 0, 'total_batches': 0}
        
        return epoch_results
    
    def save_checkpoint(self, path: str, additional_info: Dict = None):
        """Save model checkpoint with additional monitoring info"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'dataset_metadata': self.dataset_metadata.__dict__
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        self.logger.info(f"✅ Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop with enhanced monitoring"""
        
        self.logger.info(f"\n🎯 Starting Optimized HRM Code Generation Training")
        self.logger.info(f"   Epochs: {self.config['epochs']}")
        self.logger.info(f"   Batch size: {self.config['batch_size']}")
        self.logger.info(f"   Learning rate: {self.config['learning_rate']}")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Model size: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        
        # Create checkpoints directory
        checkpoint_dir = Path("checkpoints") / self.config.get('run_name', 'hrm-optimized')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            training_start_time = time.time()
            
            for epoch in range(self.config['epochs']):
                self.epoch = epoch
                epoch_start_time = time.time()
                
                # Train epoch
                epoch_results = self.train_epoch()
                epoch_time = time.time() - epoch_start_time
                
                # Add timing info
                epoch_results['epoch_time'] = epoch_time
                self.training_history.append(epoch_results)
                
                # Enhanced results display
                self.logger.info(f"\n📊 Epoch {epoch} Results:")
                self.logger.info(f"   Loss: {epoch_results['loss']:.4f}")
                self.logger.info(f"   Avg batch time: {epoch_results.get('avg_batch_time', 0):.2f}s")
                self.logger.info(f"   Total batches: {epoch_results.get('total_batches', 0)}")
                self.logger.info(f"   Epoch time: {epoch_time:.1f}s")
                
                for key, value in epoch_results.items():
                    if key not in ['loss', 'avg_batch_time', 'total_batches', 'epoch_time']:
                        self.logger.info(f"   {key}: {value:.4f}")
                
                # Save checkpoint periodically
                if epoch % self.config.get('checkpoint_interval', 10) == 0:
                    checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
                    self.save_checkpoint(checkpoint_path, {'epoch_results': epoch_results})
                
                # Save best model
                if epoch_results['loss'] < self.best_loss:
                    self.best_loss = epoch_results['loss']
                    best_path = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(best_path, {'epoch_results': epoch_results})
                    self.logger.info(f"🏆 New best model saved! Loss: {self.best_loss:.4f}")
                
                # Performance analysis
                if epoch > 0 and epoch % 5 == 0:
                    self._analyze_training_performance()
        
        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Training interrupted by user")
            final_path = checkpoint_dir / "interrupted_model.pt"
            self.save_checkpoint(final_path, {'interrupted_at_epoch': self.epoch})
        
        training_time = time.time() - training_start_time
        self.logger.info(f"\n✅ Training completed!")
        self.logger.info(f"   Best loss: {self.best_loss:.4f}")
        self.logger.info(f"   Total training time: {training_time/3600:.2f} hours")
        self.logger.info(f"   Checkpoints saved in: {checkpoint_dir}")
    
    def _analyze_training_performance(self):
        """Analyze training performance and suggest optimizations"""
        if len(self.training_history) < 5:
            return
        
        recent_losses = [h['loss'] for h in self.training_history[-5:]]
        recent_times = [h.get('avg_batch_time', 0) for h in self.training_history[-5:]]
        
        # Loss trend analysis
        loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
        avg_batch_time = sum(recent_times) / len(recent_times)
        
        self.logger.info(f"\n📈 Performance Analysis:")
        self.logger.info(f"   Loss trend (last 5 epochs): {loss_trend*100:.1f}%")
        self.logger.info(f"   Average batch time: {avg_batch_time:.2f}s")
        
        # Suggestions
        if loss_trend > -0.01:  # Less than 1% improvement
            self.logger.info("   💡 Consider: increasing learning rate or adjusting architecture")
        
        if avg_batch_time > 5.0:
            self.logger.info("   💡 Consider: reducing batch size or model complexity")


def main():
    parser = argparse.ArgumentParser(description="Optimized HRM code generation training")
    
    parser.add_argument('--data-path', default='data/swe-smith-1k', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=512, help='Model hidden size')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--run-name', default=None, help='Run name')
    parser.add_argument('--max-batches-per-epoch', type=int, default=None, help='Limit batches per epoch for testing')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name or f'hrm-optimized-{int(time.time())}',
        'project_name': 'HRM-Code-Generation-Optimized',
        'checkpoint_interval': 10,
        'weight_decay': 0.1,
        'swe_search_weight': 0.2,  # Reduced for faster training
        'reverse_learning_weight': 0.1,
        'H_cycles': 2,
        'L_cycles': 3,
        'halt_max_steps': 16,
        'scheduler_t0': 50,
        'max_batches_per_epoch': args.max_batches_per_epoch
    }
    
    print("🚀 Optimized HRM Code Generation Training")
    print("=" * 60)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create trainer and start training
    trainer = OptimizedCodeGenerationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()