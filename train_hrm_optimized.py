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
from torch.cuda.amp import GradScaler, autocast
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available - training will continue without logging")
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from code_generation_dataset import CodeGenerationDataset, CodeGenerationDatasetConfig, create_code_generation_dataloader
from mixed_dataset_loader import MixedCodeGenerationDataset, MixedDatasetConfig, create_mixed_dataloader

# Model configuration constants
MODEL_SEQ_LEN = 1024  # 512 input + 512 output
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.losses import ACTSWESearchLossHead
from utils.device import get_device
from utils.error_handling import (
    robust_error_handler, ErrorRecoveryManager, safe_save_checkpoint,
    create_error_report, validate_dataset_path, TrainingError, MemoryError
)


class OptimizedCodeGenerationTrainer:
    """Optimized trainer for HRM code generation with better monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize error recovery manager
        self.error_manager = ErrorRecoveryManager(
            logger=self.logger,
            max_retries=3
        )
        
        self.logger.info(f"🚀 Initializing Optimized HRM Code Generation Trainer")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Dataset: {config['data_path']}")
        
        try:
            self._validate_config()
        except Exception as e:
            raise TrainingError(f"Invalid training configuration: {e}") from e
        
        # Initialize dataset - support both single and mixed datasets
        if config['data_path'] == 'mixed':
            # Mixed dataset configuration
            self.mixed_config = MixedDatasetConfig(
                swe_smith_path="data/swe-smith-1k",
                livecodebench_path="data/livecodebench_real",
                swe_smith_ratio=0.7,  # 70% SWE-Smith
                livecodebench_ratio=0.3,  # 30% LiveCodeBench
                max_input_length=512,
                max_output_length=512,
                batch_size=config['batch_size'],
                validation_split=0.1,
                shuffle=True
            )
            
            self.train_loader = create_mixed_dataloader(self.mixed_config, "train", MODEL_SEQ_LEN)
            self.val_loader = create_mixed_dataloader(self.mixed_config, "validation", MODEL_SEQ_LEN)
            
            # Create dummy metadata for compatibility
            from types import SimpleNamespace
            self.dataset_metadata = SimpleNamespace(
                num_instances=len(self.train_loader.dataset),
                vocab_size=self.train_loader.dataset.tokenizer.vocab_size,
                num_domains=5,  # Mixed domains
                num_languages=1,  # Mostly Python
                average_complexity=0.7,  # Mixed complexity
                pad_token_id=self.train_loader.dataset.tokenizer.pad_token_id
            )
            
            self.logger.info(f"🔄 Using mixed dataset mode")
            self.logger.info(f"   Train instances: {len(self.train_loader.dataset)}")
            self.logger.info(f"   Validation instances: {len(self.val_loader.dataset)}")
            
        else:
            # Single dataset configuration
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
            self.val_loader = None  # No validation for single dataset mode
            
            self.logger.info(f"📝 Using single dataset mode: {config['data_path']}")
        
        self.logger.info(f"   Loaded {self.dataset_metadata.num_instances} instances")
        self.logger.info(f"   Vocab size: {self.dataset_metadata.vocab_size}")
        self.logger.info(f"   Domains: {self.dataset_metadata.num_domains}")
        self.logger.info(f"   Languages: {self.dataset_metadata.num_languages}")
        self.logger.info(f"   Avg complexity: {self.dataset_metadata.average_complexity:.3f}")
        
        # Initialize model
        self.model = self._create_model_safely()
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"   Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
        
        # Initialize loss function with tokenizer integration
        self.loss_fn = ACTSWESearchLossHead(
            self.model,
            loss_type=config.get('loss_type', 'softmax_cross_entropy'),
            swe_search_weight=config.get('swe_search_weight', 0.3),
            reverse_learning_weight=config.get('reverse_learning_weight', 0.2),
            tokenizer=getattr(self.train_loader.dataset, 'tokenizer', None)
        )
        
        # Validate tokenizer integration
        if hasattr(self.train_loader.dataset, 'tokenizer'):
            self.logger.info("✅ Tokenizer successfully integrated for code metrics")
        else:
            self.logger.warning("⚠️ No tokenizer found in dataset - will use fallback")
        
        # Initialize optimizer with better settings
        self.optimizer = torch.optim.AdamW(  # AdamW instead of Adam
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.1),
            eps=1e-8
        )
        # Initialize mixed precision scaler
        self.use_mixed_precision = self.config.get('mixed_precision', True) and (
            self.device == 'cuda' or 
            (self.device == 'mps' and torch.__version__ >= '2.0')
        )
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
            self.logger.info("⚠️  Mixed precision training disabled (device compatibility)")
        
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
    
    def _validate_config(self) -> None:
        """Validate training configuration"""
        required_keys = ['data_path', 'epochs', 'batch_size', 'learning_rate']
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise TrainingError(f"Missing required config keys: {missing_keys}")
        
        # Validate data path
        if self.config['data_path'] != 'mixed':
            validate_dataset_path(self.config['data_path'])
        
        # Validate numeric values
        if self.config['batch_size'] <= 0:
            raise TrainingError(f"batch_size must be positive, got {self.config['batch_size']}")
        
        if self.config['epochs'] <= 0:
            raise TrainingError(f"epochs must be positive, got {self.config['epochs']}")
        
        if self.config['learning_rate'] <= 0:
            raise TrainingError(f"learning_rate must be positive, got {self.config['learning_rate']}")
    
    @robust_error_handler(max_retries=2)
    def _create_model_safely(self) -> nn.Module:
        """Create model with error handling"""
        try:
            self.error_manager.log_system_state()
            model = self._create_model()
            self.logger.info("✅ Model created successfully")
            return model
        except Exception as e:
            self.error_manager.record_error(e, {'operation': 'model_creation'})
            raise TrainingError(f"Failed to create model: {e}") from e
    
    def _create_model(self) -> nn.Module:
        """Create HRM model with optimized configuration"""
        
        model_config = HierarchicalReasoningModel_ACTV1Config(
            # Dataset-specific config
            vocab_size=self.dataset_metadata.vocab_size,
            seq_len=MODEL_SEQ_LEN,  # 512 input + 512 output
            batch_size=self.config['batch_size'],
            
            # Optimized architecture for faster training
            hidden_size=self.config.get('hidden_size', 512),  # Smaller for faster training
            num_heads=self.config.get('num_heads', 8),
            H_layers=self.config.get('H_layers', 4),  # Reduced layers
            L_layers=self.config.get('L_layers', 4),
            H_cycles=self.config.get('H_cycles', 1),  # Further reduced for halting
            L_cycles=self.config.get('L_cycles', 2),  # Reduced for halting
            
            # Code generation specific
            enable_swe_search=True,
            enable_reverse_learning=True,
            
            # Memory optimization features
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            mixed_precision=self.config.get('mixed_precision', True),
            memory_efficient_attention=self.config.get('memory_efficient_attention', True),
            
            # ACT settings optimized for halting (FIXED)
            halt_max_steps=self.config.get('halt_max_steps', 6),  # Reduced from 16
            halt_exploration_prob=self.config.get('halt_exploration_prob', 0.4),  # Increased from 0.1
            
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
        if WANDB_AVAILABLE and self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('project_name', 'hrm-code-generation'),
                name=self.config.get('run_name', f'optimized-run-{int(time.time())}'),
                config=self.config,
                mode=self.config.get('wandb_mode', 'online')
            )
            self.logger.info("✅ W&B logging initialized")
        else:
            self.logger.info("⚠️  W&B logging disabled")
    
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
            
            # Forward pass with mixed precision support
            self.optimizer.zero_grad()
            
            try:
                if self.use_mixed_precision:
                    # Mixed precision forward pass
                    with autocast():
                        new_carry, loss, metrics, outputs, all_halted = self.loss_fn(
                            return_keys=['logits'],
                            carry=carry,
                            batch=batch
                        )
                else:
                    # Standard precision forward pass
                    new_carry, loss, metrics, outputs, all_halted = self.loss_fn(
                        return_keys=['logits'],
                        carry=carry,
                        batch=batch
                    )
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at step {self.global_step}, skipping batch")
                    continue
                
                # Backward pass with mixed precision support
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                    
                    if WANDB_AVAILABLE and self.config.get('use_wandb', False):
                        wandb.log(log_dict)
                
                # Early stopping check for debugging
                max_batches = self.config.get('max_batches_per_epoch', None)
                if max_batches is not None and batch_idx >= max_batches:
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
        """Save model checkpoint with error handling and additional monitoring info"""
        checkpoint_metadata = {
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'system_stats': self.error_manager.system_stats,
            'dataset_metadata': self.dataset_metadata.__dict__,
            **(additional_info or {})
        }
        
        success = safe_save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=Path(path),
            metadata=checkpoint_metadata
        )
        
        if success:
            self.logger.info(f"✅ Checkpoint saved: {path}")
        else:
            self.logger.error(f"❌ Failed to save checkpoint: {path}")
            raise TrainingError(f"Checkpoint save failed: {path}")
    
    @robust_error_handler(max_retries=1)
    def train(self):
        """Main training loop with enhanced monitoring and error handling"""
        
        self.logger.info(f"\n🎯 Starting Optimized HRM Code Generation Training")
        
        try:
            return self._train_with_error_handling()
        except Exception as e:
            # Create error report
            error_report_path = Path("training_error_report.json")
            create_error_report(self.error_manager, error_report_path)
            self.logger.error(f"Training failed. Error report saved to: {error_report_path}")
            raise
    
    def _train_with_error_handling(self):
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
    
    # Configuration - Dataset selection logic
    real_lcb_path = 'data/livecodebench_real/livecodebench_real.json'
    swe_smith_path = 'data/swe-smith-1k'
    
    if args.data_path == 'mixed':
        # Mixed training mode - combine both datasets
        if os.path.exists(real_lcb_path) and os.path.exists(swe_smith_path):
            data_path = 'mixed'
            print(f"🔄 Using mixed training: LiveCodeBench + SWE-Smith-1k")
        else:
            # Fallback to available dataset
            data_path = real_lcb_path if os.path.exists(real_lcb_path) else swe_smith_path
            print(f"⚠️ Mixed mode requested but datasets missing, using: {data_path}")
    elif os.path.exists(real_lcb_path) and args.data_path == 'data/swe-smith-1k':
        # Auto-upgrade to real LiveCodeBench if available
        data_path = real_lcb_path
        print(f"🔥 Auto-upgraded to real LiveCodeBench dataset: {real_lcb_path}")
    else:
        data_path = args.data_path
        print(f"📝 Using dataset: {data_path}")
    
    config = {
        'data_path': data_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name or f'hrm-optimized-{int(time.time())}',
        'project_name': 'HRM-Code-Generation-Optimized',
        'checkpoint_interval': 10,
        'weight_decay': 0.2,  # Optimized weight decay
        'swe_search_weight': 0.3,  # Increased for better search behavior
        'reverse_learning_weight': 0.2,  # Enhanced reverse learning
        'H_cycles': 1,            # FIXED: Reduced for halting
        'L_cycles': 2,            # FIXED: Reduced for halting  
        'halt_max_steps': 6,      # FIXED: Reduced for halting
        'halt_exploration_prob': 0.5,  # Optimized exploration
        'swe_candidates': 15,     # SWE-Search candidate count
        'enable_swe_search': True,  # Enable SWE-Search features
        'enable_reverse_learning': True,  # Enable reverse learning
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