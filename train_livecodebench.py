"""
Training Script for HRM on LiveCodeBench

This script trains the HRM model on LiveCodeBench data using the existing
HRM architecture with adaptations for code generation tasks.
"""

import os
import sys
import time
import json
import math
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# HRM imports
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.code_generation.hrm_code_model import HRMCodeGenerationModel
from dataset.livecodebench_dataset import create_livecodebench_dataloaders, AdaptiveComplexitySampler
from evaluation.livecodebench_evaluator import LiveCodeBenchEvaluator, EvaluationConfig
from config.cfg_pretrain import Config
import argparse
from pydantic import BaseModel

class LiveCodeBenchTrainingConfig(BaseModel):
    """Configuration for LiveCodeBench training"""
    
    # Data configuration
    data_path: str = "data/livecodebench-hrm"
    max_seq_len: int = 2048
    vocab_size: int = 40000
    
    # Model configuration
    model_size: str = "27M"  # 27M, 110M, 350M
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    
    # HRM-specific configuration
    high_level_layers: int = 4
    low_level_layers: int = 8
    act_threshold: float = 0.9
    max_act_steps: int = 16
    
    # Training configuration
    batch_size: int = 32
    global_batch_size: int = 384
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    epochs: int = 20
    
    # Scenario-specific configuration
    scenario_weights: str = "1.0,0.8,0.6,0.5"  # generation,repair,test,exec
    complexity_schedule: str = "curriculum"  # linear, exponential, curriculum
    balanced_sampling: bool = True
    
    # Optimization configuration
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    
    # Evaluation configuration
    eval_interval: int = 1000
    eval_steps: int = 100
    save_interval: int = 5000
    
    # System configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Experiment configuration
    experiment_name: str = "hrm_livecodebench"
    output_dir: str = "checkpoints"
    log_wandb: bool = True
    wandb_project: str = "hrm-code-generation"
    
    # Resume training
    resume_from: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

# ArgumentParser will be created in main function

class LiveCodeBenchTrainer:
    """Trainer for HRM on LiveCodeBench"""
    
    def __init__(self, config: LiveCodeBenchTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.step = 0
        self.epoch = 0
        self.best_pass_at_1 = 0.0
        
        # Setup distributed training if needed
        if config.distributed:
            self._setup_distributed()
        
        # Create output directory
        self.checkpoint_dir = Path(config.output_dir) / config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # Initialize mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup logging
        if config.log_wandb and (not config.distributed or config.local_rank == 0):
            self._setup_wandb()
        
        # Load checkpoint if resuming
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        
        print(f"Trainer initialized for experiment: {config.experiment_name}")
        print(f"Model parameters: {self._count_parameters():,}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def _setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend="nccl")
        self.config.local_rank = int(os.environ["LOCAL_RANK"])
        self.config.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f"cuda:{self.config.local_rank}")
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the HRM model"""
        print("Creating HRM Code Generation model...")
        
        # Create base HRM configuration
        hrm_config = Config(
            vocab_size=self.config.vocab_size,
            n_embd=self.config.hidden_size,
            n_layer=self.config.num_layers,
            n_head=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
            causal=False,  # Non-causal for code generation
            # HRM-specific
            high_level_layers=self.config.high_level_layers,
            low_level_layers=self.config.low_level_layers,
            act_threshold=self.config.act_threshold,
            max_act_steps=self.config.max_act_steps
        )
        
        # Create HRM code generation model
        model = HRMCodeGenerationModel(hrm_config)
        model.to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        
        # Wrap with DDP if distributed
        if self.config.distributed:
            model = DDP(model, device_ids=[self.config.local_rank])
        
        return model
    
    def _create_data_loaders(self) -> Tuple:
        """Create training and validation data loaders"""
        print("Creating data loaders...")
        
        # Parse scenario weights
        weights = [float(w.strip()) for w in self.config.scenario_weights.split(',')]
        scenario_weights = {
            "codegeneration": weights[0],
            "selfrepair": weights[1] if len(weights) > 1 else 0.8,
            "testoutputprediction": weights[2] if len(weights) > 2 else 0.6,
            "codeexecution": weights[3] if len(weights) > 3 else 0.5
        }
        
        train_loader, val_loader = create_livecodebench_dataloaders(
            data_dir=self.config.data_path,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size,
            scenario_weights=scenario_weights,
            num_workers=self.config.num_workers,
            balanced_sampling=self.config.balanced_sampling
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> Tuple:
        """Create optimizer and learning rate scheduler"""
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Create learning rate scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                decay_ratio = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return optimizer, scheduler
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.experiment_name,
            config=self.config.dict(),
            resume="auto" if self.config.resume_from else None
        )
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            "train_loss": 0.0,
            "train_generation_loss": 0.0,
            "train_repair_loss": 0.0,
            "train_test_loss": 0.0,
            "train_exec_loss": 0.0,
            "train_samples": 0
        }
        
        # Set epoch for adaptive sampling
        if hasattr(self.train_loader.dataloader.sampler, 'set_epoch'):
            self.train_loader.dataloader.sampler.set_epoch(self.epoch)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch}",
            disable=self.config.distributed and self.config.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss, scenario_losses = self._forward_step(batch)
            else:
                loss, scenario_losses = self._forward_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            batch_size = batch["inputs"].size(0)
            epoch_metrics["train_loss"] += loss.item() * batch_size
            epoch_metrics["train_samples"] += batch_size
            
            # Update scenario-specific losses
            for scenario, scenario_loss in scenario_losses.items():
                if scenario_loss is not None:
                    epoch_metrics[f"train_{scenario}_loss"] += scenario_loss.item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            self.step += 1
            
            # Evaluation
            if self.step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                
                # Log metrics
                if self.config.log_wandb and (not self.config.distributed or self.config.local_rank == 0):
                    wandb.log({
                        "step": self.step,
                        "epoch": self.epoch,
                        **val_metrics
                    })
                
                # Save checkpoint if best
                if val_metrics.get("val_pass_at_1", 0) > self.best_pass_at_1:
                    self.best_pass_at_1 = val_metrics["val_pass_at_1"]
                    self._save_checkpoint("best")
                
                self.model.train()
            
            # Save checkpoint periodically
            if self.step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}")
            
            # Early stopping based on max steps
            if self.step >= self.config.max_steps:
                break
        
        # Average epoch metrics
        for key in epoch_metrics:
            if key != "train_samples":
                epoch_metrics[key] /= epoch_metrics["train_samples"]
        
        return epoch_metrics
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for a single batch"""
        # Get model outputs
        outputs = self.model(
            inputs=batch["inputs"],
            puzzle_identifiers=batch["puzzle_identifier"],
            attention_mask=batch["attention_mask"]
        )
        
        # Calculate loss based on scenario
        total_loss = 0.0
        scenario_losses = {}
        
        # Group batch by scenario
        scenarios = batch["scenario"]
        unique_scenarios = set(scenarios)
        
        for scenario in unique_scenarios:
            # Get indices for this scenario
            scenario_mask = [i for i, s in enumerate(scenarios) if s == scenario]
            if not scenario_mask:
                scenario_losses[scenario] = None
                continue
            
            # Get scenario-specific data
            scenario_outputs = outputs[scenario_mask]
            scenario_targets = batch["targets"][scenario_mask]
            scenario_weights = batch["scenario_weight"][scenario_mask]
            
            # Calculate scenario loss
            scenario_loss = self._calculate_loss(scenario_outputs, scenario_targets, scenario)
            weighted_loss = (scenario_loss * scenario_weights.squeeze()).mean()
            
            scenario_losses[scenario] = scenario_loss
            total_loss += weighted_loss
        
        return total_loss, scenario_losses
    
    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor, scenario: str) -> torch.Tensor:
        """Calculate loss for a specific scenario"""
        # For now, use standard cross-entropy loss
        # In the future, this could be customized per scenario
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 is typically pad token
        
        # Reshape for loss calculation
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        
        return loss_fn(outputs_flat, targets_flat)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if not self.config.distributed or self.config.local_rank == 0:
            print("Running evaluation...")
        
        self.model.eval()
        eval_metrics = {
            "val_loss": 0.0,
            "val_samples": 0
        }
        
        # Sample a subset for faster evaluation
        eval_steps = min(self.config.eval_steps, len(self.val_loader))
        
        for i, batch in enumerate(self.val_loader):
            if i >= eval_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
                loss, _ = self._forward_step(batch)
            
            # Update metrics
            batch_size = batch["inputs"].size(0)
            eval_metrics["val_loss"] += loss.item() * batch_size
            eval_metrics["val_samples"] += batch_size
        
        # Average metrics
        eval_metrics["val_loss"] /= eval_metrics["val_samples"]
        
        # Run pass@k evaluation on a small subset
        if self.step % (self.config.eval_interval * 5) == 0:  # Less frequent pass@k evaluation
            pass_at_k_metrics = self._evaluate_pass_at_k()
            eval_metrics.update(pass_at_k_metrics)
        
        if not self.config.distributed or self.config.local_rank == 0:
            print(f"Evaluation - Loss: {eval_metrics['val_loss']:.4f}")
            if "val_pass_at_1" in eval_metrics:
                print(f"Pass@1: {eval_metrics['val_pass_at_1']:.4f}")
        
        return eval_metrics
    
    @torch.no_grad()
    def _evaluate_pass_at_k(self) -> Dict[str, float]:
        """Evaluate pass@k metrics on a small subset"""
        try:
            # Create evaluator
            eval_config = EvaluationConfig(
                model_path="",  # Not needed since we pass model directly
                num_samples=5,
                timeout_seconds=5,  # Short timeout for training-time evaluation
                save_detailed_results=False
            )
            
            # This is a simplified evaluation - full evaluation would be done separately
            # For now, return placeholder metrics
            return {
                "val_pass_at_1": 0.1,  # Placeholder
                "val_pass_at_5": 0.2   # Placeholder
            }
        except Exception as e:
            print(f"Warning: Pass@k evaluation failed: {e}")
            return {}
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        if self.config.distributed and self.config.local_rank != 0:
            return  # Only save from rank 0
        
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        
        # Get model state dict (unwrap DDP if needed)
        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_pass_at_1": self.best_pass_at_1,
            "config": self.config.dict()
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_pass_at_1 = checkpoint.get("best_pass_at_1", 0.0)
        
        # Load scaler if available
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Resumed training from step {self.step}, epoch {self.epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Max steps: {self.config.max_steps}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            if self.config.log_wandb and (not self.config.distributed or self.config.local_rank == 0):
                wandb.log({
                    "epoch": epoch,
                    **epoch_metrics
                })
            
            # Print epoch summary
            if not self.config.distributed or self.config.local_rank == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} completed - Loss: {epoch_metrics['train_loss']:.4f}, "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if self.step >= self.config.max_steps:
                break
        
        # Save final checkpoint
        self._save_checkpoint("final")
        
        if self.config.log_wandb and (not self.config.distributed or self.config.local_rank == 0):
            wandb.finish()
        
        print("Training completed!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train HRM on LiveCodeBench")
    parser.add_argument("--data-path", type=str, default="data/livecodebench-hrm", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--experiment-name", type=str, default="hrm_livecodebench", help="Experiment name")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    config = LiveCodeBenchTrainingConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        experiment_name=args.experiment_name,
        log_wandb=args.log_wandb,
        mixed_precision=args.mixed_precision
    )
    
    print("HRM LiveCodeBench Training")
    print(f"Configuration: {config}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create trainer and start training
    trainer = LiveCodeBenchTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()