"""
Code Generation Loss Functions with GSPO Integration

This module implements sophisticated loss functions for code generation tasks,
including GSPO-enhanced sequence-level optimization, code-specific objectives,
and multi-task learning support.
"""

from typing import Dict, List, Optional, Tuple, Any, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from models.losses import ACTLossHead, IGNORE_LABEL_ID, softmax_cross_entropy, stablemax_cross_entropy
from models.code_generation.input_processor import CodeGenerationTask, ProgrammingLanguage
from models.code_generation.output_generator import CodeOutput, OutputFormat


@dataclass
class CodeGenerationMetrics:
    """Comprehensive metrics for code generation tasks"""
    # Core metrics
    count: torch.Tensor
    accuracy: torch.Tensor
    exact_accuracy: torch.Tensor
    
    # Code-specific metrics
    syntax_accuracy: torch.Tensor
    compilation_success: torch.Tensor
    test_pass_rate: torch.Tensor
    function_coverage: torch.Tensor
    
    # Language-specific metrics
    language_accuracy: Dict[str, torch.Tensor]
    
    # Task-specific metrics
    task_accuracy: Dict[str, torch.Tensor]
    
    # GSPO metrics
    sequence_importance: torch.Tensor
    policy_ratio: torch.Tensor
    
    # ACT metrics
    q_halt_accuracy: torch.Tensor
    avg_steps: torch.Tensor
    
    # Loss components
    lm_loss: torch.Tensor
    syntax_loss: torch.Tensor
    compilation_loss: torch.Tensor
    test_loss: torch.Tensor
    gspo_loss: torch.Tensor
    q_halt_loss: torch.Tensor
    q_continue_loss: torch.Tensor


class SyntaxAwareLoss(nn.Module):
    """
    Loss function that understands code syntax and structure
    Provides higher penalties for syntax-breaking errors
    """
    
    def __init__(self, 
                 base_loss_fn: str = "softmax_cross_entropy",
                 syntax_weight: float = 2.0,
                 structure_weight: float = 1.5):
        super().__init__()
        self.base_loss_fn = globals()[base_loss_fn]
        self.syntax_weight = syntax_weight
        self.structure_weight = structure_weight
        
        # Special token ranges for different code elements
        self.syntax_tokens = {
            'brackets': list(range(40, 46)),      # (){}[]
            'operators': list(range(46, 70)),     # +, -, ==, etc.
            'keywords': list(range(70, 150)),     # if, for, def, etc.
            'punctuation': list(range(150, 200))  # :, ;, ,, etc.
        }
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor,
                token_types: Optional[torch.Tensor] = None,
                ignore_index: int = IGNORE_LABEL_ID) -> torch.Tensor:
        """Compute syntax-aware loss with higher penalties for structural errors"""
        
        # Base cross-entropy loss
        base_loss = self.base_loss_fn(logits, labels, ignore_index=ignore_index)
        
        # Syntax weighting
        if token_types is not None:
            # Higher weights for syntax-critical tokens
            weights = torch.ones_like(labels, dtype=torch.float32)
            
            for token_type, token_range in self.syntax_tokens.items():
                mask = torch.isin(labels, torch.tensor(token_range, device=labels.device))
                if token_type in ['brackets', 'keywords']:
                    weights[mask] *= self.syntax_weight
                elif token_type in ['operators', 'punctuation']:
                    weights[mask] *= self.structure_weight
            
            # Apply weights
            weighted_loss = base_loss * weights
            return weighted_loss
        
        return base_loss


class CompilationLoss(nn.Module):
    """
    Loss function that encourages code compilation success
    Uses differentiable approximations of compilation checks
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # Learnable patterns for common compilation errors
        self.error_patterns = nn.ModuleDict({
            'bracket_mismatch': nn.Linear(512, 1),
            'missing_semicolon': nn.Linear(512, 1),
            'undefined_variable': nn.Linear(512, 1),
            'type_mismatch': nn.Linear(512, 1)
        })
    
    def forward(self, 
                hidden_states: torch.Tensor,
                generated_tokens: torch.Tensor,
                language_ids: torch.Tensor) -> torch.Tensor:
        """Estimate compilation success probability and penalize failures"""
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract features for compilation checking
        # Use mean pooling over sequence
        pooled_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Predict compilation error probabilities
        error_probs = {}
        for error_type, detector in self.error_patterns.items():
            error_probs[error_type] = torch.sigmoid(detector(pooled_features))
        
        # Combine error probabilities
        total_error_prob = sum(error_probs.values()) / len(error_probs)
        
        # Loss encourages low error probability
        compilation_loss = -torch.log(1 - total_error_prob.clamp(min=1e-8, max=1-1e-8))
        
        return self.weight * compilation_loss.mean()


class TestPassLoss(nn.Module):
    """
    Loss function that encourages generated code to pass test cases
    Uses execution simulation for differentiable test evaluation
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # Test execution simulator
        self.execution_simulator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                code_embeddings: torch.Tensor,
                test_embeddings: torch.Tensor) -> torch.Tensor:
        """Simulate test execution and compute pass rate loss"""
        
        # Combine code and test representations
        combined_repr = torch.cat([
            code_embeddings.mean(dim=1),  # Pool code sequence
            test_embeddings.mean(dim=1)   # Pool test sequence
        ], dim=-1)
        
        # Predict test pass probability
        pass_prob = self.execution_simulator(combined_repr)
        
        # Loss encourages high pass probability
        test_loss = -torch.log(pass_prob.clamp(min=1e-8))
        
        return self.weight * test_loss.mean()


class GSPOSequenceLoss(nn.Module):
    """
    Group Sequence Policy Optimization loss for code generation
    Implements sequence-level optimization with importance sampling
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 clip_range: float = 0.2,
                 entropy_weight: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.clip_range = clip_range
        self.entropy_weight = entropy_weight
    
    def compute_sequence_importance(self, 
                                   current_logprobs: torch.Tensor,
                                   old_logprobs: torch.Tensor,
                                   sequence_lengths: torch.Tensor) -> torch.Tensor:
        """Compute GSPO sequence-level importance ratios"""
        
        # Sequence-level probability ratios
        # Shape: [batch_size]
        current_seq_logprob = current_logprobs.sum(dim=-1)  # Sum over sequence
        old_seq_logprob = old_logprobs.sum(dim=-1)
        
        # Normalize by sequence length (GSPO's key innovation)
        normalized_current = current_seq_logprob / sequence_lengths.float()
        normalized_old = old_seq_logprob / sequence_lengths.float()
        
        # Importance ratio: (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
        importance_ratio = torch.exp(normalized_current - normalized_old)
        
        return importance_ratio
    
    def forward(self, 
                current_logits: torch.Tensor,
                old_logits: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                sequence_lengths: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute GSPO loss with sequence-level optimization"""
        
        # Convert logits to log probabilities
        current_logprobs = F.log_softmax(current_logits, dim=-1)
        old_logprobs = F.log_softmax(old_logits.detach(), dim=-1)
        
        # Gather action log probabilities
        current_action_logprobs = torch.gather(
            current_logprobs, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        old_action_logprobs = torch.gather(
            old_logprobs, dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply attention mask
        current_action_logprobs = current_action_logprobs * attention_mask
        old_action_logprobs = old_action_logprobs * attention_mask
        
        # Sequence-level importance ratios
        importance_ratios = self.compute_sequence_importance(
            current_action_logprobs, old_action_logprobs, sequence_lengths
        )
        
        # GSPO policy loss with sequence-level clipping
        advantages = rewards - rewards.mean()  # Center advantages
        
        # Clipped importance ratios
        clipped_ratios = torch.clamp(
            importance_ratios, 
            1 - self.clip_range, 
            1 + self.clip_range
        )
        
        # Policy loss terms
        policy_loss_1 = importance_ratios * advantages
        policy_loss_2 = clipped_ratios * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Entropy bonus for exploration
        probs = F.softmax(current_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        entropy_bonus = self.entropy_weight * (entropy * attention_mask).sum(dim=-1).mean()
        
        # Total GSPO loss
        total_loss = policy_loss - entropy_bonus
        
        # Metrics
        metrics = {
            'gspo_policy_loss': policy_loss.detach(),
            'gspo_entropy': entropy_bonus.detach(),
            'gspo_importance_ratio_mean': importance_ratios.mean().detach(),
            'gspo_importance_ratio_std': importance_ratios.std().detach(),
            'gspo_clipped_fraction': (importance_ratios != clipped_ratios).float().mean().detach()
        }
        
        return total_loss, metrics


class MultiTaskCodeLoss(nn.Module):
    """
    Multi-task loss function for different code generation scenarios
    Balances between generation, repair, diff editing, test prediction, and tool use
    """
    
    def __init__(self, 
                 task_weights: Optional[Dict[str, float]] = None,
                 adaptive_weighting: bool = True):
        super().__init__()
        
        # Default task weights
        self.task_weights = task_weights or {
            'generation': 1.0,
            'repair': 1.2,      # Higher weight for repair tasks
            'diff_edit': 1.1,   # Slightly higher for diff editing
            'test_prediction': 0.8,
            'tool_use': 0.9
        }
        
        self.adaptive_weighting = adaptive_weighting
        
        if adaptive_weighting:
            # Learnable task weights
            self.learned_weights = nn.Parameter(
                torch.ones(len(CodeGenerationTask))
            )
    
    def forward(self, 
                task_losses: Dict[str, torch.Tensor],
                task_ids: torch.Tensor,
                difficulty_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted multi-task loss"""
        
        total_loss = 0.0
        task_counts = {}
        
        for task_name, loss_value in task_losses.items():
            if task_name in self.task_weights:
                weight = self.task_weights[task_name]
                
                # Use learned weights if adaptive
                if self.adaptive_weighting:
                    task_enum = getattr(CodeGenerationTask, task_name.upper(), None)
                    if task_enum:
                        task_idx = list(CodeGenerationTask).index(task_enum)
                        learned_weight = torch.sigmoid(self.learned_weights[task_idx])
                        weight *= learned_weight
                
                # Apply difficulty weighting if provided
                if difficulty_scores is not None:
                    weight *= (1.0 + difficulty_scores.mean())
                
                total_loss += weight * loss_value
                task_counts[task_name] = 1
        
        # Normalize by number of active tasks
        if task_counts:
            total_loss /= len(task_counts)
        
        return total_loss


class CodeGenACTLossHead(ACTLossHead):
    """
    Enhanced ACT loss head for code generation with GSPO integration
    and code-specific metrics
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 loss_type: str = "softmax_cross_entropy",
                 enable_gspo: bool = True,
                 enable_code_losses: bool = True,
                 gspo_weight: float = 0.5,
                 syntax_weight: float = 0.3,
                 compilation_weight: float = 0.2):
        super().__init__(model, loss_type)
        
        self.enable_gspo = enable_gspo
        self.enable_code_losses = enable_code_losses
        self.gspo_weight = gspo_weight
        self.syntax_weight = syntax_weight
        self.compilation_weight = compilation_weight
        
        # Enhanced loss functions
        self.syntax_loss = SyntaxAwareLoss()
        self.compilation_loss = CompilationLoss()
        self.test_loss = TestPassLoss()
        self.multi_task_loss = MultiTaskCodeLoss()
        
        if enable_gspo:
            self.gspo_loss = GSPOSequenceLoss()
            # Store old policy parameters for GSPO
            self.register_buffer('old_logits', None)
    
    def compute_code_rewards(self, 
                           outputs: Dict[str, torch.Tensor],
                           labels: torch.Tensor,
                           language_ids: torch.Tensor,
                           task_ids: torch.Tensor) -> torch.Tensor:
        """Compute code-specific rewards for GSPO optimization"""
        
        batch_size = labels.size(0)
        rewards = torch.zeros(batch_size, device=labels.device)
        
        # Accuracy-based reward
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
        accuracy = is_correct.sum(-1).float() / mask.sum(-1).clamp(min=1)
        rewards += accuracy
        
        # Language-specific bonuses
        for i, lang_id in enumerate(language_ids):
            if lang_id == ProgrammingLanguage.RUST.value:  # Harder language
                rewards[i] *= 1.2
            elif lang_id == ProgrammingLanguage.PYTHON.value:  # Easier language
                rewards[i] *= 0.9
        
        # Task-specific bonuses
        for i, task_id in enumerate(task_ids):
            if task_id == CodeGenerationTask.REPAIR.value:  # Harder task
                rewards[i] *= 1.3
            elif task_id == CodeGenerationTask.TOOL_USE.value:  # Complex task
                rewards[i] *= 1.1
        
        return rewards
    
    def forward(self,
                return_keys: Sequence[str],
                # Enhanced model args
                **model_kwargs) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # Get model outputs
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        
        # Extract additional information
        language_ids = new_carry.current_data.get("language_ids", torch.zeros(labels.size(0)))
        task_ids = new_carry.current_data.get("task_ids", torch.zeros(labels.size(0)))
        complexity_scores = new_carry.current_data.get("complexity_scores", torch.zeros(labels.size(0)))
        
        # Base metrics computation
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Enhanced code-specific metrics
            valid_metrics = new_carry.halted & (loss_counts > 0)
            
            # Language-specific accuracy
            language_accuracy = {}
            for lang in ProgrammingLanguage:
                lang_mask = valid_metrics & (language_ids == lang.value)
                if lang_mask.any():
                    language_accuracy[lang.name] = (lang_mask & seq_is_correct).sum()
            
            # Task-specific accuracy
            task_accuracy = {}
            for task in CodeGenerationTask:
                task_mask = valid_metrics & (task_ids == task.value)
                if task_mask.any():
                    task_accuracy[task.name] = (task_mask & seq_is_correct).sum()
            
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "avg_steps": torch.where(valid_metrics, new_carry.steps, 0).float().mean(),
                "avg_complexity": complexity_scores.mean(),
                "language_accuracy": language_accuracy,
                "task_accuracy": task_accuracy
            }
        
        # Base language modeling loss
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        
        # Enhanced code-specific losses
        total_loss = lm_loss
        
        if self.enable_code_losses:
            # Syntax-aware loss
            syntax_loss = self.syntax_loss(outputs["logits"], labels).mean()
            total_loss += self.syntax_weight * syntax_loss
            metrics["syntax_loss"] = syntax_loss.detach()
            
            # Compilation loss (if hidden states available)
            if "hidden_states" in outputs:
                compilation_loss = self.compilation_loss(
                    outputs["hidden_states"], 
                    torch.argmax(outputs["logits"], dim=-1),
                    language_ids
                )
                total_loss += self.compilation_weight * compilation_loss
                metrics["compilation_loss"] = compilation_loss.detach()
        
        # GSPO sequence-level optimization
        if self.enable_gspo and self.old_logits is not None:
            # Compute sequence-level rewards
            rewards = self.compute_code_rewards(outputs, labels, language_ids, task_ids)
            
            # Sequence lengths (excluding padding)
            sequence_lengths = mask.sum(dim=-1)
            attention_mask = mask.float()
            
            # GSPO loss
            gspo_loss, gspo_metrics = self.gspo_loss(
                outputs["logits"],
                self.old_logits,
                labels,
                rewards,
                sequence_lengths,
                attention_mask
            )
            
            total_loss += self.gspo_weight * gspo_loss
            metrics.update(gspo_metrics)
            metrics["gspo_total_loss"] = gspo_loss.detach()
        
        # Update old logits for next GSPO iteration
        if self.enable_gspo:
            self.old_logits = outputs["logits"].detach().clone()
        
        # Q-learning losses
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], 
            seq_is_correct.to(outputs["q_halt_logits"].dtype), 
            reduction="sum"
        )
        
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], 
                outputs["target_q_continue"], 
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # Add Q-learning losses
        total_loss += 0.5 * (q_halt_loss + q_continue_loss)
        
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "total_loss": total_loss.detach()
        })
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


def create_code_generation_loss(model: nn.Module, 
                               config: Dict[str, Any]) -> CodeGenACTLossHead:
    """Factory function to create appropriate loss for code generation"""
    
    return CodeGenACTLossHead(
        model=model,
        loss_type=config.get("loss_type", "softmax_cross_entropy"),
        enable_gspo=config.get("enable_gspo", True),
        enable_code_losses=config.get("enable_code_losses", True),
        gspo_weight=config.get("gspo_weight", 0.5),
        syntax_weight=config.get("syntax_weight", 0.3),
        compilation_weight=config.get("compilation_weight", 0.2)
    )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Code Generation Loss Functions...")
    
    # Mock model for testing
    class MockModel(nn.Module):
        def initial_carry(self, batch):
            return None
        
        def forward(self, **kwargs):
            batch_size = 4
            seq_len = 128
            vocab_size = 40000
            
            outputs = {
                "logits": torch.randn(batch_size, seq_len, vocab_size),
                "q_halt_logits": torch.randn(batch_size),
                "q_continue_logits": torch.randn(batch_size),
                "hidden_states": torch.randn(batch_size, seq_len, 512)
            }
            
            carry = type('MockCarry', (), {
                'halted': torch.ones(batch_size, dtype=torch.bool),
                'steps': torch.randint(1, 10, (batch_size,)),
                'current_data': {
                    'labels': torch.randint(0, vocab_size, (batch_size, seq_len)),
                    'language_ids': torch.randint(0, 6, (batch_size,)),
                    'task_ids': torch.randint(0, 5, (batch_size,)),
                    'complexity_scores': torch.rand(batch_size)
                }
            })()
            
            return carry, outputs
    
    # Test enhanced loss head
    model = MockModel()
    loss_config = {
        "enable_gspo": True,
        "enable_code_losses": True,
        "gspo_weight": 0.5
    }
    
    loss_head = create_code_generation_loss(model, loss_config)
    
    # Test forward pass
    carry, total_loss, metrics, outputs, halted = loss_head(
        return_keys=["logits"],
        batch={}
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Accuracy: {metrics['accuracy'].item():.4f}")
    print(f"Average steps: {metrics['avg_steps'].item():.2f}")
    print(f"Language accuracy keys: {list(metrics['language_accuracy'].keys())}")
    print(f"Task accuracy keys: {list(metrics['task_accuracy'].keys())}")
    
    # Test individual loss components
    print("\nTesting individual loss components...")
    
    # Test syntax loss
    syntax_loss = SyntaxAwareLoss()
    test_logits = torch.randn(2, 10, 1000)
    test_labels = torch.randint(0, 1000, (2, 10))
    syntax_result = syntax_loss(test_logits, test_labels)
    print(f"Syntax loss shape: {syntax_result.shape}")
    
    # Test GSPO loss
    gspo_loss = GSPOSequenceLoss()
    current_logits = torch.randn(2, 10, 1000)
    old_logits = torch.randn(2, 10, 1000)
    actions = torch.randint(0, 1000, (2, 10))
    rewards = torch.rand(2)
    seq_lengths = torch.tensor([8, 9])
    attention_mask = torch.ones(2, 10)
    
    gspo_result, gspo_metrics = gspo_loss(
        current_logits, old_logits, actions, rewards, seq_lengths, attention_mask
    )
    print(f"GSPO loss: {gspo_result.item():.4f}")
    print(f"GSPO metrics: {list(gspo_metrics.keys())}")
    
    print("\nCode generation loss functions test completed successfully!")