from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Use float32 for MPS compatibility (MPS doesn't support float64)
    dtype = torch.float32 if logits.device.type == 'mps' else torch.float64
    logprobs = log_stablemax(logits.to(dtype), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class ACTSWESearchLossHead(nn.Module):
    """Enhanced ACT Loss Head with SWE-Search and Reverse Learning integration"""
    
    def __init__(self, model: nn.Module, loss_type: str, swe_search_weight: float = 0.2, reverse_learning_weight: float = 0.1):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.swe_search_weight = swe_search_weight
        self.reverse_learning_weight = reverse_learning_weight
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model forward pass (includes SWE-Search if enabled)
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Standard loss computation (same as ACTLossHead)
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Standard losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # SWE-Search enhancement loss
        swe_search_loss = 0
        if hasattr(self.model, 'get_latest_search_metrics'):
            search_metrics = self.model.get_latest_search_metrics()
            if search_metrics:
                swe_search_loss = self._compute_swe_search_loss(search_metrics, seq_is_correct)
                metrics["swe_search_loss"] = swe_search_loss.detach()
                
                # Additional search metrics
                metrics.update(self._compute_search_metrics(search_metrics))

        # Reverse Learning enhancement loss
        reverse_learning_loss = 0
        if hasattr(self.model, 'get_latest_reverse_metrics'):
            reverse_metrics = self.model.get_latest_reverse_metrics()
            if reverse_metrics:
                reverse_learning_loss = self._compute_reverse_learning_loss(reverse_metrics, seq_is_correct)
                metrics["reverse_learning_loss"] = reverse_learning_loss.detach()
                
                # Additional reverse learning metrics
                metrics.update(self._compute_reverse_learning_metrics(reverse_metrics))

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Combined loss with SWE-Search and Reverse Learning components
        total_loss = (lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + 
                     self.swe_search_weight * swe_search_loss + 
                     self.reverse_learning_weight * reverse_learning_loss)

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
    
    def _compute_swe_search_loss(self, search_metrics, seq_is_correct):
        """Compute loss that encourages effective search behavior"""
        if not search_metrics:
            return torch.tensor(0.0, device=seq_is_correct.device)
        
        device = seq_is_correct.device
        batch_size = seq_is_correct.size(0)
        
        # Extract search performance metrics
        convergence_scores = []
        efficiency_scores = []
        
        for i, metrics in enumerate(search_metrics):
            if i < batch_size:  # Ensure we don't exceed batch size
                convergence_scores.append(metrics.final_score)
                efficiency_scores.append(metrics.search_efficiency)
        
        if not convergence_scores:
            return torch.tensor(0.0, device=device)
        
        # Pad to batch size if needed
        while len(convergence_scores) < batch_size:
            convergence_scores.append(0.0)
            efficiency_scores.append(0.0)
        
        convergence_tensor = torch.tensor(convergence_scores[:batch_size], device=device)
        efficiency_tensor = torch.tensor(efficiency_scores[:batch_size], device=device)
        
        # Reward successful search convergence
        convergence_reward = convergence_tensor
        
        # Efficiency reward (balance search quality with computational cost)
        efficiency_reward = efficiency_tensor
        
        # Bonus for correct predictions with high search scores
        correctness_bonus = seq_is_correct.float() * convergence_tensor * 0.5
        
        # Combined search loss (negative because we want to maximize these metrics)
        search_loss = -(convergence_reward + 0.3 * efficiency_reward + correctness_bonus).mean()
        
        return search_loss
    
    def _compute_search_metrics(self, search_metrics):
        """Compute additional metrics for monitoring search performance"""
        if not search_metrics:
            return {}
        
        avg_final_score = sum(m.final_score for m in search_metrics) / len(search_metrics)
        avg_iterations = sum(m.iterations_used for m in search_metrics) / len(search_metrics)
        avg_candidates = sum(m.total_candidates for m in search_metrics) / len(search_metrics)
        convergence_rate = sum(1 for m in search_metrics if m.convergence_achieved) / len(search_metrics)
        avg_efficiency = sum(m.search_efficiency for m in search_metrics) / len(search_metrics)
        
        return {
            "swe_search_score": torch.tensor(avg_final_score),
            "swe_search_iterations": torch.tensor(avg_iterations),
            "swe_search_candidates": torch.tensor(avg_candidates),
            "swe_search_convergence_rate": torch.tensor(convergence_rate),
            "swe_search_efficiency": torch.tensor(avg_efficiency),
        }
    
    def _compute_reverse_learning_loss(self, reverse_metrics, seq_is_correct):
        """Compute loss that encourages effective reverse learning behavior"""
        if not reverse_metrics:
            return torch.tensor(0.0, device=seq_is_correct.device)
        
        device = seq_is_correct.device
        batch_size = seq_is_correct.size(0)
        
        # Extract reverse learning performance metrics
        insight_strengths = []
        integration_gates = []
        planning_refinements = []
        
        for i, metrics in enumerate(reverse_metrics):
            if i < batch_size:  # Ensure we don't exceed batch size
                insight_strengths.append(metrics.insight_strength)
                integration_gates.append(metrics.integration_gate_value)
                planning_refinements.append(metrics.planning_refinement_score)
        
        if not insight_strengths:
            return torch.tensor(0.0, device=device)
        
        # Pad to batch size if needed
        while len(insight_strengths) < batch_size:
            insight_strengths.append(0.0)
            integration_gates.append(0.0)
            planning_refinements.append(0.0)
        
        insight_tensor = torch.tensor(insight_strengths[:batch_size], device=device)
        gate_tensor = torch.tensor(integration_gates[:batch_size], device=device)
        refinement_tensor = torch.tensor(planning_refinements[:batch_size], device=device)
        
        # Reward meaningful insight extraction
        insight_reward = insight_tensor.clamp(0, 10)  # Cap at reasonable value
        
        # Reward appropriate gating (not too conservative, not too aggressive)
        # Optimal gate values are around 0.3-0.7
        optimal_gate = 0.5
        gate_quality = 1.0 - torch.abs(gate_tensor - optimal_gate) * 2  # Convert to quality score
        gate_reward = gate_quality.clamp(0, 1)
        
        # Reward effective planning refinement
        refinement_reward = refinement_tensor.clamp(0, 1)  # Keep within reasonable bounds
        
        # Bonus for correct predictions with good reverse learning
        correctness_bonus = seq_is_correct.float() * refinement_tensor * 0.3
        
        # Combined reverse learning loss (negative because we want to maximize these metrics)
        consistency_penalty = torch.abs(refinement_tensor - 0.2)  # Encourage moderate refinement
        reverse_loss = -(
            0.4 * insight_reward + 
            0.3 * gate_reward + 
            0.2 * refinement_reward + 
            correctness_bonus - 
            0.1 * consistency_penalty
        ).mean()
        
        return reverse_loss
    
    def _compute_reverse_learning_metrics(self, reverse_metrics):
        """Compute additional metrics for monitoring reverse learning performance"""
        if not reverse_metrics:
            return {}
        
        avg_insight_strength = sum(m.insight_strength for m in reverse_metrics) / len(reverse_metrics)
        avg_integration_gate = sum(m.integration_gate_value for m in reverse_metrics) / len(reverse_metrics)
        avg_feedback_magnitude = sum(m.feedback_magnitude for m in reverse_metrics) / len(reverse_metrics)
        avg_planning_refinement = sum(m.planning_refinement_score for m in reverse_metrics) / len(reverse_metrics)
        
        return {
            "reverse_insight_strength": torch.tensor(avg_insight_strength),
            "reverse_integration_gate": torch.tensor(avg_integration_gate),
            "reverse_feedback_magnitude": torch.tensor(avg_feedback_magnitude),
            "reverse_planning_refinement": torch.tensor(avg_planning_refinement),
        }
