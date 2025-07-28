"""
Reverse-Order Learning Module for HRM

This module implements reverse-order learning that extracts insights from 
low-level implementation details and feeds them back to high-level strategic 
planning, enabling the model to refine its architectural decisions based on 
implementation experience.

Key components:
- Insight extraction from low-level states
- Reverse projection to high-level space
- Gated integration for refined planning
- Bidirectional feedback loops
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ReverseLearningMetrics:
    """Metrics for tracking reverse learning performance"""
    insight_strength: float
    integration_gate_value: float
    feedback_magnitude: float
    planning_refinement_score: float


class ReverseLearningModule(nn.Module):
    """
    Implements reverse-order learning for hierarchical feedback from 
    low-level implementation details back to high-level strategic planning.
    
    This enables the model to:
    1. Learn from implementation details to improve architectural decisions
    2. Refine high-level planning based on low-level execution patterns
    3. Develop better code structure and maintainability awareness
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Insight extraction dimensions
        self.insight_dim = config.hidden_size // 4
        self.enable_reverse_learning = getattr(config, 'enable_reverse_learning', False)
        self.feedback_weight = getattr(config, 'reverse_feedback_weight', 0.1)
        
        # Get forward dtype from config
        self.forward_dtype = getattr(torch, getattr(config, 'forward_dtype', 'float32'))
        
        # Insight extraction from low-level implementation patterns
        self.insight_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.insight_dim),
            nn.LayerNorm(self.insight_dim)
        )
        
        # Reverse projection back to high-level planning space
        self.reverse_projector = nn.Sequential(
            nn.Linear(self.insight_dim, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Gating mechanism for controlled insight integration
        self.insight_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Quality assessment for insights
        self.insight_quality_assessor = nn.Linear(self.insight_dim, 1)
        
        # Attention mechanism for selective insight focus
        self.insight_attention = nn.MultiheadAttention(
            self.hidden_size, 
            num_heads=config.num_heads // 4,
            batch_first=True
        )
        
        # Pattern memory for learning from implementation experiences
        self.pattern_memory_size = 512
        self.register_buffer(
            'pattern_memory', 
            torch.zeros(self.pattern_memory_size, self.insight_dim, dtype=self.forward_dtype)
        )
        self.register_buffer('memory_pointer', torch.tensor(0, dtype=torch.long))
        
        # Initialize parameters
        self._init_parameters()
        # Cast all modules to forward dtype
        self.to(dtype=self.forward_dtype)
    
    def _init_parameters(self):
        """Initialize module parameters"""
        for module in [self.insight_extractor, self.reverse_projector, self.insight_gate]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize quality assessor
        nn.init.xavier_uniform_(self.insight_quality_assessor.weight)
        nn.init.zeros_(self.insight_quality_assessor.bias)
    
    def extract_implementation_insights(self, z_L: torch.Tensor) -> Tuple[torch.Tensor, ReverseLearningMetrics]:
        """
        Extract insights from low-level implementation details
        
        Args:
            z_L: Low-level states from HRM [batch, seq_len, hidden_size]
            
        Returns:
            insights: Extracted insights [batch, seq_len, insight_dim]
            metrics: Performance tracking metrics
        """
        batch_size, seq_len, _ = z_L.shape
        
        # Ensure dtype compatibility
        z_L = z_L.to(dtype=self.forward_dtype)
        
        # Extract patterns from low-level implementation
        # Use multiple pooling strategies to capture different aspects
        
        # 1. Temporal attention pooling - focus on important implementation moments
        temporal_weights = F.softmax(
            self.insight_quality_assessor(
                self.insight_extractor(z_L)
            ).squeeze(-1), 
            dim=-1
        )
        weighted_L = (z_L * temporal_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        
        # 2. Mean pooling for overall patterns
        mean_pooled_L = z_L.mean(dim=1, keepdim=True)
        
        # 3. Max pooling for peak implementation details
        max_pooled_L, _ = z_L.max(dim=1, keepdim=True)
        
        # Combine different pooling strategies
        combined_L = torch.cat([weighted_L, mean_pooled_L, max_pooled_L], dim=1)
        combined_L = combined_L.mean(dim=1, keepdim=True)  # Average the strategies
        
        # Extract core insights
        insights = self.insight_extractor(combined_L)
        
        # Assess insight quality
        insight_quality = torch.sigmoid(self.insight_quality_assessor(insights))
        
        # Update pattern memory with high-quality insights
        if self.training and insight_quality.mean() > 0.7:
            self._update_pattern_memory(insights.detach())
        
        # Expand insights back to sequence length for detailed feedback
        insights_expanded = insights.expand(-1, seq_len, -1)
        
        # Create metrics
        metrics = ReverseLearningMetrics(
            insight_strength=float(insights.norm(dim=-1).mean()),
            integration_gate_value=0.0,  # Will be set in integration step
            feedback_magnitude=float(insight_quality.mean()),
            planning_refinement_score=0.0  # Will be computed in integration
        )
        
        return insights_expanded, metrics
    
    def integrate_reverse_feedback(self, 
                                 z_H: torch.Tensor, 
                                 insights: torch.Tensor,
                                 previous_metrics: ReverseLearningMetrics) -> Tuple[torch.Tensor, ReverseLearningMetrics]:
        """
        Integrate reverse feedback into high-level planning
        
        Args:
            z_H: High-level states [batch, seq_len, hidden_size]
            insights: Extracted insights [batch, seq_len, insight_dim]
            previous_metrics: Metrics from insight extraction
            
        Returns:
            z_H_refined: Refined high-level states
            updated_metrics: Updated performance metrics
        """
        batch_size, seq_len, _ = z_H.shape
        
        # Ensure dtype compatibility
        z_H = z_H.to(dtype=self.forward_dtype)
        insights = insights.to(dtype=self.forward_dtype)
        
        # Project insights back to high-level space
        reverse_feedback = self.reverse_projector(insights)
        
        # Apply attention mechanism to focus on relevant insights
        refined_feedback, attention_weights = self.insight_attention(
            z_H, reverse_feedback, reverse_feedback
        )
        
        # Combine original high-level state with reverse feedback
        combined_state = torch.cat([z_H, refined_feedback], dim=-1)
        
        # Compute gating weights for controlled integration
        gate_weights = self.insight_gate(combined_state)
        
        # Apply gated integration with learnable weight
        feedback_contribution = gate_weights * refined_feedback * self.feedback_weight
        z_H_refined = z_H + feedback_contribution
        
        # Compute planning refinement score
        refinement_magnitude = feedback_contribution.norm(dim=-1).mean()
        original_magnitude = z_H.norm(dim=-1).mean()
        planning_refinement_score = float(refinement_magnitude / (original_magnitude + 1e-8))
        
        # Update metrics
        updated_metrics = ReverseLearningMetrics(
            insight_strength=previous_metrics.insight_strength,
            integration_gate_value=float(gate_weights.mean()),
            feedback_magnitude=previous_metrics.feedback_magnitude,
            planning_refinement_score=planning_refinement_score
        )
        
        return z_H_refined, updated_metrics
    
    def _update_pattern_memory(self, insights: torch.Tensor):
        """Update pattern memory with new insights"""
        batch_size = insights.size(0)
        
        for i in range(batch_size):
            # Store insight in circular buffer
            self.pattern_memory[self.memory_pointer] = insights[i, 0]  # Use first token's insight
            self.memory_pointer = (self.memory_pointer + 1) % self.pattern_memory_size
    
    def get_pattern_similarity(self, current_insights: torch.Tensor) -> torch.Tensor:
        """Compute similarity with stored patterns"""
        if not self.training:
            return torch.zeros(current_insights.size(0), device=current_insights.device)
        
        # Compute similarity with pattern memory
        similarities = F.cosine_similarity(
            current_insights.view(-1, self.insight_dim).unsqueeze(1),
            self.pattern_memory.unsqueeze(0),
            dim=-1
        )
        
        # Return max similarity for each sample
        max_similarities, _ = similarities.max(dim=-1)
        return max_similarities.view(current_insights.size(0), -1).mean(dim=-1)
    
    def compute_reverse_learning_loss(self, 
                                    z_H_original: torch.Tensor,
                                    z_H_refined: torch.Tensor,
                                    z_L: torch.Tensor,
                                    targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that encourages effective reverse learning
        
        Args:
            z_H_original: Original high-level states
            z_H_refined: Refined high-level states after reverse feedback
            z_L: Low-level states
            targets: Target outputs
            
        Returns:
            reverse_learning_loss: Loss encouraging effective feedback
        """
        # Consistency loss - refined states should remain similar to originals
        consistency_loss = F.mse_loss(z_H_refined, z_H_original.detach())
        
        # Improvement loss - refined states should lead to better predictions
        # (This is a placeholder - in practice would use actual task performance)
        improvement_incentive = -torch.mean(torch.norm(z_H_refined - z_H_original, dim=-1))
        
        # Pattern diversity loss - encourage diverse insight extraction
        insights, _ = self.extract_implementation_insights(z_L)
        diversity_loss = -torch.mean(torch.std(insights, dim=1))
        
        # Combine losses
        total_loss = (
            0.5 * consistency_loss + 
            0.3 * improvement_incentive + 
            0.2 * diversity_loss
        )
        
        return total_loss
    
    def get_reverse_learning_statistics(self) -> Dict[str, float]:
        """Get current reverse learning performance statistics"""
        if not hasattr(self, '_recent_metrics') or not self._recent_metrics:
            return {}
        
        # Compute averages from recent metrics
        recent_metrics = self._recent_metrics[-20:]  # Last 20 batches
        
        return {
            'avg_insight_strength': sum(m.insight_strength for m in recent_metrics) / len(recent_metrics),
            'avg_integration_gate': sum(m.integration_gate_value for m in recent_metrics) / len(recent_metrics),
            'avg_feedback_magnitude': sum(m.feedback_magnitude for m in recent_metrics) / len(recent_metrics),
            'avg_planning_refinement': sum(m.planning_refinement_score for m in recent_metrics) / len(recent_metrics),
            'pattern_memory_utilization': float(self.memory_pointer) / self.pattern_memory_size
        }
    
    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> Tuple[torch.Tensor, ReverseLearningMetrics]:
        """
        Complete reverse learning forward pass
        
        Args:
            z_H: High-level states [batch, seq_len, hidden_size]
            z_L: Low-level states [batch, seq_len, hidden_size]
            
        Returns:
            z_H_refined: Refined high-level states
            metrics: Performance tracking metrics
        """
        if not self.enable_reverse_learning:
            # Return original states with dummy metrics
            dummy_metrics = ReverseLearningMetrics(0.0, 0.0, 0.0, 0.0)
            return z_H, dummy_metrics
        
        # Extract insights from implementation details
        insights, extract_metrics = self.extract_implementation_insights(z_L)
        
        # Integrate feedback into high-level planning
        z_H_refined, final_metrics = self.integrate_reverse_feedback(
            z_H, insights, extract_metrics
        )
        
        # Store metrics for statistics
        if not hasattr(self, '_recent_metrics'):
            self._recent_metrics = []
        self._recent_metrics.append(final_metrics)
        
        # Keep only recent metrics to prevent memory growth
        if len(self._recent_metrics) > 100:
            self._recent_metrics = self._recent_metrics[-50:]
        
        return z_H_refined, final_metrics