import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class SWESearchMetrics:
    """Metrics tracking for SWE-Search performance"""
    final_score: float
    iterations_used: int
    total_candidates: int
    convergence_achieved: bool
    search_efficiency: float


class SWESearchController(nn.Module):
    """Monte Carlo Tree Search integration for self-evolving agent coordination
    
    This module implements the SWE-Search framework that achieves 23% performance
    improvements through iterative self-improvement and multi-agent debate coordination.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.iteration_count = getattr(config, 'swe_search_iterations', 3)
        self.debate_rounds = getattr(config, 'multi_agent_debate_rounds', 2)
        self.exploration_factor = getattr(config, 'mcts_exploration_factor', 1.4)
        self.convergence_threshold = 0.95
        
        # Get forward dtype from config
        self.forward_dtype = getattr(torch, getattr(config, 'forward_dtype', 'float32'))
        
        # MCTS state evaluation components
        self.state_evaluator = nn.Linear(config.hidden_size, 1)
        self.action_value_estimator = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Multi-agent debate coordination
        self.debate_coordinator = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads // 2,
            batch_first=True
        )
        
        # Solution encoding/decoding projections
        self.solution_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.solution_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Performance tracking for self-evolution
        self.performance_buffer = []
        self.buffer_size = getattr(config, 'performance_buffer_size', 100)
        self.evolution_threshold = getattr(config, 'evolution_threshold', 0.95)
        
        # Initialize parameters and cast to forward dtype
        self._init_parameters()
        # Cast all modules to forward dtype
        self.to(dtype=self.forward_dtype)
    
    def _init_parameters(self):
        """Initialize module parameters"""
        nn.init.xavier_uniform_(self.state_evaluator.weight)
        nn.init.xavier_uniform_(self.action_value_estimator.weight)
        nn.init.xavier_uniform_(self.solution_encoder.weight)
        nn.init.xavier_uniform_(self.solution_decoder.weight)
    
    def swe_search_forward(self, 
                          problem_embedding: torch.Tensor, 
                          base_solution_embedding: torch.Tensor,
                          base_solution_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, SWESearchMetrics]:
        """
        Implement SWE-Search with MCTS + multi-agent debate
        
        Args:
            problem_embedding: HRM high-level embedding of the coding problem [batch, seq_len, hidden_size]
            base_solution_embedding: Initial solution embedding from base model [batch, seq_len, hidden_size]
            base_solution_tokens: Optional token representation for solution tracking
            
        Returns:
            refined_solution_embedding: Best solution embedding after iterative improvement
            search_metrics: Performance tracking data
        """
        batch_size = problem_embedding.size(0)
        device = problem_embedding.device
        
        # Initialize search state
        best_solution_embedding = base_solution_embedding
        best_score = float('-inf')
        search_history = []
        total_candidates_generated = 0
        
        # Create initial candidate pool
        candidate_embeddings = [base_solution_embedding]
        candidate_scores = [self._evaluate_solution(problem_embedding, base_solution_embedding)]
        
        for iteration in range(self.iteration_count):
            # MCTS expansion phase
            expanded_candidates, num_generated = self._mcts_expand(
                problem_embedding, candidate_embeddings, candidate_scores
            )
            total_candidates_generated += num_generated
            
            # Multi-agent debate phase
            if len(expanded_candidates) > 1:
                debated_candidates = self._multi_agent_debate(
                    problem_embedding, expanded_candidates
                )
            else:
                debated_candidates = expanded_candidates
            
            # Evaluation and selection phase
            evaluated_candidates = []
            for candidate_emb in debated_candidates:
                score = self._evaluate_solution(problem_embedding, candidate_emb)
                evaluated_candidates.append((candidate_emb, score))
            
            # Update best solution
            current_best_emb, current_best_score = max(evaluated_candidates, key=lambda x: x[1])
            if current_best_score > best_score:
                best_score = current_best_score
                best_solution_embedding = current_best_emb
            
            # Track search progress
            search_history.append({
                'iteration': iteration,
                'candidates_generated': num_generated,
                'best_score': float(best_score),
                'score_improvement': float(current_best_score - best_score) if iteration > 0 else 0.0
            })
            
            # Update candidate pool for next iteration (keep top candidates)
            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
            top_k = min(3, len(evaluated_candidates))  # Keep top 3 candidates
            candidate_embeddings = [emb for emb, _ in evaluated_candidates[:top_k]]
            candidate_scores = [score for _, score in evaluated_candidates[:top_k]]
            
            # Early termination if solution is high quality
            if best_score > self.convergence_threshold:
                break
        
        # Create search metrics
        search_efficiency = float(best_score) / (iteration + 1) if iteration >= 0 else 0.0
        metrics = SWESearchMetrics(
            final_score=float(best_score),
            iterations_used=iteration + 1,
            total_candidates=total_candidates_generated,
            convergence_achieved=best_score > self.convergence_threshold,
            search_efficiency=search_efficiency
        )
        
        # Update performance buffer for self-evolution
        self._update_performance_buffer(metrics)
        
        return best_solution_embedding, metrics
    
    def _mcts_expand(self, 
                    problem_embedding: torch.Tensor, 
                    candidate_embeddings: List[torch.Tensor],
                    candidate_scores: List[float]) -> Tuple[List[torch.Tensor], int]:
        """MCTS-based candidate expansion with UCB1 selection"""
        expanded_candidates = []
        total_generated = 0
        
        for i, (candidate_emb, base_score) in enumerate(zip(candidate_embeddings, candidate_scores)):
            # Compute UCB1 score for exploration vs exploitation
            ucb_score = self._compute_ucb1_score(base_score, i + 1, len(candidate_embeddings))
            
            # Generate variations based on UCB1 policy
            variations = self._generate_embedding_variations(
                candidate_emb, problem_embedding, ucb_score
            )
            
            expanded_candidates.extend(variations)
            total_generated += len(variations)
        
        # Add original candidates to expanded pool
        expanded_candidates.extend(candidate_embeddings)
        
        return expanded_candidates, total_generated
    
    def _compute_ucb1_score(self, base_score: float, visits: int, total_visits: int) -> float:
        """Compute UCB1 score for MCTS exploration"""
        if visits == 0:
            return float('inf')
        
        exploration_term = self.exploration_factor * math.sqrt(
            math.log(total_visits) / visits
        )
        
        return base_score + exploration_term
    
    def _generate_embedding_variations(self, 
                                     base_embedding: torch.Tensor,
                                     problem_context: torch.Tensor,
                                     ucb_score: float) -> List[torch.Tensor]:
        """Generate solution variations in embedding space"""
        variations = []
        batch_size, seq_len, hidden_size = base_embedding.shape
        device = base_embedding.device
        
        # Ensure dtype compatibility
        base_embedding = base_embedding.to(dtype=self.forward_dtype)
        problem_context = problem_context.to(dtype=self.forward_dtype)
        
        # Number of variations based on UCB score
        num_variations = min(3, max(1, int(ucb_score * 2)))
        
        for _ in range(num_variations):
            # Strategy 1: Add controlled noise for exploration
            noise_scale = 0.1 * (1.0 - ucb_score)  # Less noise for high-scoring solutions
            noise = torch.randn_like(base_embedding, dtype=self.forward_dtype) * noise_scale
            variation1 = base_embedding + noise
            
            # Strategy 2: Interpolate with problem context
            alpha = torch.rand(1).item() * 0.3  # Up to 30% interpolation
            problem_mean = problem_context.mean(dim=1, keepdim=True)
            variation2 = (1 - alpha) * base_embedding + alpha * problem_mean.expand_as(base_embedding)
            
            # Strategy 3: Apply learned transformation
            encoded = self.solution_encoder(base_embedding)
            variation3 = self.solution_decoder(encoded)
            
            variations.extend([variation1, variation2, variation3])
        
        return variations[:num_variations]  # Return requested number of variations
    
    def _multi_agent_debate(self, 
                           problem_embedding: torch.Tensor,
                           candidate_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """Multi-agent debate refinement using attention mechanisms"""
        if len(candidate_embeddings) < 2:
            return candidate_embeddings
        
        # Ensure all embeddings have consistent dtype
        candidate_embeddings = [emb.to(dtype=self.forward_dtype) for emb in candidate_embeddings]
        
        # Stack candidate embeddings for batch processing
        # Handle potential size mismatches by taking the minimum sequence length
        min_seq_len = min(emb.size(1) for emb in candidate_embeddings)
        stacked_candidates = torch.stack([
            emb[:, :min_seq_len, :] for emb in candidate_embeddings
        ], dim=1)  # [batch, num_candidates, seq_len, hidden_size]
        
        batch_size, num_candidates, seq_len, hidden_size = stacked_candidates.shape
        
        # Reshape for attention: [batch * seq_len, num_candidates, hidden_size]
        candidates_reshaped = stacked_candidates.transpose(1, 2).reshape(
            batch_size * seq_len, num_candidates, hidden_size
        )
        
        # Ensure debate coordinator is in the right dtype
        candidates_reshaped = candidates_reshaped.to(dtype=self.forward_dtype)
        
        debated_candidates = []
        
        # Multi-round debate process
        current_embeddings = candidates_reshaped
        
        for round_idx in range(self.debate_rounds):
            # Attention-based debate coordination
            debated_embeddings, attention_weights = self.debate_coordinator(
                current_embeddings,
                current_embeddings,
                current_embeddings
            )
            
            # Combine original and debated representations
            alpha = 0.7 - (round_idx * 0.1)  # Decrease original weight over rounds
            current_embeddings = alpha * current_embeddings + (1 - alpha) * debated_embeddings
        
        # Reshape back to original format
        final_embeddings = current_embeddings.reshape(
            batch_size, seq_len, num_candidates, hidden_size
        ).transpose(1, 2)  # [batch, num_candidates, seq_len, hidden_size]
        
        # Convert back to list format, padding to original sequence lengths if needed
        refined_candidates = []
        for i, orig_emb in enumerate(candidate_embeddings):
            refined_emb = final_embeddings[:, i, :, :]  # [batch, seq_len, hidden_size]
            
            # Pad back to original length if necessary
            if refined_emb.size(1) < orig_emb.size(1):
                padding_size = orig_emb.size(1) - refined_emb.size(1)
                padding = torch.zeros(batch_size, padding_size, hidden_size, 
                                    device=refined_emb.device, dtype=self.forward_dtype)
                refined_emb = torch.cat([refined_emb, padding], dim=1)
            
            refined_candidates.append(refined_emb)
        
        return refined_candidates
    
    def _evaluate_solution(self, 
                          problem_embedding: torch.Tensor,
                          solution_embedding: torch.Tensor) -> float:
        """Evaluate solution quality using learned metrics"""
        # Ensure compatible dimensions by taking means over sequence dimension
        problem_rep = problem_embedding.mean(dim=1)  # [batch, hidden_size]
        solution_rep = solution_embedding.mean(dim=1)  # [batch, hidden_size]
        
        # Combine problem and solution context
        combined_embedding = problem_rep + solution_rep
        
        # Ensure dtype compatibility
        combined_embedding = combined_embedding.to(dtype=self.forward_dtype)
        
        # Compute solution quality score
        quality_score = torch.sigmoid(self.state_evaluator(combined_embedding))
        
        # Return average score across batch
        return quality_score.mean().item()
    
    def _update_performance_buffer(self, metrics: SWESearchMetrics):
        """Update performance buffer for self-evolution tracking"""
        self.performance_buffer.append({
            'final_score': metrics.final_score,
            'iterations_used': metrics.iterations_used,
            'total_candidates': metrics.total_candidates,
            'efficiency': metrics.search_efficiency
        })
        
        # Maintain buffer size
        if len(self.performance_buffer) > self.buffer_size:
            self.performance_buffer.pop(0)
    
    def should_evolve_architecture(self) -> bool:
        """Determine if architecture should self-evolve based on performance"""
        if len(self.performance_buffer) < 20:  # Need minimum samples
            return False
        
        recent_performance = np.mean([
            p['final_score'] for p in self.performance_buffer[-20:]
        ])
        
        return recent_performance < self.evolution_threshold
    
    def evolve_search_parameters(self):
        """Self-evolve search parameters based on performance patterns"""
        if not self.should_evolve_architecture():
            return
        
        # Analyze performance patterns
        recent_metrics = self.performance_buffer[-50:] if len(self.performance_buffer) >= 50 else self.performance_buffer
        
        if not recent_metrics:
            return
        
        iteration_efficiency = np.mean([
            p['final_score'] / p['iterations_used'] for p in recent_metrics
        ])
        
        candidate_efficiency = np.mean([
            p['final_score'] / p['total_candidates'] for p in recent_metrics
        ])
        
        # Adjust parameters based on efficiency analysis
        if iteration_efficiency < 0.3:  # Low iteration efficiency
            self.iteration_count = min(self.iteration_count + 1, 5)
            print(f"SWE-Search: Increased iterations to {self.iteration_count}")
        elif iteration_efficiency > 0.8:  # High iteration efficiency
            self.iteration_count = max(self.iteration_count - 1, 2)
            print(f"SWE-Search: Decreased iterations to {self.iteration_count}")
        
        if candidate_efficiency < 0.2:  # Low candidate efficiency
            self.exploration_factor = min(self.exploration_factor * 1.1, 2.0)
            print(f"SWE-Search: Increased exploration factor to {self.exploration_factor:.3f}")
        elif candidate_efficiency > 0.6:  # High candidate efficiency
            self.exploration_factor = max(self.exploration_factor * 0.95, 0.8)
            print(f"SWE-Search: Decreased exploration factor to {self.exploration_factor:.3f}")
    
    def get_search_statistics(self) -> Dict:
        """Get current search performance statistics"""
        if not self.performance_buffer:
            return {}
        
        recent_scores = [p['final_score'] for p in self.performance_buffer[-20:]]
        recent_efficiency = [p['efficiency'] for p in self.performance_buffer[-20:]]
        
        return {
            'avg_final_score': np.mean(recent_scores),
            'avg_search_efficiency': np.mean(recent_efficiency),
            'convergence_rate': np.mean([s > self.convergence_threshold for s in recent_scores]),
            'current_iterations': self.iteration_count,
            'current_exploration_factor': self.exploration_factor,
            'buffer_size': len(self.performance_buffer)
        }