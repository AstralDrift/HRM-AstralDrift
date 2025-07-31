"""
Code-Specific Embeddings System for HRM

This module implements the multi-language, code-aware embedding system designed
in Phase 1a. It supports hierarchical tokenization, language-specific embeddings,
and syntax-aware representations.
"""

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.sparse_embedding import CastedSparseEmbedding
from models.layers import CastedEmbedding, CastedLinear


class LanguageType(Enum):
    """Programming language types for embedding specialization"""
    PYTHON = 0
    JAVASCRIPT = 1
    JAVA = 2
    CPP = 3
    GO = 4
    RUST = 5


class CodeTokenType(Enum):
    """Types of code tokens for enhanced representation"""
    SHARED = 0           # Common programming concepts
    LANGUAGE_SPECIFIC = 1 # Language-specific syntax
    IDENTIFIER = 2       # Variable/function names
    OPERATOR = 3         # Mathematical/logical operators
    KEYWORD = 4          # Language keywords
    LITERAL = 5          # String/number literals
    COMMENT = 6          # Comments and documentation
    SYNTAX = 7           # Brackets, punctuation


@dataclass
class CodeEmbeddingConfig:
    """Configuration for code-specific embeddings"""
    vocab_size: int = 40000
    shared_vocab_size: int = 25000
    lang_vocab_size: int = 2500
    hidden_size: int = 512
    num_languages: int = 6
    max_seq_len: int = 2048
    embedding_init_std: float = 0.02
    use_syntax_positions: bool = True
    use_learned_positions: bool = True
    dropout_prob: float = 0.1


class SyntaxAwarePositionalEmbedding(nn.Module):
    """
    Syntax-aware positional embeddings that understand code structure
    Different from standard positional embeddings by considering:
    - Code block nesting levels
    - Function/class boundaries  
    - Logical code segments
    """
    
    def __init__(self, config: CodeEmbeddingConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Standard positional embeddings
        self.position_embeddings = CastedEmbedding(
            config.max_seq_len, 
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Syntax-specific positional embeddings
        self.nesting_embeddings = CastedEmbedding(
            16,  # Max nesting depth of 16
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Segment embeddings (problem, code, tests, etc.)
        self.segment_embeddings = CastedEmbedding(
            8,   # Different code segments
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Projection to combine different positional info
        self.position_projection = CastedLinear(
            config.hidden_size * 3,  # position + nesting + segment
            config.hidden_size,
            bias=False
        )
    
    def _extract_syntax_info(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract nesting levels and segments from token sequence"""
        batch_size, seq_len = token_ids.shape
        
        # Simple heuristic for nesting level (count brackets)
        nesting_levels = torch.zeros_like(token_ids)
        segments = torch.zeros_like(token_ids)
        
        # For now, use simple rules - in production, this would use
        # proper syntax parsing per language
        bracket_tokens = [40, 41, 123, 125, 91, 93]  # (){}[]
        
        for b in range(batch_size):
            level = 0
            segment = 0
            for i in range(seq_len):
                token_id = token_ids[b, i].item()
                
                # Update nesting level based on brackets
                if token_id in [40, 123, 91]:  # Opening brackets
                    level += 1
                elif token_id in [41, 125, 93]:  # Closing brackets
                    level = max(0, level - 1)
                
                # Simple segment detection (special tokens would define segments)
                # This is a placeholder - real implementation would use structured parsing
                if token_id > 39990:  # Special tokens
                    segment = (segment + 1) % 8
                
                nesting_levels[b, i] = min(level, 15)
                segments[b, i] = segment
        
        return nesting_levels, segments
    
    def forward(self, token_ids: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate syntax-aware positional embeddings"""
        batch_size, seq_len = token_ids.shape
        
        # Standard position embeddings
        if positions is None:
            positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(positions)
        
        # Extract syntax information
        nesting_levels, segments = self._extract_syntax_info(token_ids)
        
        # Syntax-specific embeddings
        nesting_emb = self.nesting_embeddings(nesting_levels)
        segment_emb = self.segment_embeddings(segments)
        
        # Combine all positional information
        combined_pos = torch.cat([pos_emb, nesting_emb, segment_emb], dim=-1)
        final_pos = self.position_projection(combined_pos)
        
        return final_pos


class LanguageAdapterEmbedding(nn.Module):
    """
    Language-specific adapter for embeddings using LoRA-style adaptation
    Each language gets a low-rank adaptation to the shared embedding space
    """
    
    def __init__(self, config: CodeEmbeddingConfig, language: LanguageType, adapter_rank: int = 64):
        super().__init__()
        self.config = config
        self.language = language
        self.adapter_rank = adapter_rank
        
        # LoRA-style low-rank adaptation
        self.adapter_down = CastedLinear(config.hidden_size, adapter_rank, bias=False)
        self.adapter_up = CastedLinear(adapter_rank, config.hidden_size, bias=False)
        self.scaling = 1.0 / adapter_rank
        
        # Language-specific vocabulary embeddings
        self.lang_vocab_embeddings = CastedEmbedding(
            config.lang_vocab_size,
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Initialize adapter to near-zero
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up.weight, std=config.embedding_init_std)
    
    def forward(self, shared_embeddings: torch.Tensor, lang_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply language-specific adaptation to shared embeddings"""
        # LoRA adaptation
        adapted = shared_embeddings + self.scaling * self.adapter_up(self.adapter_down(shared_embeddings))
        
        # Add language-specific token embeddings if provided
        if lang_token_ids is not None:
            lang_emb = self.lang_vocab_embeddings(lang_token_ids)
            adapted = adapted + lang_emb
        
        return adapted


class MultiLanguageCodeEmbedding(nn.Module):
    """
    Main multi-language code embedding system
    Implements the hierarchical embedding architecture from Phase 1a
    """
    
    def __init__(self, config: CodeEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Shared vocabulary embeddings (common programming concepts)
        self.shared_embeddings = CastedEmbedding(
            config.shared_vocab_size,
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Language-specific adapters
        self.language_adapters = nn.ModuleDict({
            lang.name.lower(): LanguageAdapterEmbedding(config, lang)
            for lang in LanguageType
        })
        
        # Language type embeddings (global language context)
        self.language_type_embeddings = CastedEmbedding(
            config.num_languages,
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Code token type embeddings
        self.token_type_embeddings = CastedEmbedding(
            len(CodeTokenType),
            config.hidden_size,
            init_std=config.embedding_init_std
        )
        
        # Syntax-aware positional embeddings
        if config.use_syntax_positions:
            self.syntax_positions = SyntaxAwarePositionalEmbedding(config)
        elif config.use_learned_positions:
            self.position_embeddings = CastedEmbedding(
                config.max_seq_len,
                config.hidden_size,
                init_std=config.embedding_init_std
            )
        
        # Embedding scaling and projection
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout_prob)
        
        # Cross-language alignment projection (for contrastive learning)
        self.alignment_projection = CastedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )
    
    def _split_token_ids(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split token IDs into shared and language-specific parts"""
        # Tokens < shared_vocab_size are shared, rest are language-specific
        shared_mask = token_ids < self.config.shared_vocab_size
        
        shared_ids = torch.where(shared_mask, token_ids, torch.zeros_like(token_ids))
        lang_ids = torch.where(~shared_mask, token_ids - self.config.shared_vocab_size, torch.zeros_like(token_ids))
        
        return shared_ids, lang_ids
    
    def forward(self, 
                token_ids: torch.Tensor,
                language_ids: torch.Tensor,
                token_types: Optional[torch.Tensor] = None,
                return_alignment_features: bool = False) -> torch.Tensor:
        """
        Forward pass of multi-language code embeddings
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            language_ids: Language type for each sample [batch_size]
            token_types: Token type IDs [batch_size, seq_len] (optional)
            return_alignment_features: Whether to return features for alignment loss
            
        Returns:
            Embedded representations [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = token_ids.shape
        
        # Split tokens into shared and language-specific
        shared_ids, lang_ids = self._split_token_ids(token_ids)
        
        # Get shared embeddings
        shared_emb = self.shared_embeddings(shared_ids)
        
        # Apply language-specific adaptations
        language_embeddings = []
        for i, lang_id in enumerate(language_ids):
            lang_name = LanguageType(lang_id.item()).name.lower()
            adapter = self.language_adapters[lang_name]
            
            # Get language-specific token IDs for this sample
            sample_lang_ids = lang_ids[i:i+1]  # Keep batch dimension
            
            # Apply adapter
            adapted_emb = adapter(shared_emb[i:i+1], sample_lang_ids)
            language_embeddings.append(adapted_emb)
        
        # Stack language-adapted embeddings
        embeddings = torch.cat(language_embeddings, dim=0)
        
        # Add language type embeddings
        lang_type_emb = self.language_type_embeddings(language_ids)  # [batch_size, hidden_size]
        embeddings = embeddings + lang_type_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add token type embeddings if provided
        if token_types is not None:
            token_type_emb = self.token_type_embeddings(token_types)
            embeddings = embeddings + token_type_emb
        
        # Add positional embeddings
        if hasattr(self, 'syntax_positions'):
            pos_emb = self.syntax_positions(token_ids)
            embeddings = embeddings + pos_emb
        elif hasattr(self, 'position_embeddings'):
            positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embeddings(positions)
            embeddings = embeddings + pos_emb
        
        # Scale embeddings
        embeddings = self.embed_scale * embeddings
        
        # Apply dropout
        embeddings = self.embedding_dropout(embeddings)
        
        if return_alignment_features:
            # Return both embeddings and alignment features for contrastive learning
            alignment_features = self.alignment_projection(embeddings.mean(dim=1))  # Pool to sentence level
            return embeddings, alignment_features
        
        return embeddings


class CodeSpecificSparseEmbedding(CastedSparseEmbedding):
    """
    Extended sparse embedding for code generation puzzles
    Supports task-specific and language-specific puzzle embeddings
    """
    
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 batch_size: int, 
                 init_std: float, 
                 cast_to: torch.dtype,
                 num_languages: int = 6,
                 num_tasks: int = 5):
        super().__init__(num_embeddings, embedding_dim, batch_size, init_std, cast_to)
        
        self.num_languages = num_languages
        self.num_tasks = num_tasks
        
        # Language-specific puzzle embeddings
        self.register_buffer("language_puzzle_weights", 
            trunc_normal_init_(torch.empty((num_languages, embedding_dim)), std=init_std), 
            persistent=True
        )
        
        # Task-specific puzzle embeddings
        self.register_buffer("task_puzzle_weights", 
            trunc_normal_init_(torch.empty((num_tasks, embedding_dim)), std=init_std), 
            persistent=True
        )
    
    def forward(self, 
                puzzle_ids: torch.Tensor, 
                language_ids: Optional[torch.Tensor] = None,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with language and task conditioning
        
        Args:
            puzzle_ids: Base puzzle identifiers
            language_ids: Language type for each sample
            task_ids: Task type for each sample
        """
        # Get base puzzle embeddings
        base_embeddings = super().forward(puzzle_ids)
        
        # Add language-specific components
        if language_ids is not None:
            lang_embeddings = self.language_puzzle_weights[language_ids]
            base_embeddings = base_embeddings + lang_embeddings
        
        # Add task-specific components
        if task_ids is not None:
            task_embeddings = self.task_puzzle_weights[task_ids]
            base_embeddings = base_embeddings + task_embeddings
        
        return base_embeddings


def create_cross_language_alignment_loss(embeddings_dict: Dict[str, torch.Tensor], 
                                        similar_pairs: List[Tuple[str, str]],
                                        temperature: float = 0.07) -> torch.Tensor:
    """
    Create contrastive loss for cross-language alignment
    
    Args:
        embeddings_dict: Dictionary mapping language names to embeddings
        similar_pairs: List of (lang1, lang2) pairs that should be similar
        temperature: Temperature for contrastive loss
    """
    total_loss = 0.0
    num_pairs = 0
    
    for lang1, lang2 in similar_pairs:
        if lang1 in embeddings_dict and lang2 in embeddings_dict:
            emb1 = F.normalize(embeddings_dict[lang1], dim=-1)
            emb2 = F.normalize(embeddings_dict[lang2], dim=-1)
            
            # Cosine similarity
            similarity = torch.mm(emb1, emb2.t()) / temperature
            
            # Contrastive loss (pull similar pairs together)
            labels = torch.arange(emb1.size(0), device=emb1.device)
            loss = F.cross_entropy(similarity, labels)
            
            total_loss += loss
            num_pairs += 1
    
    return total_loss / max(num_pairs, 1)


if __name__ == "__main__":
    # Test the embedding system
    config = CodeEmbeddingConfig(
        vocab_size=40000,
        hidden_size=512,
        max_seq_len=256
    )
    
    embedding_system = MultiLanguageCodeEmbedding(config)
    
    # Create test data
    batch_size = 4
    seq_len = 128
    
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    language_ids = torch.randint(0, config.num_languages, (batch_size,))
    
    print("Testing Multi-Language Code Embedding System...")
    
    # Test forward pass
    embeddings = embedding_system(token_ids, language_ids)
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Test with alignment features
    embeddings, alignment_features = embedding_system(
        token_ids, language_ids, return_alignment_features=True
    )
    print(f"Alignment features shape: {alignment_features.shape}")
    
    # Test sparse embeddings
    sparse_embedding = CodeSpecificSparseEmbedding(
        num_embeddings=100,
        embedding_dim=512,
        batch_size=batch_size,
        init_std=0.02,
        cast_to=torch.float32
    )
    
    puzzle_ids = torch.randint(0, 100, (batch_size,))
    sparse_emb = sparse_embedding(puzzle_ids, language_ids, torch.randint(0, 5, (batch_size,)))
    print(f"Sparse embeddings shape: {sparse_emb.shape}")
    
    print("Code-specific embedding system test completed successfully!")