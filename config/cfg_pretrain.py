"""
Minimal HRM Configuration for Code Generation

This provides a basic configuration structure compatible with the HRM model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """HRM model configuration"""
    
    # Model architecture
    vocab_size: int = 40000
    n_embd: int = 512
    n_layer: int = 12
    n_head: int = 8
    max_seq_len: int = 2048
    
    # HRM-specific
    high_level_layers: int = 4
    low_level_layers: int = 8
    act_threshold: float = 0.9
    max_act_steps: int = 16
    
    # Model behavior
    causal: bool = False  # Non-causal for code generation
    dropout: float = 0.1
    bias: bool = True
    
    # Position encoding
    pos_enc_type: str = "learned"  # learned, rotary, none
    
    # Activation functions
    activation: str = "swiglu"  # gelu, swish, swiglu
    
    # Normalization
    norm_type: str = "rmsnorm"  # layernorm, rmsnorm
    
    # Training specific
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.high_level_layers + self.low_level_layers <= self.n_layer, "Layer counts exceed total layers"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"