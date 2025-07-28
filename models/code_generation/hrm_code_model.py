"""
Enhanced HRM Model for Code Generation

This module implements the modified HRM architecture specifically designed for
code generation tasks, integrating multi-language support, code-specific embeddings,
and enhanced ACT mechanisms.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1ReasoningModule,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1
)
from models.layers import CastedLinear, CastedEmbedding, rms_norm, SwiGLU, Attention, CosSin
from models.code_generation.embeddings import (
    MultiLanguageCodeEmbedding, 
    CodeEmbeddingConfig,
    CodeSpecificSparseEmbedding,
    LanguageType
)
from models.code_generation.input_processor import (
    CodeGenerationInput,
    CodeGenerationTask,
    ProgrammingLanguage
)


class CodeGenHRMConfig(HierarchicalReasoningModel_ACTV1Config):
    """Extended HRM configuration for code generation"""
    
    # Code generation specific settings
    code_vocab_size: int = 40000
    shared_vocab_size: int = 25000
    lang_vocab_size: int = 2500
    num_languages: int = 6
    num_code_tasks: int = 5
    
    # Multi-language support
    language_adapter_rank: int = 64
    use_syntax_positions: bool = True
    embedding_dropout: float = 0.1
    
    # Enhanced ACT for code complexity
    complexity_aware_act: bool = True
    min_cycles_simple: int = 1
    max_cycles_complex: int = 8
    
    # Output generation settings
    use_multi_head_output: bool = True
    diff_generation_heads: bool = True
    tool_command_head: bool = True
    
    # GSPO integration settings
    use_sequence_optimization: bool = True
    gspo_temperature: float = 0.1


class CodeGenHierarchicalBlock(HierarchicalReasoningModel_ACTV1Block):
    """Enhanced HRM block with code-specific reasoning capabilities"""
    
    def __init__(self, config: CodeGenHRMConfig, level: str):
        super().__init__(config)
        self.level = level
        self.config = config
        
        # Level-specific enhancements
        if level == "high":
            # Strategic planning components
            self.algorithm_planner = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            self.complexity_analyzer = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            self.pattern_recognizer = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            
        elif level == "low":
            # Implementation components
            self.syntax_generator = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            self.api_optimizer = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            self.style_enforcer = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        
        # Inter-level communication
        self.level_communication = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads // 2,
            batch_first=True
        )
        
        # Language-aware adaptation
        self.language_adaptation = CastedLinear(config.hidden_size * 2, config.hidden_size, bias=False)
    
    def forward(self, 
                cos_sin: CosSin, 
                hidden_states: torch.Tensor,
                language_context: Optional[torch.Tensor] = None,
                level_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Standard HRM processing
        residual = hidden_states
        
        # Self attention with RMSNorm
        hidden_states = rms_norm(
            residual + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), 
            variance_epsilon=self.norm_eps
        )
        
        residual = hidden_states
        
        # Level-specific processing
        if self.level == "high" and level_context is not None:
            # Strategic planning enhancements
            strategic_features = self.algorithm_planner(hidden_states)
            complexity_features = self.complexity_analyzer(hidden_states) 
            pattern_features = self.pattern_recognizer(hidden_states)
            
            # Combine strategic information
            enhanced_features = strategic_features + complexity_features + pattern_features
            hidden_states = hidden_states + enhanced_features * 0.1  # Small residual connection
            
        elif self.level == "low" and level_context is not None:
            # Implementation enhancements
            syntax_features = self.syntax_generator(hidden_states)
            api_features = self.api_optimizer(hidden_states)
            style_features = self.style_enforcer(hidden_states)
            
            # Combine implementation information
            enhanced_features = syntax_features + api_features + style_features
            hidden_states = hidden_states + enhanced_features * 0.1
        
        # Inter-level communication if context provided
        if level_context is not None:
            attended_context, _ = self.level_communication(
                hidden_states, level_context, level_context
            )
            hidden_states = hidden_states + attended_context * 0.1
        
        # Language-aware adaptation
        if language_context is not None:
            # Expand language context to match sequence length
            seq_len = hidden_states.size(1)
            expanded_lang_context = language_context.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine with hidden states
            combined = torch.cat([hidden_states, expanded_lang_context], dim=-1)
            adapted_states = self.language_adaptation(combined)
            hidden_states = hidden_states + adapted_states * 0.1
        
        # MLP with RMSNorm
        hidden_states = rms_norm(
            residual + self.mlp(hidden_states), 
            variance_epsilon=self.norm_eps
        )
        
        return hidden_states


class CodeGenReasoningModule(HierarchicalReasoningModel_ACTV1ReasoningModule):
    """Enhanced reasoning module with code generation capabilities"""
    
    def __init__(self, layers: List[CodeGenHierarchicalBlock], level: str, config: CodeGenHRMConfig):
        super().__init__(layers)
        self.level = level
        self.config = config
        
        # Language-specific processing
        self.language_processor = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        
        # Complexity-aware processing
        self.complexity_processor = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                input_injection: torch.Tensor,
                language_context: Optional[torch.Tensor] = None,
                complexity_info: Optional[torch.Tensor] = None,
                other_level_states: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        
        # Input injection
        hidden_states = hidden_states + input_injection
        
        # Process language context
        if language_context is not None:
            lang_processed = self.language_processor(language_context)
        else:
            lang_processed = None
        
        # Enhanced layer processing with code-specific context
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                language_context=lang_processed,
                level_context=other_level_states,
                **kwargs
            )
        
        return hidden_states


class ComplexityAwareACTController:
    """
    Enhanced ACT controller that adapts computation based on code complexity
    Integrates with GSPO for sequence-level optimization
    """
    
    def __init__(self, config: CodeGenHRMConfig):
        self.config = config
        self.complexity_thresholds = {
            'simple': 0.3,      # Basic syntax, simple functions
            'medium': 0.6,      # Algorithms, data structures  
            'complex': 0.8,     # Complex algorithms, optimization
            'very_complex': 1.0 # Multi-file, advanced patterns
        }
    
    def determine_cycles(self, 
                        complexity_score: torch.Tensor,
                        language_complexity: torch.Tensor,
                        task_type: torch.Tensor) -> Tuple[int, int]:
        """Determine optimal H and L cycles based on problem characteristics"""
        
        base_h_cycles = self.config.H_cycles
        base_l_cycles = self.config.L_cycles
        
        # Adjust for problem complexity
        if complexity_score.mean() > self.complexity_thresholds['complex']:
            h_cycles = min(base_h_cycles + 3, self.config.max_cycles_complex)
            l_cycles = min(base_l_cycles + 4, self.config.max_cycles_complex)
        elif complexity_score.mean() > self.complexity_thresholds['medium']:
            h_cycles = base_h_cycles + 1
            l_cycles = base_l_cycles + 2
        elif complexity_score.mean() < self.complexity_thresholds['simple']:
            h_cycles = max(base_h_cycles - 1, self.config.min_cycles_simple)
            l_cycles = max(base_l_cycles - 1, self.config.min_cycles_simple)
        else:
            h_cycles = base_h_cycles
            l_cycles = base_l_cycles
        
        # Language-specific adjustments
        # More complex languages (Rust, C++) need more implementation cycles
        if language_complexity.mean() > 0.8:  # Rust, C++
            l_cycles += 2
        elif language_complexity.mean() < 0.4:  # Python, JavaScript
            l_cycles = max(l_cycles - 1, self.config.min_cycles_simple)
        
        return h_cycles, l_cycles


class MultiLanguageOutputHead(nn.Module):
    """
    Multi-language output generation with support for different code generation tasks
    Supports direct generation, diff-based editing, and tool command generation
    """
    
    def __init__(self, config: CodeGenHRMConfig):
        super().__init__()
        self.config = config
        
        # Shared code generation head
        self.shared_lm_head = CastedLinear(
            config.hidden_size, 
            config.shared_vocab_size, 
            bias=False
        )
        
        # Language-specific heads
        self.language_heads = nn.ModuleDict({
            lang.name.lower(): CastedLinear(config.hidden_size, config.lang_vocab_size, bias=False)
            for lang in LanguageType
        })
        
        # Task-specific heads
        if config.diff_generation_heads:
            self.diff_search_head = CastedLinear(config.hidden_size, config.shared_vocab_size, bias=False)
            self.diff_replace_head = CastedLinear(config.hidden_size, config.shared_vocab_size, bias=False)
        
        if config.tool_command_head:
            self.tool_command_head = CastedLinear(config.hidden_size, config.shared_vocab_size, bias=False)
        
        # Output mode router
        self.output_router = CastedLinear(config.hidden_size, len(CodeGenerationTask), bias=True)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                language_ids: torch.Tensor,
                task_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate outputs for multiple code generation scenarios"""
        
        outputs = {}
        
        # Shared vocabulary logits
        shared_logits = self.shared_lm_head(hidden_states)
        outputs['shared'] = shared_logits
        
        # Language-specific logits
        batch_size = hidden_states.size(0)
        lang_logits_list = []
        
        for i in range(batch_size):
            lang_id = language_ids[i].item()
            lang_name = LanguageType(lang_id).name.lower()
            lang_logits = self.language_heads[lang_name](hidden_states[i:i+1])
            lang_logits_list.append(lang_logits)
        
        outputs['language_specific'] = torch.cat(lang_logits_list, dim=0)
        
        # Combined logits for generation
        combined_logits = torch.cat([shared_logits, outputs['language_specific']], dim=-1)
        outputs['generation'] = combined_logits
        
        # Task-specific outputs
        if hasattr(self, 'diff_search_head') and task_ids is not None:
            diff_mask = (task_ids == CodeGenerationTask.DIFF_EDIT.value).unsqueeze(-1).unsqueeze(-1)
            outputs['diff_search'] = self.diff_search_head(hidden_states) * diff_mask
            outputs['diff_replace'] = self.diff_replace_head(hidden_states) * diff_mask
        
        if hasattr(self, 'tool_command_head') and task_ids is not None:
            tool_mask = (task_ids == CodeGenerationTask.TOOL_USE.value).unsqueeze(-1).unsqueeze(-1)
            outputs['tool_commands'] = self.tool_command_head(hidden_states) * tool_mask
        
        # Output routing (which head to use)
        outputs['task_routing'] = self.output_router(hidden_states.mean(dim=1))  # Pool sequence
        
        return outputs


class CodeGenHRM_Inner(HierarchicalReasoningModel_ACTV1_Inner):
    """Enhanced HRM inner model for code generation"""
    
    def __init__(self, config: CodeGenHRMConfig):
        # Initialize base config
        base_config = HierarchicalReasoningModel_ACTV1Config(**{
            k: v for k, v in config.__dict__.items() 
            if k in HierarchicalReasoningModel_ACTV1Config.__annotations__
        })
        super().__init__(base_config)
        
        self.code_config = config
        
        # Replace standard embeddings with code-specific ones
        embed_config = CodeEmbeddingConfig(
            vocab_size=config.code_vocab_size,
            shared_vocab_size=config.shared_vocab_size,
            hidden_size=config.hidden_size,
            num_languages=config.num_languages,
            max_seq_len=config.seq_len
        )
        
        self.code_embeddings = MultiLanguageCodeEmbedding(embed_config)
        
        # Replace puzzle embeddings with code-specific sparse embeddings
        if config.puzzle_emb_ndim > 0:
            self.code_puzzle_emb = CodeSpecificSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0.0,
                cast_to=self.forward_dtype,
                num_languages=config.num_languages,
                num_tasks=config.num_code_tasks
            )
        
        # Replace reasoning modules with enhanced versions
        self.H_level = CodeGenReasoningModule(
            layers=[CodeGenHierarchicalBlock(config, "high") for _ in range(config.H_layers)],
            level="high",
            config=config
        )
        self.L_level = CodeGenReasoningModule(
            layers=[CodeGenHierarchicalBlock(config, "low") for _ in range(config.L_layers)],
            level="low", 
            config=config
        )
        
        # Multi-language output generation
        self.multi_output_head = MultiLanguageOutputHead(config)
        
        # Complexity-aware ACT controller
        self.act_controller = ComplexityAwareACTController(config)
        
        # Enhanced Q-head with complexity awareness
        self.enhanced_q_head = CastedLinear(config.hidden_size + 1, 2, bias=True)  # +1 for complexity
        
        # Language complexity mapping
        self.language_complexity = {
            LanguageType.PYTHON: 0.3,
            LanguageType.JAVASCRIPT: 0.4,
            LanguageType.JAVA: 0.6,
            LanguageType.GO: 0.5,
            LanguageType.CPP: 0.8,
            LanguageType.RUST: 0.9
        }
    
    def _enhanced_input_embeddings(self, 
                                  inputs: torch.Tensor, 
                                  puzzle_identifiers: torch.Tensor,
                                  language_ids: torch.Tensor,
                                  task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced input embeddings with multi-language support"""
        
        # Multi-language token embeddings
        token_embeddings = self.code_embeddings(inputs, language_ids)
        
        # Code-specific puzzle embeddings
        if self.code_config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.code_puzzle_emb(
                puzzle_identifiers, 
                language_ids, 
                task_ids
            )
            
            # Reshape and pad puzzle embeddings
            pad_count = self.puzzle_emb_len * self.code_config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.code_config.hidden_size)
            token_embeddings = torch.cat((puzzle_embedding, token_embeddings), dim=-2)
        
        # Scale embeddings
        return self.embed_scale * token_embeddings
    
    def forward(self, 
                carry: HierarchicalReasoningModel_ACTV1InnerCarry, 
                batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        
        # Sequence information
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # Enhanced input processing
        input_embeddings = self._enhanced_input_embeddings(
            batch["inputs"], 
            batch["puzzle_identifiers"],
            batch["language_ids"],
            batch.get("task_ids")
        )
        
        # Get complexity information
        complexity_scores = batch.get("complexity_scores", torch.zeros(batch["inputs"].size(0)))
        language_complexity = torch.tensor([
            self.language_complexity[LanguageType(lid.item())] 
            for lid in batch["language_ids"]
        ])
        
        # Determine adaptive cycle allocation
        if self.code_config.complexity_aware_act:
            h_cycles, l_cycles = self.act_controller.determine_cycles(
                complexity_scores, 
                language_complexity,
                batch.get("task_ids", torch.zeros_like(batch["language_ids"]))
            )
        else:
            h_cycles, l_cycles = self.code_config.H_cycles, self.code_config.L_cycles
        
        # Language context for level communication
        language_context = self.code_embeddings.language_type_embeddings(batch["language_ids"])
        
        # Forward iterations with enhanced reasoning
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            
            for h_step in range(h_cycles):
                for l_step in range(l_cycles):
                    if not ((h_step == h_cycles - 1) and (l_step == l_cycles - 1)):
                        z_L = self.L_level(
                            z_L, 
                            z_H + input_embeddings,
                            language_context=language_context,
                            complexity_info=complexity_scores,
                            other_level_states=z_H,
                            **seq_info
                        )
                
                if not (h_step == h_cycles - 1):
                    z_H = self.H_level(
                        z_H, 
                        z_L,
                        language_context=language_context,
                        complexity_info=complexity_scores,
                        other_level_states=z_L,
                        **seq_info
                    )
        
        # Final gradient step
        z_L = self.L_level(
            z_L, 
            z_H + input_embeddings,
            language_context=language_context,
            other_level_states=z_H,
            **seq_info
        )
        z_H = self.H_level(
            z_H, 
            z_L,
            language_context=language_context,
            other_level_states=z_L,
            **seq_info
        )
        
        # Multi-language output generation
        outputs = self.multi_output_head(
            z_H[:, self.puzzle_emb_len:],
            batch["language_ids"],
            batch.get("task_ids")
        )
        
        # Enhanced Q-learning with complexity awareness
        q_input_features = torch.cat([
            z_H[:, 0],  # First token (CLS-like)
            complexity_scores.unsqueeze(-1)
        ], dim=-1)
        
        q_logits = self.enhanced_q_head(q_input_features).to(torch.float32)
        
        # New carry
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), 
            z_L=z_L.detach()
        )
        
        return new_carry, outputs, (q_logits[..., 0], q_logits[..., 1])


class CodeGenHRM(HierarchicalReasoningModel_ACTV1):
    """Enhanced HRM model for code generation tasks"""
    
    def __init__(self, config_dict: dict):
        # Convert to code generation config
        self.config = CodeGenHRMConfig(**config_dict)
        
        # Initialize with enhanced inner model
        nn.Module.__init__(self)  # Skip parent's __init__ to avoid double initialization
        self.inner = CodeGenHRM_Inner(self.config)
    
    @property
    def code_puzzle_emb(self):
        """Access to code-specific puzzle embeddings"""
        return self.inner.code_puzzle_emb
    
    def process_code_batch(self, code_inputs: List[CodeGenerationInput]) -> Dict[str, torch.Tensor]:
        """Process a batch of code generation inputs"""
        from models.code_generation.input_processor import CodeGenerationInputProcessor
        
        processor = CodeGenerationInputProcessor(
            vocab_size=self.config.code_vocab_size,
            hidden_size=self.config.hidden_size,
            max_seq_len=self.config.seq_len
        )
        
        return processor.batch_process(code_inputs)


if __name__ == "__main__":
    # Test the enhanced HRM code generation model
    config_dict = {
        "batch_size": 4,
        "seq_len": 512,
        "puzzle_emb_ndim": 512,
        "num_puzzle_identifiers": 1000,
        "vocab_size": 32000,  # Standard vocab
        "code_vocab_size": 40000,  # Extended for code
        "H_cycles": 2,
        "L_cycles": 4,
        "H_layers": 6,
        "L_layers": 6,
        "hidden_size": 512,
        "expansion": 2.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 10,
        "halt_exploration_prob": 0.1,
        "complexity_aware_act": True,
        "use_multi_head_output": True
    }
    
    model = CodeGenHRM(config_dict)
    
    # Create test batch
    batch_size = 4
    seq_len = 256
    
    test_batch = {
        "inputs": torch.randint(0, 40000, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, 1000, (batch_size,)),
        "language_ids": torch.randint(0, 6, (batch_size,)),
        "task_ids": torch.randint(0, 5, (batch_size,)),
        "complexity_scores": torch.rand(batch_size)
    }
    
    print("Testing Enhanced HRM Code Generation Model...")
    
    # Test forward pass
    carry = model.initial_carry(test_batch)
    new_carry, outputs = model(carry, test_batch)
    
    print(f"Output keys: {outputs.keys()}")
    print(f"Generation output shape: {outputs['logits']['generation'].shape}")
    print(f"Q halt logits shape: {outputs['q_halt_logits'].shape}")
    
    print("Enhanced HRM code generation model test completed successfully!")