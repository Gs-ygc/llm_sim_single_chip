#!/usr/bin/env python3
"""
Print Model Structure and Operators

This script prints detailed model structure, operators used, and acceleration recommendations.

Usage:
    python examples/print_model_structure.py <model_name>
    
Examples:
    python examples/print_model_structure.py Qwen/Qwen3-8B-FP8
    python examples/print_model_structure.py Qwen/Qwen3-1.7B
    python examples/print_model_structure.py meta-llama/Llama-3.1-8B
"""

import sys
import os
import argparse
import importlib
from typing import Dict, List, Tuple, Any, Optional

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_sim.mma_analyzer import MMAAnalyzer
from llm_sim.config_loader import ModelConfigLoader

class ModelStructureAnalyzer:
    """Analyzer for model structure and operators."""
    
    def __init__(self, batch_size: int = 1, input_seq_len: int = 512, output_seq_len: int = 1, torch_dtype: str = None):
        self.config_loader = ModelConfigLoader()
        self.analyzer = MMAAnalyzer()
        self._model_modules_cache = {}  # Cache for loaded model modules
        self.batch_size = batch_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.torch_dtype = torch_dtype  # Will be set from model config if None
    
    def _get_bytes_per_element(self, dtype: str) -> int:
        """Get bytes per element for a given torch dtype."""
        dtype_mapping = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
            "bool": 1,
        }
        return dtype_mapping.get(dtype.lower(), 4)  # Default to float32 if unknown
    
    def get_operator_info(self, operator_type: str, dimensions: Tuple[int, ...], dtype: str = None) -> Dict[str, Any]:
        """Get information about an operator including acceleration recommendations."""
        total_elements = 1
        for dim in dimensions:
            total_elements *= dim
        
        # Use provided dtype or fall back to instance dtype
        effective_dtype = dtype or self.torch_dtype or "float32"
        bytes_per_element = self._get_bytes_per_element(effective_dtype)
        
        # Determine acceleration type based on operator and dimensions
        if operator_type in ["linear", "matmul", "conv2d"]:
            if len(dimensions) >= 2 and min(dimensions) >= 16:
                acceleration_type = "TensorCore"
                acceleration_reason = "Large matrix operations suitable for TensorCore acceleration"
            else:
                acceleration_type = "Vector"
                acceleration_reason = "Smaller operations suitable for vector processing"
        elif operator_type in ["softmax", "layer_norm", "rms_norm"]:
            acceleration_type = "Vector"
            acceleration_reason = "Element-wise operations best suited for vector processing"
        elif operator_type in ["relu", "gelu", "silu", "swish"]:
            acceleration_type = "Vector"
            acceleration_reason = "Activation functions optimized for vector processing"
        elif operator_type in ["attention", "flash_attention"]:
            acceleration_type = "Mixed"
            acceleration_reason = "Combination of matrix operations (TensorCore) and element-wise ops (Vector)"
        else:
            acceleration_type = "Vector"
            acceleration_reason = "General element-wise operations"
        
        return {
            "total_elements": total_elements,
            "acceleration_type": acceleration_type,
            "acceleration_reason": acceleration_reason,
            "memory_usage_mb": total_elements * bytes_per_element / (1024 * 1024),
            "dtype": effective_dtype,
            "bytes_per_element": bytes_per_element
        }
    
    def _get_model_architecture_mapping(self) -> Dict[str, str]:
        """Map model architecture names to their corresponding transformers module paths."""
        return {
            # Qwen models
            "Qwen3MoeForCausalLM": "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3ForCausalLM": "transformers.models.qwen3.modeling_qwen3",
            "Qwen2MoeForCausalLM": "transformers.models.qwen2_moe.modeling_qwen2_moe",
            "Qwen2ForCausalLM": "transformers.models.qwen2.modeling_qwen2",
            "QwenForCausalLM": "transformers.models.qwen.modeling_qwen",
            
            # LLaMA models
            "LlamaForCausalLM": "transformers.models.llama.modeling_llama",
            "Llama3ForCausalLM": "transformers.models.llama.modeling_llama",
            "Llama2ForCausalLM": "transformers.models.llama.modeling_llama",
            
            # Mistral models
            "MistralForCausalLM": "transformers.models.mistral.modeling_mistral",
            "MixtralForCausalLM": "transformers.models.mixtral.modeling_mixtral",
            
            # GLM models
            "GLMForCausalLM": "transformers.models.glm.modeling_glm",
            "ChatGLMModel": "transformers.models.chatglm.modeling_chatglm",
            
            # BERT models
            "BertModel": "transformers.models.bert.modeling_bert",
            "BertForCausalLM": "transformers.models.bert.modeling_bert",
            
            # GPT models
            "GPT2LMHeadModel": "transformers.models.gpt2.modeling_gpt2",
            "GPTNeoForCausalLM": "transformers.models.gpt_neo.modeling_gpt_neo",
            "GPTNeoXForCausalLM": "transformers.models.gpt_neox.modeling_gpt_neox",
            
            # T5 models
            "T5ForConditionalGeneration": "transformers.models.t5.modeling_t5",
            
            # Add more mappings as needed
        }
    
    def _load_model_architecture_module(self, architecture_name: str) -> Optional[Any]:
        """Dynamically load the model architecture module based on architecture name."""
        if architecture_name in self._model_modules_cache:
            return self._model_modules_cache[architecture_name]
        
        mapping = self._get_model_architecture_mapping()
        if architecture_name not in mapping:
            print(f"Warning: No module mapping found for architecture '{architecture_name}'")
            return None
        
        module_path = mapping[architecture_name]
        try:
            print(f"Loading model architecture module: {module_path}")
            module = importlib.import_module(module_path)
            self._model_modules_cache[architecture_name] = module
            return module
        except ImportError as e:
            print(f"Warning: Could not import module {module_path}: {e}")
            return None
    
    def _analyze_model_architecture_from_module(self, model_info: Dict, architecture_module: Any) -> Dict[str, Any]:
        """Analyze model architecture using the loaded transformers module."""
        architecture_info = {
            "model_type": model_info.get("model_type", "unknown"),
            "architecture_class": "From transformers module",
            "layers": [],
            "operators": {},
            "parameters": {},
            "hierarchical_structure": {}
        }
        
        # Extract key parameters from model info
        hidden_size = model_info["hidden_size"]
        intermediate_size = model_info["intermediate_size"]
        num_attention_heads = model_info["num_attention_heads"]
        num_key_value_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        num_hidden_layers = model_info["num_hidden_layers"]
        vocab_size = model_info["vocab_size"]
        max_position_embeddings = model_info.get("max_position_embeddings", 0)
        
        # Calculate parameter counts based on typical transformer architecture
        total_params = self._calculate_parameter_count(
            hidden_size, intermediate_size, num_attention_heads, num_key_value_heads,
            head_dim, num_hidden_layers, vocab_size, max_position_embeddings
        )
        
        architecture_info["parameters"] = {
            "total": total_params,
            "trainable": total_params,
            "non_trainable": 0
        }
        
        # Build hierarchical structure
        self._build_hierarchical_structure(model_info, architecture_module, architecture_info)
        
        # Infer operators based on model info and architecture module
        self._infer_operators_from_architecture(model_info, architecture_module, architecture_info)
        
        return architecture_info
    
    def _calculate_parameter_count(self, hidden_size: int, intermediate_size: int, 
                                 num_attention_heads: int, num_key_value_heads: int,
                                 head_dim: int, num_hidden_layers: int, 
                                 vocab_size: int, max_position_embeddings: int) -> int:
        """Calculate total parameter count based on transformer architecture."""
        total_params = 0
        
        # Embedding layers
        if vocab_size > 0:
            total_params += vocab_size * hidden_size  # Token embeddings
        if max_position_embeddings > 0:
            total_params += max_position_embeddings * hidden_size  # Position embeddings
        
        # Transformer layers
        for _ in range(num_hidden_layers):
            # Attention layers
            total_params += hidden_size * (num_attention_heads * head_dim)  # Q projection
            total_params += hidden_size * (num_key_value_heads * head_dim)  # K projection  
            total_params += hidden_size * (num_key_value_heads * head_dim)  # V projection
            total_params += (num_attention_heads * head_dim) * hidden_size  # O projection
            
            # Feed-forward layers
            total_params += hidden_size * intermediate_size  # Up projection
            total_params += intermediate_size * hidden_size  # Down projection
            
            # Layer normalization (2 per layer typically)
            total_params += 2 * hidden_size * 2  # weight + bias for each norm
        
        # Final layer norm
        total_params += hidden_size * 2
        
        # Language modeling head (if applicable)
        if vocab_size > 0:
            total_params += hidden_size * vocab_size
        
        return total_params
    
    def _build_hierarchical_structure(self, model_info: Dict, architecture_module: Any, architecture_info: Dict[str, Any]):
        """Build hierarchical structure based on the model architecture."""
        hidden_size = model_info["hidden_size"]
        intermediate_size = model_info["intermediate_size"]
        num_attention_heads = model_info["num_attention_heads"]
        num_key_value_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        num_hidden_layers = model_info["num_hidden_layers"]
        vocab_size = model_info["vocab_size"]
        max_position_embeddings = model_info.get("max_position_embeddings", 0)
        
        # Check if this is a MoE model
        is_moe = hasattr(architecture_module, 'Qwen3MoeSparseMoeBlock') or hasattr(architecture_module, 'MixtralSparseMoeBlock')
        num_experts = model_info.get("num_experts", 0)
        moe_intermediate_size = model_info.get("moe_intermediate_size", intermediate_size)
        
        # Build the hierarchical structure
        model_name = model_info.get("model_name", "Unknown")
        architecture_name = model_info.get("architecture", "Unknown")
        structure = {
            "name": f"{model_name} ({architecture_name})",
            "type": "Model",
            "children": []
        }
        
        # Embedding layer
        if vocab_size > 0:
            structure["children"].append({
                "name": "embed_tokens",
                "type": "Embedding",
                "params": f"({vocab_size}, {hidden_size})",
                "children": []
            })
        
        # Rotary embedding
        structure["children"].append({
            "name": "rotary_emb",
            "type": "RotaryEmbedding",
            "params": f"(head_dim={head_dim})",
            "children": []
        })
        
        # Decoder layers
        decoder_layers = {
            "name": f"layers (×{num_hidden_layers})",
            "type": "ModuleList",
            "params": f"num_layers={num_hidden_layers}",
            "children": []
        }
        
        # Single decoder layer structure
        decoder_layer = {
            "name": "DecoderLayer",
            "type": "DecoderLayer",
            "params": f"(hidden_size={hidden_size})",
            "children": []
        }
        
        # Input layer norm
        decoder_layer["children"].append({
            "name": "input_layernorm",
            "type": "RMSNorm",
            "params": f"({hidden_size})",
            "children": []
        })
        
        # Determine attention type based on heads configuration
        if num_key_value_heads == 0:
            attention_type = "MultiHeadAttention"
            attention_params = f"(heads={num_attention_heads}, head_dim={head_dim})"
        elif num_key_value_heads == num_attention_heads:
            attention_type = "MultiHeadAttention"
            attention_params = f"(heads={num_attention_heads}, head_dim={head_dim})"
        elif num_key_value_heads == 1:
            attention_type = "MultiQueryAttention"
            attention_params = f"(heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim})"
        else:
            attention_type = "GroupedQueryAttention"
            attention_params = f"(heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim})"
        
        # Self attention with complete attention mechanism
        self_attn = {
            "name": "self_attn",
            "type": attention_type,
            "params": attention_params,
            "children": []
        }
        
        # Attention projections
        self_attn["children"].extend([
            {
                "name": "q_proj",
                "type": "Linear",
                "params": f"({hidden_size}, {num_attention_heads * head_dim})",
                "children": []
            },
            {
                "name": "k_proj", 
                "type": "Linear",
                "params": f"({hidden_size}, {num_key_value_heads * head_dim})",
                "children": []
            },
            {
                "name": "v_proj",
                "type": "Linear", 
                "params": f"({hidden_size}, {num_key_value_heads * head_dim})",
                "children": []
            }
        ])
        
        # Q and K normalization (for Qwen models)
        if hasattr(architecture_module, 'Qwen3MoeAttention') or hasattr(architecture_module, 'Qwen3Attention'):
            self_attn["children"].extend([
                {
                    "name": "q_norm",
                    "type": "RMSNorm",
                    "params": f"({head_dim})",
                    "children": []
                },
                {
                    "name": "k_norm",
                    "type": "RMSNorm", 
                    "params": f"({head_dim})",
                    "children": []
                }
            ])
        
        # Attention computation operations
        attention_ops = {
            "name": "attention_computation",
            "type": "AttentionOps",
            "params": f"(heads={num_attention_heads}, head_dim={head_dim})",
            "children": []
        }
        
        # Prefill phase (initial prompt processing)
        prefill_ops = {
            "name": "prefill_phase",
            "type": "PrefillPhase",
            "params": f"(processes full input sequence: {self.input_seq_len} tokens)",
            "children": []
        }
        
        prefill_ops["children"].extend([
            {
                "name": "qk_matmul",
                "type": "MatMul",
                "params": f"(Q: [{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {head_dim}] × K^T: [{self.batch_size}, {num_attention_heads}, {head_dim}, {self.input_seq_len}])",
                "children": []
            },
            {
                "name": "scaling",
                "type": "Scale",
                "params": f"(scale_factor=1/sqrt({head_dim})={1/(head_dim**0.5):.4f})",
                "children": []
            },
            {
                "name": "attention_scores",
                "type": "AttentionScores",
                "params": f"([{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {self.input_seq_len}])",
                "children": []
            },
            {
                "name": "attention_softmax",
                "type": "Softmax",
                "params": f"(dim=-1, [{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {self.input_seq_len}])",
                "children": []
            },
            {
                "name": "attention_weights",
                "type": "AttentionWeights",
                "params": f"([{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {self.input_seq_len}])",
                "children": []
            },
            {
                "name": "attn_v_matmul",
                "type": "MatMul",
                "params": f"(Attention: [{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {self.input_seq_len}] × V: [{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {head_dim}])",
                "children": []
            },
            {
                "name": "attention_output",
                "type": "AttentionOutput",
                "params": f"([{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {head_dim}])",
                "children": []
            },
            {
                "name": "kv_cache_store",
                "type": "KVCacheStore",
                "params": f"(store K,V: [{self.batch_size}, {num_key_value_heads}, {self.input_seq_len}, {head_dim}])",
                "children": []
            }
        ])
        
        # Decode phase (token generation with KV cache)
        decode_ops = {
            "name": "decode_phase",
            "type": "DecodePhase",
            "params": f"(generates {self.output_seq_len} tokens using KV cache)",
            "children": []
        }
        
        decode_ops["children"].extend([
            {
                "name": "qk_matmul",
                "type": "MatMul",
                "params": f"(Q: [{self.batch_size}, {num_attention_heads}, 1, {head_dim}] × K^T: [{self.batch_size}, {num_attention_heads}, {head_dim}, {self.input_seq_len + self.output_seq_len}])",
                "children": []
            },
            {
                "name": "scaling",
                "type": "Scale",
                "params": f"(scale_factor=1/sqrt({head_dim})={1/(head_dim**0.5):.4f})",
                "children": []
            },
            {
                "name": "attention_scores",
                "type": "AttentionScores",
                "params": f"([{self.batch_size}, {num_attention_heads}, 1, {self.input_seq_len + self.output_seq_len}])",
                "children": []
            },
            {
                "name": "attention_softmax",
                "type": "Softmax",
                "params": f"(dim=-1, [{self.batch_size}, {num_attention_heads}, 1, {self.input_seq_len + self.output_seq_len}])",
                "children": []
            },
            {
                "name": "attention_weights",
                "type": "AttentionWeights",
                "params": f"([{self.batch_size}, {num_attention_heads}, 1, {self.input_seq_len + self.output_seq_len}])",
                "children": []
            },
            {
                "name": "attn_v_matmul",
                "type": "MatMul",
                "params": f"(Attention: [{self.batch_size}, {num_attention_heads}, 1, {self.input_seq_len + self.output_seq_len}] × V: [{self.batch_size}, {num_attention_heads}, {self.input_seq_len + self.output_seq_len}, {head_dim}])",
                "children": []
            },
            {
                "name": "attention_output",
                "type": "AttentionOutput",
                "params": f"([{self.batch_size}, {num_attention_heads}, 1, {head_dim}])",
                "children": []
            },
            {
                "name": "kv_cache_append",
                "type": "KVCacheAppend",
                "params": f"(append new K,V: [{self.batch_size}, {num_key_value_heads}, 1, {head_dim}])",
                "children": []
            }
        ])
        
        attention_ops["children"].extend([prefill_ops, decode_ops])
        
        self_attn["children"].append(attention_ops)
        
        # Output projection
        self_attn["children"].append({
            "name": "o_proj",
            "type": "Linear",
            "params": f"({num_attention_heads * head_dim}, {hidden_size})",
            "children": []
        })
        
        decoder_layer["children"].append(self_attn)
        
        # Residual connection after attention
        decoder_layer["children"].append({
            "name": "residual_connection_1",
            "type": "ResidualAdd",
            "params": f"(hidden_states + attention_output)",
            "children": []
        })
        
        # Post attention layer norm
        decoder_layer["children"].append({
            "name": "post_attention_layernorm",
            "type": "RMSNorm",
            "params": f"({hidden_size})",
            "children": []
        })
        
        # MLP layer
        if is_moe and num_experts > 0:
            # MoE MLP
            mlp = {
                "name": "mlp",
                "type": "SparseMoEBlock",
                "params": f"(experts={num_experts}, intermediate_size={moe_intermediate_size})",
                "children": []
            }
            
            # MoE gate
            mlp["children"].append({
                "name": "gate",
                "type": "Linear",
                "params": f"({hidden_size}, {num_experts})",
                "children": []
            })
            
            # Router operations
            mlp["children"].extend([
                {
                    "name": "router_softmax",
                    "type": "Softmax",
                    "params": f"(dim=-1, [{self.batch_size}, {self.input_seq_len}, {num_experts}])",
                    "children": []
                },
                {
                    "name": "top_k_selection",
                    "type": "TopK",
                    "params": f"(k=2, select top {min(2, num_experts)} experts from {num_experts})",
                    "children": []
                }
            ])
            
            # MoE experts
            experts = {
                "name": f"experts (×{num_experts})",
                "type": "ModuleList",
                "params": f"num_experts={num_experts}",
                "children": []
            }
            
            # Single expert structure
            expert = {
                "name": "Expert",
                "type": "MLP",
                "params": f"(hidden_size={hidden_size}, intermediate_size={moe_intermediate_size})",
                "children": []
            }
            
            expert["children"].extend([
                {
                    "name": "gate_proj",
                    "type": "Linear",
                    "params": f"({hidden_size}, {moe_intermediate_size})",
                    "children": []
                },
                {
                    "name": "up_proj",
                    "type": "Linear",
                    "params": f"({hidden_size}, {moe_intermediate_size})",
                    "children": []
                },
                {
                    "name": "gate_up_combine",
                    "type": "ElementWiseMul",
                    "params": f"(gate_proj * up_proj)",
                    "children": []
                },
                {
                    "name": "act_fn",
                    "type": "SiLU",
                    "params": f"(activation on {moe_intermediate_size})",
                    "children": []
                },
                {
                    "name": "down_proj",
                    "type": "Linear",
                    "params": f"({moe_intermediate_size}, {hidden_size})",
                    "children": []
                }
            ])
            
            experts["children"].append(expert)
            mlp["children"].append(experts)
            
            # Expert combination
            mlp["children"].append({
                "name": "expert_combination",
                "type": "WeightedSum",
                "params": f"(weighted sum of selected experts)",
                "children": []
            })
            
            decoder_layer["children"].append(mlp)
        else:
            # Regular MLP
            mlp = {
                "name": "mlp",
                "type": "MLP",
                "params": f"(hidden_size={hidden_size}, intermediate_size={intermediate_size})",
                "children": []
            }
            
            mlp["children"].extend([
                {
                    "name": "gate_proj",
                    "type": "Linear",
                    "params": f"({hidden_size}, {intermediate_size})",
                    "children": []
                },
                {
                    "name": "up_proj",
                    "type": "Linear",
                    "params": f"({hidden_size}, {intermediate_size})",
                    "children": []
                },
                {
                    "name": "gate_up_combine",
                    "type": "ElementWiseMul",
                    "params": f"(gate_proj * up_proj)",
                    "children": []
                },
                {
                    "name": "act_fn",
                    "type": "SiLU",
                    "params": f"(activation on {intermediate_size})",
                    "children": []
                },
                {
                    "name": "down_proj",
                    "type": "Linear",
                    "params": f"({intermediate_size}, {hidden_size})",
                    "children": []
                }
            ])
            
            decoder_layer["children"].append(mlp)
        
        # Residual connection after MLP
        decoder_layer["children"].append({
            "name": "residual_connection_2",
            "type": "ResidualAdd",
            "params": f"(hidden_states + mlp_output)",
            "children": []
        })
        
        decoder_layers["children"].append(decoder_layer)
        structure["children"].append(decoder_layers)
        
        # Final layer norm
        structure["children"].append({
            "name": "norm",
            "type": "RMSNorm",
            "params": f"({hidden_size})",
            "children": []
        })
        
        # Language modeling head (if applicable)
        if vocab_size > 0:
            structure["children"].append({
                "name": "lm_head",
                "type": "Linear",
                "params": f"({hidden_size}, {vocab_size})",
                "children": []
            })
        
        architecture_info["hierarchical_structure"] = structure
    
    def _infer_operators_from_architecture(self, model_info: Dict, architecture_module: Any, architecture_info: Dict[str, Any]):
        """Infer operators from model info and architecture module."""
        hidden_size = model_info["hidden_size"]
        intermediate_size = model_info["intermediate_size"]
        num_attention_heads = model_info["num_attention_heads"]
        num_key_value_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        num_hidden_layers = model_info["num_hidden_layers"]
        vocab_size = model_info["vocab_size"]
        max_position_embeddings = model_info.get("max_position_embeddings", 0)
        
        operators = {}
        
        # Check if this is a MoE model
        is_moe = hasattr(architecture_module, 'Qwen3MoeSparseMoeBlock') or hasattr(architecture_module, 'MixtralSparseMoeBlock')
        num_experts = model_info.get("num_experts", 0)
        moe_intermediate_size = model_info.get("moe_intermediate_size", intermediate_size)
        
        # Linear operations
        linear_ops = []
        for i in range(num_hidden_layers):
            layer_prefix = f"layer_{i}"
            
            # Attention projections
            linear_ops.append({
                "name": f"{layer_prefix}.attention.query_projection",
                "type": "linear",
                "dimensions": (hidden_size, num_attention_heads * head_dim),
                "parameters": hidden_size * (num_attention_heads * head_dim),
                "memory_mb": hidden_size * (num_attention_heads * head_dim) * 4 / (1024 * 1024)
            })
            
            linear_ops.append({
                "name": f"{layer_prefix}.attention.key_projection", 
                "type": "linear",
                "dimensions": (hidden_size, num_key_value_heads * head_dim),
                "parameters": hidden_size * (num_key_value_heads * head_dim),
                "memory_mb": hidden_size * (num_key_value_heads * head_dim) * 4 / (1024 * 1024)
            })
            
            linear_ops.append({
                "name": f"{layer_prefix}.attention.value_projection",
                "type": "linear", 
                "dimensions": (hidden_size, num_key_value_heads * head_dim),
                "parameters": hidden_size * (num_key_value_heads * head_dim),
                "memory_mb": hidden_size * (num_key_value_heads * head_dim) * 4 / (1024 * 1024)
            })
            
            linear_ops.append({
                "name": f"{layer_prefix}.attention.output_projection",
                "type": "linear",
                "dimensions": (num_attention_heads * head_dim, hidden_size),
                "parameters": (num_attention_heads * head_dim) * hidden_size,
                "memory_mb": (num_attention_heads * head_dim) * hidden_size * 4 / (1024 * 1024)
            })
            
            # Feed-forward projections (MoE or regular)
            if is_moe and num_experts > 0:
                # MoE gate
                linear_ops.append({
                    "name": f"{layer_prefix}.mlp.gate",
                    "type": "linear",
                    "dimensions": (hidden_size, num_experts),
                    "parameters": hidden_size * num_experts,
                    "memory_mb": hidden_size * num_experts * 4 / (1024 * 1024)
                })
                
                # MoE expert projections (per expert)
                for expert_idx in range(num_experts):
                    linear_ops.append({
                        "name": f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj",
                        "type": "linear",
                        "dimensions": (hidden_size, moe_intermediate_size),
                        "parameters": hidden_size * moe_intermediate_size,
                        "memory_mb": hidden_size * moe_intermediate_size * 4 / (1024 * 1024)
                    })
                    
                    linear_ops.append({
                        "name": f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj",
                        "type": "linear",
                        "dimensions": (hidden_size, moe_intermediate_size),
                        "parameters": hidden_size * moe_intermediate_size,
                        "memory_mb": hidden_size * moe_intermediate_size * 4 / (1024 * 1024)
                    })
                    
                    linear_ops.append({
                        "name": f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj",
                        "type": "linear",
                        "dimensions": (moe_intermediate_size, hidden_size),
                        "parameters": moe_intermediate_size * hidden_size,
                        "memory_mb": moe_intermediate_size * hidden_size * 4 / (1024 * 1024)
                    })
            else:
                # Regular feed-forward projections
                linear_ops.append({
                    "name": f"{layer_prefix}.mlp.gate_proj",
                    "type": "linear",
                    "dimensions": (hidden_size, intermediate_size),
                    "parameters": hidden_size * intermediate_size,
                    "memory_mb": hidden_size * intermediate_size * 4 / (1024 * 1024)
                })
                
                linear_ops.append({
                    "name": f"{layer_prefix}.mlp.up_proj",
                    "type": "linear",
                    "dimensions": (hidden_size, intermediate_size),
                    "parameters": hidden_size * intermediate_size,
                    "memory_mb": hidden_size * intermediate_size * 4 / (1024 * 1024)
                })
                
                linear_ops.append({
                    "name": f"{layer_prefix}.mlp.down_proj",
                    "type": "linear",
                    "dimensions": (intermediate_size, hidden_size),
                    "parameters": intermediate_size * hidden_size,
                    "memory_mb": intermediate_size * hidden_size * 4 / (1024 * 1024)
                })
        
        operators["linear_operations"] = linear_ops
        
        # Attention operations
        attention_ops = []
        for i in range(num_hidden_layers):
            layer_prefix = f"layer_{i}"
            attention_ops.append({
                "name": f"{layer_prefix}.attention.qk_matmul",
                "type": "matmul",
                "dimensions": (head_dim, head_dim),
                "parameters": 0,  # No parameters for matmul
                "memory_mb": head_dim * head_dim * 4 / (1024 * 1024)
            })
            
            attention_ops.append({
                "name": f"{layer_prefix}.attention.pv_matmul",
                "type": "matmul", 
                "dimensions": (head_dim, head_dim),
                "parameters": 0,
                "memory_mb": head_dim * head_dim * 4 / (1024 * 1024)
            })
            
            attention_ops.append({
                "name": f"{layer_prefix}.attention.softmax",
                "type": "softmax",
                "dimensions": (num_attention_heads, head_dim, head_dim),
                "parameters": 0,
                "memory_mb": num_attention_heads * head_dim * head_dim * 4 / (1024 * 1024)
            })
        
        operators["attention_operations"] = attention_ops
        
        # Normalization operations
        norm_ops = []
        for i in range(num_hidden_layers):
            layer_prefix = f"layer_{i}"
            norm_ops.append({
                "name": f"{layer_prefix}.input_layernorm",
                "type": "rms_norm",  # Most modern models use RMSNorm
                "dimensions": (hidden_size,),
                "parameters": hidden_size,  # RMSNorm only has weight, no bias
                "memory_mb": hidden_size * 4 / (1024 * 1024)
            })
            
            norm_ops.append({
                "name": f"{layer_prefix}.post_attention_layernorm",
                "type": "rms_norm",
                "dimensions": (hidden_size,),
                "parameters": hidden_size,
                "memory_mb": hidden_size * 4 / (1024 * 1024)
            })
        
        # Final layer norm
        norm_ops.append({
            "name": "final_layernorm",
            "type": "rms_norm", 
            "dimensions": (hidden_size,),
            "parameters": hidden_size,
            "memory_mb": hidden_size * 4 / (1024 * 1024)
        })
        
        operators["norm_operations"] = norm_ops
        
        # Activation operations
        activation_ops = []
        for i in range(num_hidden_layers):
            layer_prefix = f"layer_{i}"
            activation_ops.append({
                "name": f"{layer_prefix}.mlp.activation",
                "type": "silu",  # Common activation in modern LLMs
                "dimensions": (intermediate_size if not is_moe else moe_intermediate_size,),
                "parameters": 0,
                "memory_mb": 0
            })
        
        operators["activation_operations"] = activation_ops
        
        # Embedding operations
        embedding_ops = []
        if vocab_size > 0:
            embedding_ops.append({
                "name": "token_embedding",
                "type": "embedding",
                "dimensions": (vocab_size, hidden_size),
                "parameters": vocab_size * hidden_size,
                "memory_mb": vocab_size * hidden_size * 4 / (1024 * 1024)
            })
        
        if max_position_embeddings > 0:
            embedding_ops.append({
                "name": "position_embedding",
                "type": "embedding",
                "dimensions": (max_position_embeddings, hidden_size),
                "parameters": max_position_embeddings * hidden_size,
                "memory_mb": max_position_embeddings * hidden_size * 4 / (1024 * 1024)
            })
        
        operators["embedding_operations"] = embedding_ops
        
        # MoE-specific operations
        if is_moe and num_experts > 0:
            moe_ops = []
            for i in range(num_hidden_layers):
                layer_prefix = f"layer_{i}"
                moe_ops.append({
                    "name": f"{layer_prefix}.mlp.router_softmax",
                    "type": "softmax",
                    "dimensions": (num_experts,),
                    "parameters": 0,
                    "memory_mb": num_experts * 4 / (1024 * 1024)
                })
                
                moe_ops.append({
                    "name": f"{layer_prefix}.mlp.top_k_selection",
                    "type": "top_k",
                    "dimensions": (num_experts,),
                    "parameters": 0,
                    "memory_mb": num_experts * 4 / (1024 * 1024)
                })
            
            operators["moe_operations"] = moe_ops
        
        architecture_info["operators"] = operators
    
    def analyze_model_structure(self, model_name: str) -> Dict[str, Any]:
        """Analyze the complete model structure and operators."""
        try:
            model_info = self.config_loader.get_model_info(model_name)
        except Exception as e:
            raise ValueError(f"Could not load model info for {model_name}: {e}")
        
        # Set torch_dtype from model config if not manually specified
        if self.torch_dtype is None:
            self.torch_dtype = model_info.get("torch_dtype", "float32")
        
        # Try to load the model architecture module
        architecture_name = model_info.get("architecture", "")
        architecture_module = None
        architecture_analysis = None
        
        if architecture_name:
            architecture_module = self._load_model_architecture_module(architecture_name)
            if architecture_module:
                print(f"Successfully loaded architecture module for {architecture_name}")
                architecture_analysis = self._analyze_model_architecture_from_module(model_info, architecture_module)
            else:
                print(f"Could not load architecture module for {architecture_name}, using fallback analysis")
        
        # Get model configurations for matrix operations
        configs, n_layers = self.analyzer.create_model_configs(model_name)
        
        # Analyze operators (fallback method)
        operators = self._analyze_operators(model_info, configs, n_layers)
        
        result = {
            "model_name": model_name,
            "model_info": model_info,
            "n_layers": n_layers,
            "operators": operators,
            "matrix_operations": configs,
            "architecture_loaded": architecture_module is not None,
            "architecture_name": architecture_name
        }
        
        if architecture_analysis:
            result["architecture_analysis"] = architecture_analysis
        
        return result
    
    def _analyze_operators(self, model_info: Dict, configs: List, n_layers: int) -> Dict[str, Any]:
        """Analyze all operators used in the model."""
        hidden_dim = model_info["hidden_size"]
        intermediate_size = model_info["intermediate_size"]
        num_attention_heads = model_info["num_attention_heads"]
        num_kv_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        vocab_size = model_info["vocab_size"]
        
        operators = {}
        
        # Matrix multiplication operations
        operators["linear_operations"] = {
            "up_projection": {
                "type": "linear",
                "dimensions": (hidden_dim, intermediate_size),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Feed-forward network up projection"
            },
            "down_projection": {
                "type": "linear", 
                "dimensions": (intermediate_size, hidden_dim),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Feed-forward network down projection"
            },
            "query_projection": {
                "type": "linear",
                "dimensions": (hidden_dim, num_attention_heads * head_dim),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Query projection in attention"
            },
            "key_value_projection": {
                "type": "linear",
                "dimensions": (hidden_dim, num_kv_heads * head_dim),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Key-Value projection in attention"
            },
            "output_projection": {
                "type": "linear",
                "dimensions": (num_attention_heads * head_dim, hidden_dim),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Output projection in attention"
            }
        }
        
        # Attention operations
        operators["attention_operations"] = {
            "qk_matmul": {
                "type": "matmul",
                "dimensions": (head_dim, head_dim),
                "count_per_layer": num_attention_heads,
                "total_count": n_layers * num_attention_heads,
                "description": "Query-Key matrix multiplication"
            },
            "pv_matmul": {
                "type": "matmul", 
                "dimensions": (head_dim, head_dim),
                "count_per_layer": num_attention_heads,
                "total_count": n_layers * num_attention_heads,
                "description": "Attention-Value matrix multiplication"
            },
            "attention_softmax": {
                "type": "softmax",
                "dimensions": (num_attention_heads, head_dim, head_dim),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Attention weights softmax normalization"
            }
        }
        
        # Normalization operations
        operators["normalization_operations"] = {
            "pre_attention_norm": {
                "type": "rms_norm",
                "dimensions": (hidden_dim,),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Pre-attention RMS normalization"
            },
            "pre_ffn_norm": {
                "type": "rms_norm",
                "dimensions": (hidden_dim,),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Pre-feed-forward RMS normalization"
            }
        }
        
        # Activation functions
        operators["activation_operations"] = {
            "ffn_activation": {
                "type": "silu",  # Common in modern LLMs
                "dimensions": (intermediate_size,),
                "count_per_layer": 1,
                "total_count": n_layers,
                "description": "Feed-forward network activation (SiLU/Swish)"
            }
        }
        
        # Embedding operations
        operators["embedding_operations"] = {
            "token_embedding": {
                "type": "embedding",
                "dimensions": (vocab_size, hidden_dim),
                "count_per_layer": 1,
                "total_count": 1,
                "description": "Token embedding lookup"
            },
            "position_embedding": {
                "type": "embedding",
                "dimensions": (model_info.get("max_position_embeddings", 2048), hidden_dim),
                "count_per_layer": 1,
                "total_count": 1,
                "description": "Position embedding lookup"
            }
        }
        
        # Add acceleration recommendations to all operators
        for category, ops in operators.items():
            for op_name, op_info in ops.items():
                op_info.update(self.get_operator_info(op_info["type"], op_info["dimensions"]))
        
        return operators
    
    def print_model_structure(self, model_name: str):
        """Print detailed model structure and operator analysis."""
        print(f"Model Structure Analysis: {model_name}")
        print("=" * 100)
        print(f"Configuration: batch_size={self.batch_size}, input_seq_len={self.input_seq_len}, output_seq_len={self.output_seq_len}")
        print("=" * 100)
        
        # Print inference phase analysis
        self._print_inference_phase_analysis(model_name)
        
        try:
            analysis = self.analyze_model_structure(model_name)
        except Exception as e:
            print(f"Error analyzing model: {e}")
            return
        
        model_info = analysis["model_info"]
        n_layers = analysis["n_layers"]
        operators = analysis["operators"]
        architecture_loaded = analysis.get("architecture_loaded", False)
        architecture_name = analysis.get("architecture_name", "Unknown")
        
        # Print model configuration
        print("Model Configuration:")
        print("-" * 50)
        print(f"  Model Name: {model_name}")
        print(f"  Architecture: {architecture_name}")
        print(f"  Architecture Module Loaded: {'Yes' if architecture_loaded else 'No'}")
        print(f"  Hidden Size: {model_info['hidden_size']:,}")
        print(f"  Intermediate Size: {model_info['intermediate_size']:,}")
        print(f"  Attention Heads: {model_info['num_attention_heads']}")
        print(f"  Key-Value Heads: {model_info['num_key_value_heads']}")
        print(f"  Head Dimension: {model_info['head_dim']}")
        print(f"  Number of Layers: {n_layers}")
        print(f"  Vocabulary Size: {model_info['vocab_size']:,}")
        print(f"  Max Position Embeddings: {model_info.get('max_position_embeddings', 'N/A'):,}")
        if 'estimated_params' in model_info:
            print(f"  Estimated Parameters: {model_info['estimated_params']:,}")
        print()
        
        # Print architecture analysis if available
        if "architecture_analysis" in analysis:
            self._print_architecture_analysis(analysis["architecture_analysis"])
            # Print hierarchical structure
            if "hierarchical_structure" in analysis["architecture_analysis"]:
                self._print_hierarchical_structure(analysis["architecture_analysis"]["hierarchical_structure"])
        
        # Print fallback analysis
        if not architecture_loaded:
            print("Note: Using fallback analysis (architecture module not loaded)")
            print()
        
        # Print operator analysis
        for category, ops in operators.items():
            print(f"{category.replace('_', ' ').title()}:")
            print("-" * 80)
            print(f"{'Operator':<25} {'Type':<12} {'Dimensions':<20} {'Count':<8} {'Total':<8} {'Acceleration':<12} {'Memory (MB)':<12}")
            print("-" * 80)
            
            for op_name, op_info in ops.items():
                dims_str = "×".join(map(str, op_info["dimensions"]))
                count_per_layer = op_info["count_per_layer"]
                total_count = op_info["total_count"]
                acceleration = op_info["acceleration_type"]
                memory_mb = op_info["memory_usage_mb"] * total_count
                
                print(f"{op_name:<25} {op_info['type']:<12} {dims_str:<20} {count_per_layer:<8} {total_count:<8} {acceleration:<12} {memory_mb:<12.2f}")
            
            print()
        
        # Print acceleration summary
        self._print_acceleration_summary(operators)
        
        # Print matrix operations summary
        self._print_matrix_operations_summary(analysis["matrix_operations"], n_layers)
        
        print("=" * 100)
    
    def _print_architecture_analysis(self, architecture_analysis: Dict[str, Any]):
        """Print detailed architecture analysis from transformers module."""
        print("Architecture Analysis (from transformers module):")
        print("-" * 60)
        
        # Print parameter summary
        params = architecture_analysis["parameters"]
        print(f"  Total Parameters: {params['total']:,}")
        print(f"  Trainable Parameters: {params['trainable']:,}")
        print(f"  Non-trainable Parameters: {params['non_trainable']:,}")
        print()
        
        # Print operators by category
        operators = architecture_analysis["operators"]
        if operators:
            for category, ops in operators.items():
                if ops:  # Only print categories that have operations
                    print(f"{category.replace('_', ' ').title()}:")
                    print("-" * 80)
                    print(f"{'Name':<40} {'Type':<15} {'Dimensions':<25} {'Parameters':<12} {'Memory (MB)':<12}")
                    print("-" * 80)
                    
                    for op in ops:
                        dims_str = str(op["dimensions"]) if isinstance(op["dimensions"], (tuple, list)) else str(op["dimensions"])
                        if len(dims_str) > 25:
                            dims_str = dims_str[:22] + "..."
                        print(f"{op['name']:<40} {op['type']:<15} {dims_str:<25} {op['parameters']:<12,} {op['memory_mb']:<12.2f}")
                    
                    print()
        
        print()
    
    def _print_hierarchical_structure(self, structure: Dict[str, Any], indent: int = 0):
        """Print hierarchical model structure in a tree-like format."""
        if indent == 0:
            print("Model Hierarchical Structure:")
            print("-" * 60)
        
        # Create indentation string
        indent_str = "  " * indent
        
        # Print current node
        name = structure["name"]
        node_type = structure["type"]
        params = structure.get("params", "")
        
        if params:
            print(f"{indent_str}{name}: {node_type}{params}")
        else:
            print(f"{indent_str}{name}: {node_type}")
        
        # Print children recursively
        children = structure.get("children", [])
        for child in children:
            self._print_hierarchical_structure(child, indent + 1)
        
        if indent == 0:
            print()
    
    def _print_inference_phase_analysis(self, model_name: str):
        """Print inference phase analysis focusing on prefill vs decode differences."""
        try:
            model_info = self.config_loader.get_model_info(model_name)
        except Exception as e:
            print(f"Warning: Could not load model info for inference analysis: {e}")
            return
        
        hidden_size = model_info["hidden_size"]
        num_attention_heads = model_info["num_attention_heads"]
        num_key_value_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        num_hidden_layers = model_info["num_hidden_layers"]
        
        # Get dtype and bytes per element
        dtype = self.torch_dtype or model_info.get("torch_dtype", "float32")
        bytes_per_element = self._get_bytes_per_element(dtype)
        
        print("Inference Phase Analysis:")
        print("-" * 60)
        print(f"Data Type: {dtype} ({bytes_per_element} bytes per element)")
        print()
        
        # Prefill phase analysis
        print("Prefill Phase (Initial Prompt Processing):")
        print(f"  Processes: {self.input_seq_len} tokens in parallel")
        print(f"  Attention Matrix: [{self.batch_size}, {num_attention_heads}, {self.input_seq_len}, {self.input_seq_len}]")
        print(f"  Memory Complexity: O({self.input_seq_len}²) per head")
        print(f"  KV Cache Size: {self.batch_size * num_key_value_heads * self.input_seq_len * head_dim * 2 * bytes_per_element / (1024**2):.2f} MB per layer")
        print(f"  Total KV Cache: {self.batch_size * num_key_value_heads * self.input_seq_len * head_dim * 2 * bytes_per_element * num_hidden_layers / (1024**2):.2f} MB")
        
        # Decode phase analysis
        print(f"\nDecode Phase (Token Generation):")
        print(f"  Generates: {self.output_seq_len} tokens sequentially")
        print(f"  Attention Matrix: [{self.batch_size}, {num_attention_heads}, 1, {self.input_seq_len + self.output_seq_len}]")
        print(f"  Memory Complexity: O({self.input_seq_len + self.output_seq_len}) per head")
        kv_cache_growth_per_token = self.batch_size * num_key_value_heads * 1 * head_dim * 2 * bytes_per_element / (1024**2)
        print(f"  KV Cache Growth: {kv_cache_growth_per_token:.4f} MB per token per layer")
        print(f"  Total Additional Cache: {self.batch_size * num_key_value_heads * self.output_seq_len * head_dim * 2 * bytes_per_element * num_hidden_layers / (1024**2):.2f} MB")
        
        # Performance implications
        print(f"\nPerformance Implications:")
        # Prefill: QK^T matmul + attention*V matmul
        prefill_qk_flops = self.batch_size * num_attention_heads * self.input_seq_len * self.input_seq_len * head_dim * 2
        prefill_av_flops = self.batch_size * num_attention_heads * self.input_seq_len * self.input_seq_len * head_dim * 2
        prefill_flops = prefill_qk_flops + prefill_av_flops
        
        # Decode: QK^T matmul + attention*V matmul (per token)
        decode_qk_flops = self.batch_size * num_attention_heads * 1 * (self.input_seq_len + self.output_seq_len) * head_dim * 2
        decode_av_flops = self.batch_size * num_attention_heads * 1 * (self.input_seq_len + self.output_seq_len) * head_dim * 2
        decode_flops_per_token = decode_qk_flops + decode_av_flops
        total_decode_flops = decode_flops_per_token * self.output_seq_len
        
        print(f"  Prefill FLOPs: {prefill_flops / 1e9:.2f} GFLOPs")
        print(f"  Decode FLOPs per token: {decode_flops_per_token / 1e6:.2f} MFLOPs")
        print(f"  Total Decode FLOPs: {total_decode_flops / 1e9:.2f} GFLOPs")
        print(f"  Prefill/Decode Ratio: {prefill_flops / total_decode_flops:.2f}x")
        
        # Memory bandwidth analysis
        print(f"\nMemory Bandwidth Analysis:")
        # Attention scores memory
        prefill_attention_scores = self.batch_size * num_attention_heads * self.input_seq_len * self.input_seq_len * bytes_per_element
        decode_attention_scores = self.batch_size * num_attention_heads * 1 * (self.input_seq_len + self.output_seq_len) * bytes_per_element
        
        print(f"  Prefill Memory Access: {prefill_attention_scores / (1024**2):.2f} MB")
        print(f"  Decode Memory Access per token: {decode_attention_scores / (1024**2):.2f} MB")
        print(f"  Memory Access Ratio: {prefill_attention_scores / decode_attention_scores:.2f}x")
        
        print()
    
    def _print_acceleration_summary(self, operators: Dict[str, Any]):
        """Print acceleration recommendations summary."""
        print("Acceleration Recommendations Summary:")
        print("-" * 50)
        
        acceleration_counts = {"TensorCore": 0, "Vector": 0, "Mixed": 0}
        total_memory = 0
        
        for category, ops in operators.items():
            for op_name, op_info in ops.items():
                acc_type = op_info["acceleration_type"]
                acceleration_counts[acc_type] += op_info["total_count"]
                total_memory += op_info["memory_usage_mb"] * op_info["total_count"]
        
        print(f"  TensorCore Operations: {acceleration_counts['TensorCore']:,}")
        print(f"  Vector Operations: {acceleration_counts['Vector']:,}")
        print(f"  Mixed Operations: {acceleration_counts['Mixed']:,}")
        print(f"  Total Memory Usage: {total_memory:.2f} MB")
        print()
        
        print("Hardware Recommendations:")
        print("  - TensorCore: Optimize for large matrix multiplications")
        print("  - Vector Units: Optimize for element-wise operations (activations, norms)")
        print("  - Memory Bandwidth: Critical for embedding lookups and attention")
        print()
    
    def _print_matrix_operations_summary(self, matrix_ops: List, n_layers: int):
        """Print summary of matrix operations."""
        print("Matrix Operations Summary:")
        print("-" * 50)
        
        total_elements = 0
        total_memory_mb = 0
        
        for config in matrix_ops:
            m, k, n = config.m, config.k, config.n
            elements = m * k + k * n
            memory_mb = elements * 0.5 / (1024 * 1024)  # Assuming FP16 weights
            total_elements += elements
            total_memory_mb += memory_mb
        
        print(f"  Operations per layer: {len(matrix_ops)}")
        print(f"  Total operations: {len(matrix_ops) * n_layers}")
        print(f"  Elements per layer: {total_elements:,}")
        print(f"  Total elements: {total_elements * n_layers:,}")
        print(f"  Memory per layer: {total_memory_mb:.2f} MB")
        print(f"  Total memory: {total_memory_mb * n_layers:.2f} MB")
        print()


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Print model structure and operator analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/print_model_structure.py Qwen/Qwen3-8B-FP8
  python examples/print_model_structure.py Qwen/Qwen3-1.7B --batch-size 4 --input-seq-len 1024
  python examples/print_model_structure.py meta-llama/Llama-3.1-8B --batch-size 2 --input-seq-len 2048 --output-seq-len 128
  python examples/print_model_structure.py Qwen/Qwen3-30B-A3B --torch-dtype bfloat16 --input-seq-len 1024 --output-seq-len 128
        """
    )
    
    parser.add_argument(
        "model_name", 
        help="Hugging Face model identifier (e.g., Qwen/Qwen3-8B-FP8)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for inference (default: 1)"
    )
    
    parser.add_argument(
        "--input-seq-len", 
        type=int, 
        default=512,
        help="Input sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--output-seq-len", 
        type=int, 
        default=1,
        help="Output sequence length for generation (default: 1)"
    )
    
    parser.add_argument(
        "--torch-dtype", 
        type=str, 
        default=None,
        help="Override torch dtype (e.g., float32, float16, bfloat16). If not specified, uses model config dtype."
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer with specified parameters and print model structure
        analyzer = ModelStructureAnalyzer(
            batch_size=args.batch_size,
            input_seq_len=args.input_seq_len,
            output_seq_len=args.output_seq_len,
            torch_dtype=args.torch_dtype
        )
        analyzer.print_model_structure(args.model_name)
    except BrokenPipeError:
        # Handle broken pipe gracefully (e.g., when using head, tail, etc.)
        import sys
        sys.stderr.close()
        sys.exit(0)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
