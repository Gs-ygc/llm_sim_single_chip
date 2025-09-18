"""Model configuration loader for Hugging Face models."""

import json
from typing import Dict, Any, Optional
from transformers import AutoConfig
from huggingface_hub import hf_hub_download


class ModelConfigLoader:
    """Load and parse model configurations from Hugging Face."""
    
    def __init__(self):
        """Initialize the model configuration loader."""
        self.cache = {}
    
    def load_config(self, model_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load model configuration from Hugging Face.
        
        Args:
            model_name: Hugging Face model identifier (e.g., "Qwen/Qwen3-1.7B")
            use_cache: Whether to use cached configuration
            
        Returns:
            Dictionary containing model configuration
        """
        if use_cache and model_name in self.cache:
            return self.cache[model_name]
        
        try:
            # Load configuration using transformers
            config = AutoConfig.from_pretrained(model_name)
            
            # Convert to dictionary
            config_dict = config.to_dict()
            
            # Add model name for reference
            config_dict["model_name"] = model_name
            
            # Cache the configuration
            if use_cache:
                self.cache[model_name] = config_dict
            
            return config_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration for {model_name}: {e}")
    
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from local file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Dictionary containing model configuration
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return config_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information.
        
        Args:
            model_name: Hugging Face model identifier
            
        Returns:
            Dictionary containing model information
        """
        config = self.load_config(model_name)
        
        # Extract key information
        info = {
            "model_name": model_name,
            "hidden_size": config.get("hidden_size", 0),
            "intermediate_size": config.get("intermediate_size", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "num_key_value_heads": config.get("num_key_value_heads", 0),
            "num_hidden_layers": config.get("num_hidden_layers", 0),
            "head_dim": config.get("head_dim", 0),
            "vocab_size": config.get("vocab_size", 0),
            "max_position_embeddings": config.get("max_position_embeddings", 0),
            "model_type": config.get("model_type", "unknown"),
            "torch_dtype": config.get("torch_dtype", "float32"),
        }
        
        # Calculate derived metrics
        if info["hidden_size"] and info["num_attention_heads"]:
            info["head_dim_calculated"] = info["hidden_size"] // info["num_attention_heads"]
        
        if info["num_attention_heads"] and info["num_key_value_heads"]:
            info["group_size"] = info["num_attention_heads"] // info["num_key_value_heads"]
        
        # Estimate model size
        if info["hidden_size"] and info["num_hidden_layers"]:
            # Rough estimation: embedding + transformer layers + lm_head
            embedding_params = info["vocab_size"] * info["hidden_size"]
            transformer_params = info["num_hidden_layers"] * (
                info["hidden_size"] * info["hidden_size"] * 4 +  # attention
                info["hidden_size"] * info["intermediate_size"] * 2  # ffn
            )
            lm_head_params = info["vocab_size"] * info["hidden_size"]
            info["estimated_params"] = embedding_params + transformer_params + lm_head_params
        
        return info
    
    def print_model_info(self, model_name: str):
        """Print formatted model information.
        
        Args:
            model_name: Hugging Face model identifier
        """
        info = self.get_model_info(model_name)
        
        print(f"\n=== Model Information: {model_name} ===")
        print(f"Model Type: {info['model_type']}")
        print(f"Hidden Size: {info['hidden_size']:,}")
        print(f"Intermediate Size: {info['intermediate_size']:,}")
        print(f"Number of Layers: {info['num_hidden_layers']}")
        print(f"Attention Heads: {info['num_attention_heads']}")
        print(f"Key-Value Heads: {info['num_key_value_heads']}")
        print(f"Head Dimension: {info['head_dim']}")
        print(f"Group Size (GQA): {info.get('group_size', 'N/A')}")
        print(f"Vocabulary Size: {info['vocab_size']:,}")
        print(f"Max Position Embeddings: {info['max_position_embeddings']:,}")
        print(f"Estimated Parameters: {info.get('estimated_params', 'N/A'):,}")
        print(f"Data Type: {info['torch_dtype']}")
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    loader = ModelConfigLoader()
    
    # Load Qwen3-1.7B configuration
    config = loader.load_config("Qwen/Qwen3-1.7B")
    loader.print_model_info("Qwen/Qwen3-1.7B")
    
    # Load Qwen3-4B configuration
    loader.print_model_info("Qwen/Qwen3-4B")



