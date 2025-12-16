"""Tests for ModelConfigLoader."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_sim.config_loader import ModelConfigLoader


class TestModelConfigLoader:
    """Test cases for ModelConfigLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ModelConfigLoader()
    
    def test_init(self):
        """Test ModelConfigLoader initialization."""
        assert isinstance(self.loader.cache, dict)
        assert len(self.loader.cache) == 0
        assert self.loader.adapter is not None
        assert self.loader.current_source is not None
    
    @patch('llm_sim.model_source.HuggingFaceAdapter.load_config')
    def test_load_config_success(self, mock_load_config):
        """Test successful configuration loading."""
        # Mock the adapter's load_config method
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "model_type": "qwen3"
        }
        mock_load_config.return_value = mock_config
        
        # Test loading
        config = self.loader.load_config("Qwen/Qwen3-1.7B")
        
        assert config["hidden_size"] == 2048
        assert config["num_attention_heads"] == 16
        assert config["model_name"] == "Qwen/Qwen3-1.7B"
        assert config["model_source"] == self.loader.current_source
        cache_key = f"{self.loader.current_source}:Qwen/Qwen3-1.7B"
        assert cache_key in self.loader.cache
    
    @patch('llm_sim.model_source.HuggingFaceAdapter.load_config')
    def test_load_config_failure(self, mock_load_config):
        """Test configuration loading failure."""
        mock_load_config.side_effect = Exception("Model not found")
        
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            self.loader.load_config("NonExistent/Model")
    
    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        # Create a temporary config file
        config_file = tmp_path / "config.json"
        config_data = {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "model_type": "test"
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test loading
        config = self.loader.load_config_from_file(str(config_file))
        
        assert config["hidden_size"] == 1024
        assert config["num_attention_heads"] == 8
        assert config["model_type"] == "test"
    
    def test_load_config_from_file_failure(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            self.loader.load_config_from_file("non_existent_file.json")
    
    @patch('llm_sim.model_source.HuggingFaceAdapter.load_config')
    def test_get_model_info(self, mock_load_config):
        """Test getting model information."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "model_type": "qwen3",
            "torch_dtype": "bfloat16"
        }
        mock_load_config.return_value = mock_config
        
        info = self.loader.get_model_info("Qwen/Qwen3-1.7B")
        
        assert info["model_name"] == "Qwen/Qwen3-1.7B"
        assert info["hidden_size"] == 2048
        assert info["num_attention_heads"] == 16
        assert info["head_dim_calculated"] == 128  # 2048 // 16
        assert info["group_size"] == 2  # 16 // 8
        assert "estimated_params" in info
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        # First load should populate cache
        with patch('llm_sim.model_source.HuggingFaceAdapter.load_config') as mock_load_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"test": "data"}
            mock_load_config.return_value = mock_config
            
            config1 = self.loader.load_config("Test/Model", use_cache=True)
            config2 = self.loader.load_config("Test/Model", use_cache=True)
            
            # Should only call load_config once due to caching
            assert mock_load_config.call_count == 1
            assert config1["test"] == config2["test"]


if __name__ == "__main__":
    pytest.main([__file__])



