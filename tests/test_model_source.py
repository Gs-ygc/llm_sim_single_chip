"""Tests for model source abstraction."""

import pytest
import os
from llm_sim.model_source import (
    ModelSource, 
    ModelSourceFactory, 
    HuggingFaceAdapter,
    ModelScopeAdapter
)


def test_model_source_enum():
    """Test ModelSource enum values."""
    assert ModelSource.HUGGINGFACE.value == "huggingface"
    assert ModelSource.MODELSCOPE.value == "modelscope"


def test_factory_create_adapter():
    """Test factory creates correct adapters."""
    # Test Hugging Face adapter
    adapter_hf = ModelSourceFactory.create_adapter(ModelSource.HUGGINGFACE)
    assert isinstance(adapter_hf, HuggingFaceAdapter)
    assert adapter_hf.get_source_name() == "Hugging Face"
    
    # Note: ModelScope adapter will raise ImportError if modelscope is not installed
    # This is expected behavior
    try:
        adapter_ms = ModelSourceFactory.create_adapter(ModelSource.MODELSCOPE)
        assert isinstance(adapter_ms, ModelScopeAdapter)
        assert adapter_ms.get_source_name() == "ModelScope (魔塔社区)"
    except ImportError:
        pytest.skip("ModelScope not installed")


def test_factory_from_string():
    """Test factory creates adapter from string."""
    # Test case-insensitive
    adapter1 = ModelSourceFactory.get_adapter_from_string("huggingface")
    assert isinstance(adapter1, HuggingFaceAdapter)
    
    adapter2 = ModelSourceFactory.get_adapter_from_string("HUGGINGFACE")
    assert isinstance(adapter2, HuggingFaceAdapter)
    
    # Test invalid source
    with pytest.raises(ValueError):
        ModelSourceFactory.get_adapter_from_string("invalid_source")


def test_factory_default_adapter():
    """Test factory returns default adapter."""
    # Without environment variable
    if "LLM_SIM_MODEL_SOURCE" in os.environ:
        del os.environ["LLM_SIM_MODEL_SOURCE"]
    
    adapter = ModelSourceFactory.get_default_adapter()
    assert isinstance(adapter, HuggingFaceAdapter)


def test_factory_default_from_env():
    """Test factory uses environment variable."""
    # Set environment variable to modelscope
    os.environ["LLM_SIM_MODEL_SOURCE"] = "modelscope"
    
    try:
        adapter = ModelSourceFactory.get_default_adapter()
        # Should try to create ModelScope adapter
        # Will raise ImportError if not installed, which is fine
        assert adapter.get_source_name() in ["ModelScope (魔塔社区)", "Hugging Face"]
    except ImportError:
        pytest.skip("ModelScope not installed")
    finally:
        # Clean up
        if "LLM_SIM_MODEL_SOURCE" in os.environ:
            del os.environ["LLM_SIM_MODEL_SOURCE"]


def test_huggingface_adapter():
    """Test HuggingFace adapter initialization."""
    adapter = HuggingFaceAdapter()
    assert adapter.get_source_name() == "Hugging Face"
    assert hasattr(adapter, 'AutoConfig')
    assert hasattr(adapter, 'hf_hub_download')


def test_modelscope_adapter():
    """Test ModelScope adapter initialization."""
    try:
        adapter = ModelScopeAdapter()
        assert adapter.get_source_name() == "ModelScope (魔塔社区)"
        assert hasattr(adapter, 'AutoConfig')
        assert hasattr(adapter, 'snapshot_download')
    except ImportError:
        pytest.skip("ModelScope not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
