"""Model source abstraction for supporting multiple model repositories."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import os


class ModelSource(str, Enum):
    """Supported model sources."""
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


class ModelSourceAdapter(ABC):
    """Abstract base class for model source adapters."""
    
    @abstractmethod
    def load_config(self, model_name: str) -> Any:
        """Load model configuration from the source.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model configuration object
        """
        pass
    
    @abstractmethod
    def download_file(self, model_name: str, filename: str, cache_dir: Optional[str] = None) -> str:
        """Download a file from the model repository.
        
        Args:
            model_name: Model identifier
            filename: File to download
            cache_dir: Optional cache directory
            
        Returns:
            Path to the downloaded file
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of the model source.
        
        Returns:
            Source name
        """
        pass


class HuggingFaceAdapter(ModelSourceAdapter):
    """Adapter for Hugging Face model hub."""
    
    def __init__(self):
        """Initialize Hugging Face adapter."""
        try:
            from transformers import AutoConfig
            from huggingface_hub import hf_hub_download
            self.AutoConfig = AutoConfig
            self.hf_hub_download = hf_hub_download
        except ImportError as e:
            raise ImportError(
                "Hugging Face dependencies not installed. "
                "Please install with: pip install transformers huggingface_hub"
            ) from e
    
    def load_config(self, model_name: str) -> Any:
        """Load model configuration from Hugging Face.
        
        Args:
            model_name: Hugging Face model identifier
            
        Returns:
            Model configuration object
        """
        return self.AutoConfig.from_pretrained(model_name)
    
    def download_file(self, model_name: str, filename: str, cache_dir: Optional[str] = None) -> str:
        """Download a file from Hugging Face model hub.
        
        Args:
            model_name: Hugging Face model identifier
            filename: File to download
            cache_dir: Optional cache directory
            
        Returns:
            Path to the downloaded file
        """
        return self.hf_hub_download(
            repo_id=model_name,
            filename=filename,
            cache_dir=cache_dir
        )
    
    def get_source_name(self) -> str:
        """Get the name of the model source."""
        return "Hugging Face"


class ModelScopeAdapter(ModelSourceAdapter):
    """Adapter for ModelScope (魔塔社区) model hub."""
    
    def __init__(self):
        """Initialize ModelScope adapter."""
        try:
            from modelscope import AutoConfig, snapshot_download
            self.AutoConfig = AutoConfig
            self.snapshot_download = snapshot_download
        except ImportError as e:
            raise ImportError(
                "ModelScope dependencies not installed. "
                "Please install with: pip install modelscope"
            ) from e
    
    def load_config(self, model_name: str) -> Any:
        """Load model configuration from ModelScope.
        
        Args:
            model_name: ModelScope model identifier
            
        Returns:
            Model configuration object
        """
        return self.AutoConfig.from_pretrained(model_name)
    
    def download_file(self, model_name: str, filename: str, cache_dir: Optional[str] = None) -> str:
        """Download a file from ModelScope model hub.
        
        Args:
            model_name: ModelScope model identifier
            filename: File to download
            cache_dir: Optional cache directory
            
        Returns:
            Path to the downloaded file
        """
        # ModelScope uses snapshot_download to get the model directory
        # then we construct the path to the specific file
        model_dir = self.snapshot_download(model_name, cache_dir=cache_dir)
        file_path = os.path.join(model_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in model {model_name}")
        
        return file_path
    
    def get_source_name(self) -> str:
        """Get the name of the model source."""
        return "ModelScope (魔塔社区)"


class ModelSourceFactory:
    """Factory for creating model source adapters."""
    
    _adapters = {
        ModelSource.HUGGINGFACE: HuggingFaceAdapter,
        ModelSource.MODELSCOPE: ModelScopeAdapter,
    }
    
    @classmethod
    def create_adapter(cls, source: ModelSource) -> ModelSourceAdapter:
        """Create a model source adapter.
        
        Args:
            source: Model source type
            
        Returns:
            Model source adapter instance
            
        Raises:
            ValueError: If source is not supported
        """
        if source not in cls._adapters:
            raise ValueError(f"Unsupported model source: {source}")
        
        adapter_class = cls._adapters[source]
        return adapter_class()
    
    @classmethod
    def get_adapter_from_string(cls, source_str: str) -> ModelSourceAdapter:
        """Create a model source adapter from string.
        
        Args:
            source_str: Model source name as string
            
        Returns:
            Model source adapter instance
            
        Raises:
            ValueError: If source string is not supported
        """
        try:
            source = ModelSource(source_str.lower())
            return cls.create_adapter(source)
        except ValueError:
            raise ValueError(
                f"Unsupported model source: {source_str}. "
                f"Supported sources: {', '.join([s.value for s in ModelSource])}"
            )
    
    @classmethod
    def get_default_adapter(cls) -> ModelSourceAdapter:
        """Get the default model source adapter.
        
        Returns:
            Default model source adapter (Hugging Face)
        """
        # Check environment variable for default source
        env_source = os.environ.get("LLM_SIM_MODEL_SOURCE", ModelSource.HUGGINGFACE.value)
        try:
            return cls.get_adapter_from_string(env_source)
        except ValueError:
            # Fallback to Hugging Face if environment variable is invalid
            return cls.create_adapter(ModelSource.HUGGINGFACE)
