"""Inference configuration for LLM simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for LLM inference parameters.
    
    This class defines software-level parameters that affect inference behavior
    but are not hardware-specific. These parameters can be tuned for different
    use cases and scenarios.
    """
    
    # Context and sequence parameters
    total_context_len: int = 8192  # Total context length
    fa_len_tile: int = 512  # Flash Attention tile length
    
    # Batch processing parameters
    default_batch_size: int = 1  # Default batch size for most operations
    flash_attention_batch_multiplier: int = 1  # Multiplier for Flash Attention batch size
    
    # Speculative decoding parameters
    sd_mal: float = 4.0  # Speculative decode Maximum Acceptable Length
    
    # Parallel processing parameters
    mtp_num: int = 64  # Multi-Token Parallelism number (parallel tokens per operation)
    
    # Memory management parameters
    memory_efficiency_mode: bool = False  # Enable aggressive memory optimization
    max_memory_usage_ratio: float = 0.9  # Maximum SRAM usage ratio
    
    # Performance tuning parameters
    enable_aggressive_tiling: bool = False  # Enable more aggressive tiling strategies
    min_mfu_threshold: float = 0.2  # Minimum acceptable MFU
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.total_context_len <= 0:
            raise ValueError("Total context length must be positive")
        if self.fa_len_tile <= 0:
            raise ValueError("Flash Attention tile length must be positive")
        if self.total_context_len % self.fa_len_tile != 0:
            raise ValueError("Total context length must be divisible by Flash Attention tile length")
        if self.default_batch_size <= 0:
            raise ValueError("Default batch size must be positive")
        if self.sd_mal <= 0:
            raise ValueError("Speculative decode MAL must be positive")
        if self.mtp_num <= 0:
            raise ValueError("Multi-Token Parallelism number must be positive")
        if not 0 < self.max_memory_usage_ratio <= 1:
            raise ValueError("Max memory usage ratio must be between 0 and 1")
        if not 0 < self.min_mfu_threshold <= 1:
            raise ValueError("Min MFU threshold must be between 0 and 1")
    
    def get_flash_attention_chunks(self) -> int:
        """Calculate number of Flash Attention chunks."""
        return self.total_context_len // self.fa_len_tile
    
    def get_flash_attention_batch_size(self, num_kv_heads: int) -> int:
        """Calculate Flash Attention batch size."""
        return num_kv_heads * self.get_flash_attention_chunks() * self.flash_attention_batch_multiplier
    
    def get_inference_info(self) -> dict:
        """Get inference configuration information."""
        return {
            "total_context_len": self.total_context_len,
            "fa_len_tile": self.fa_len_tile,
            "fa_chunks": self.get_flash_attention_chunks(),
            "default_batch_size": self.default_batch_size,
            "sd_mal": self.sd_mal,
            "mtp_num": self.mtp_num,
            "memory_efficiency_mode": self.memory_efficiency_mode,
            "max_memory_usage_ratio": self.max_memory_usage_ratio,
        }


class InferencePresets:
    """Predefined inference configurations for different use cases."""
    
    @staticmethod
    def standard() -> InferenceConfig:
        """Standard inference configuration."""
        return InferenceConfig()
    
    @staticmethod
    def memory_efficient() -> InferenceConfig:
        """Memory-efficient configuration for constrained environments."""
        return InferenceConfig(
            total_context_len=4096,
            fa_len_tile=256,
            default_batch_size=1,
            memory_efficiency_mode=True,
            max_memory_usage_ratio=0.8,
            enable_aggressive_tiling=True,
        )
    
    @staticmethod
    def high_throughput() -> InferenceConfig:
        """High-throughput configuration for batch processing."""
        return InferenceConfig(
            total_context_len=8192,
            fa_len_tile=512,
            default_batch_size=4,
            flash_attention_batch_multiplier=2,
            memory_efficiency_mode=False,
            max_memory_usage_ratio=0.95,
        )
    
    @staticmethod
    def long_context() -> InferenceConfig:
        """Long context configuration for extended sequences."""
        return InferenceConfig(
            total_context_len=32768,
            fa_len_tile=1024,
            default_batch_size=1,
            memory_efficiency_mode=True,
            max_memory_usage_ratio=0.85,
        )
    
    @staticmethod
    def research() -> InferenceConfig:
        """Research configuration with maximum flexibility."""
        return InferenceConfig(
            total_context_len=16384,
            fa_len_tile=512,
            default_batch_size=2,
            flash_attention_batch_multiplier=1,
            memory_efficiency_mode=False,
            max_memory_usage_ratio=0.9,
            enable_aggressive_tiling=True,
            min_mfu_threshold=0.15,
        )
