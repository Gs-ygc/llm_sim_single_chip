"""LLM Single Chip Simulation Framework.

A comprehensive framework for simulating and optimizing Large Language Model (LLM) 
inference on single-chip hardware accelerators.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config_loader import ModelConfigLoader
from .hardware_config import HardwareConfig, HardwarePresets
from .inference_config import InferenceConfig, InferencePresets
from .mma_analyzer import MMAAnalyzer, MMAConfig
from .hardware_recommender import HardwareRecommender, HardwareRecommendation

__all__ = [
    "ModelConfigLoader",
    "HardwareConfig",
    "HardwarePresets",
    "InferenceConfig",
    "InferencePresets",
    "MMAAnalyzer",
    "MMAConfig",
    "HardwareRecommender",
    "HardwareRecommendation",
]
