# Model Source Configuration Guide

This document explains how to configure and use different model sources (Hugging Face and ModelScope) with the LLM Single Chip Simulation framework.

## Overview

The framework supports loading models from two sources:
- **ModelScope (魔塔社区)** (default): https://modelscope.cn
- **Hugging Face Hub**: https://huggingface.co

This flexibility allows users to access models from different repositories based on their region, network conditions, or model availability.

## Installation

### ModelScope (Default)

ModelScope support is included by default:

```bash
pip install modelscope
```

### Hugging Face

To enable Hugging Face support, install the additional packages:

```bash
pip install transformers huggingface_hub
```

## Configuration Methods

### Method 1: Specify Source in Code

#### Using ModelConfigLoader

```python
from llm_sim import ModelConfigLoader

# Use ModelScope (default)
loader_ms = ModelConfigLoader(model_source="modelscope")
config_ms = loader_ms.load_config("qwen/Qwen3-1.7B")

# Use Hugging Face
loader_hf = ModelConfigLoader(model_source="huggingface")
config_hf = loader_hf.load_config("Qwen/Qwen3-1.7B")
```

#### Using MMAAnalyzer

```python
from llm_sim import MMAAnalyzer, HardwarePresets

hw_config = HardwarePresets.mobile_config()

# Analyze with Hugging Face
analyzer_hf = MMAAnalyzer(hw_config, model_source="huggingface")
results_hf = analyzer_hf.analyze_model("Qwen/Qwen3-1.7B")

# Analyze with ModelScope
analyzer_ms = MMAAnalyzer(hw_config, model_source="modelscope")
results_ms = analyzer_ms.analyze_model("qwen/Qwen3-1.7B")
```

#### Using HardwareRecommender

```python
from llm_sim import HardwareRecommender

# Recommend hardware for Hugging Face model
recommender_hf = HardwareRecommender(model_source="huggingface")
recommendations_hf = recommender_hf.recommend_hardware("Qwen/Qwen3-1.7B")

# Recommend hardware for ModelScope model
recommender_ms = HardwareRecommender(model_source="modelscope")
recommendations_ms = recommender_ms.recommend_hardware("qwen/Qwen3-1.7B")
```

### Method 2: Environment Variable

Set a default model source for all operations:

```bash
# Linux/Mac
export LLM_SIM_MODEL_SOURCE=modelscope

# Windows (Command Prompt)
set LLM_SIM_MODEL_SOURCE=modelscope

# Windows (PowerShell)
$env:LLM_SIM_MODEL_SOURCE="modelscope"
```

Then use the framework without specifying the source:

```python
from llm_sim import ModelConfigLoader

# Automatically uses ModelScope if LLM_SIM_MODEL_SOURCE is set
loader = ModelConfigLoader()
config = loader.load_config("qwen/Qwen3-1.7B")
```

### Method 3: CLI Arguments

Use the `--source` flag with CLI commands:

```bash
# Analyze from Hugging Face
llm-sim analyze Qwen/Qwen3-1.7B --source huggingface

# Analyze from ModelScope
llm-sim analyze qwen/Qwen3-1.7B --source modelscope

# Compare models from ModelScope
llm-sim compare qwen/Qwen3-0.6B qwen/Qwen3-1.7B --source modelscope

# Get hardware recommendations
llm-sim recommend qwen/Qwen3-4B --source modelscope --use-case datacenter
```

## Switching Between Sources

You can dynamically switch between sources:

```python
from llm_sim import ModelConfigLoader

loader = ModelConfigLoader(model_source="huggingface")

# Load from Hugging Face
config_hf = loader.load_config("Qwen/Qwen3-1.7B")

# Switch to ModelScope
loader.switch_source("modelscope")
config_ms = loader.load_config("qwen/Qwen3-1.7B")
```

## Model Naming Conventions

### Hugging Face
- Format: `organization/model-name`
- Example: `Qwen/Qwen3-1.7B`
- URL: https://huggingface.co/Qwen/Qwen3-1.7B

### ModelScope
- Format: `organization/model-name`
- Example: `qwen/Qwen3-1.7B`
- URL: https://modelscope.cn/models/qwen/Qwen3-1.7B

**Note**: Model names may differ slightly between platforms. Always verify the correct model identifier on the respective platform.

## Common Models on Both Platforms

| Model | Hugging Face | ModelScope |
|-------|-------------|------------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | `qwen/Qwen3-0.6B` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `qwen/Qwen3-1.7B` |
| Qwen3-4B | `Qwen/Qwen3-4B` | `qwen/Qwen3-4B` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `qwen/Qwen3-8B` |
| Qwen3-14B | `Qwen/Qwen3-14B` | `qwen/Qwen3-14B` |
| Qwen3-32B | `Qwen/Qwen3-32B` | `qwen/Qwen3-32B` |

## Troubleshooting

### ModelScope Not Installed

If you see an error like:

```
ImportError: ModelScope dependencies not installed.
Please install with: pip install modelscope
```

Solution:
```bash
pip install modelscope
```

### Model Not Found

If a model is not found on one platform, try:
1. Verify the model name on the platform's website
2. Check if the model exists on an alternative platform
3. Ensure you have network access to the platform

### Network Issues

#### For Hugging Face (China users)
If you have trouble accessing Hugging Face, use ModelScope:

```bash
export LLM_SIM_MODEL_SOURCE=modelscope
llm-sim analyze qwen/Qwen3-1.7B
```

#### For ModelScope (International users)
If ModelScope is slow or inaccessible, use Hugging Face:

```bash
export LLM_SIM_MODEL_SOURCE=huggingface
llm-sim analyze Qwen/Qwen3-1.7B
```

## Advanced Usage

### Custom Model Source Adapter

You can create custom adapters for additional model sources:

```python
from llm_sim.model_source import ModelSourceAdapter

class CustomAdapter(ModelSourceAdapter):
    def load_config(self, model_name: str):
        # Implement custom loading logic
        pass
    
    def download_file(self, model_name: str, filename: str, cache_dir=None):
        # Implement custom download logic
        pass
    
    def get_source_name(self) -> str:
        return "Custom Source"
```

### Caching Behavior

Both adapters use caching to improve performance:
- **Hugging Face**: Uses default HF cache directory (`~/.cache/huggingface/`)
- **ModelScope**: Uses default ModelScope cache directory (`~/.cache/modelscope/`)

To specify a custom cache directory:

```python
from llm_sim.model_source import ModelSourceFactory

adapter = ModelSourceFactory.get_adapter_from_string("modelscope")
file_path = adapter.download_file(
    model_name="qwen/Qwen3-1.7B",
    filename="config.json",
    cache_dir="/path/to/custom/cache"
)
```

## Examples

See the `examples/modelscope_integration.py` file for comprehensive examples demonstrating:
- Basic ModelScope usage
- Analyzing models from ModelScope
- Switching between sources
- Using environment variables
- Comparing models from different sources

Run the examples:

```bash
cd examples
python modelscope_integration.py
```

## Best Practices

1. **Set Default Source**: Use environment variable to set your preferred default source
2. **Verify Model Names**: Check model identifiers on the respective platform
3. **Handle Errors Gracefully**: Add try-except blocks for network or availability issues
4. **Cache Models**: Let the framework cache models to improve performance
5. **Regional Considerations**: Use ModelScope for better performance in China, Hugging Face elsewhere

## Support

For issues specific to:
- **Framework integration**: Open an issue on this repository
- **Hugging Face models**: Check Hugging Face documentation
- **ModelScope models**: Check ModelScope documentation (魔塔社区)
