# API Reference

This document provides detailed API reference for the LLM Single Chip Simulation Framework.

## ModelConfigLoader

### `ModelConfigLoader()`

Initialize a model configuration loader.

**Returns:** `ModelConfigLoader` instance

### `load_config(model_name: str, use_cache: bool = True) -> Dict[str, Any]`

Load model configuration from Hugging Face.

**Parameters:**
- `model_name` (str): Hugging Face model identifier (e.g., "Qwen/Qwen3-1.7B")
- `use_cache` (bool): Whether to use cached configuration

**Returns:** Dictionary containing model configuration

**Raises:** `RuntimeError` if configuration loading fails

### `load_config_from_file(config_path: str) -> Dict[str, Any]`

Load model configuration from local file.

**Parameters:**
- `config_path` (str): Path to configuration JSON file

**Returns:** Dictionary containing model configuration

**Raises:** `RuntimeError` if file loading fails

### `get_model_info(model_name: str) -> Dict[str, Any]`

Get detailed model information.

**Parameters:**
- `model_name` (str): Hugging Face model identifier

**Returns:** Dictionary containing model information including:
- `model_name`: Model identifier
- `hidden_size`: Hidden layer dimension
- `intermediate_size`: FFN intermediate layer dimension
- `num_attention_heads`: Number of attention heads
- `num_key_value_heads`: Number of key-value heads
- `num_hidden_layers`: Number of transformer layers
- `head_dim`: Dimension per attention head
- `vocab_size`: Vocabulary size
- `max_position_embeddings`: Maximum position embeddings
- `model_type`: Model architecture type
- `torch_dtype`: Data type
- `head_dim_calculated`: Calculated head dimension
- `group_size`: Group size for GQA
- `estimated_params`: Estimated parameter count

### `print_model_info(model_name: str)`

Print formatted model information.

**Parameters:**
- `model_name` (str): Hugging Face model identifier

## HardwareConfig

### `HardwareConfig(...)`

Initialize hardware configuration.

**Parameters:**
- `sram_size` (int): SRAM size in bytes (default: 1MB)
- `dram_bandwidth` (int): DRAM bandwidth in bytes/cycle (default: 16)
- `mult_per_cycle` (int): Multipliers per cycle (default: 32×8×32)
- `bytes_per_element` (int): Input data bytes per element (default: 1 for FP8)
- `bytes_per_acc` (int): Accumulation bytes per element (default: 2 for BF16)
- `tile_quants` (tuple): Register tile dimensions (default: (64, 256, 64))
- `mtp_num` (int): Multi-threaded parallel number (default: 64)
- `sd_mal` (float): Speculative decode MAL (default: 4.0)
- `freq` (float): Frequency in Hz (default: 2GHz)

**Raises:** `ValueError` if configuration is invalid

### `get_memory_hierarchy_info() -> dict`

Get memory hierarchy information.

**Returns:** Dictionary containing:
- `sram_size_bytes`: SRAM size in bytes
- `sram_size_mb`: SRAM size in MB
- `dram_bandwidth_bytes_per_cycle`: DRAM bandwidth in bytes/cycle
- `dram_bandwidth_gb_per_sec`: DRAM bandwidth in GB/s

### `get_compute_info() -> dict`

Get compute unit information.

**Returns:** Dictionary containing:
- `multipliers_per_cycle`: Multipliers per cycle
- `frequency_hz`: Frequency in Hz
- `frequency_ghz`: Frequency in GHz
- `peak_compute_ops_per_sec`: Peak compute operations per second

### `get_data_type_info() -> dict`

Get data type information.

**Returns:** Dictionary containing:
- `input_bytes_per_element`: Input data bytes per element
- `accumulation_bytes_per_element`: Accumulation bytes per element
- `input_precision`: Input data precision
- `accumulation_precision`: Accumulation precision

### `get_tiling_info() -> dict`

Get tiling configuration information.

**Returns:** Dictionary containing:
- `register_tile_m`: Register tile M dimension
- `register_tile_k`: Register tile K dimension
- `register_tile_n`: Register tile N dimension
- `mtp_num`: Multi-threaded parallel number
- `sd_mal`: Speculative decode MAL

### `print_summary()`

Print hardware configuration summary.

## HardwarePresets

### `mobile_config() -> HardwareConfig`

Get mobile/edge device hardware configuration.

**Returns:** `HardwareConfig` optimized for mobile devices

### `datacenter_config() -> HardwareConfig`

Get datacenter/server hardware configuration.

**Returns:** `HardwareConfig` optimized for datacenter deployment

### `research_config() -> HardwareConfig`

Get research/experimental hardware configuration.

**Returns:** `HardwareConfig` optimized for research use cases

## Example Usage

```python
from llm_sim import ModelConfigLoader, HardwareConfig, HardwarePresets

# Load model configuration
loader = ModelConfigLoader()
config = loader.load_config("Qwen/Qwen3-1.7B")
loader.print_model_info("Qwen/Qwen3-1.7B")

# Create hardware configuration
hw_config = HardwareConfig()
hw_config.print_summary()

# Use predefined configurations
mobile_hw = HardwarePresets.mobile_config()
datacenter_hw = HardwarePresets.datacenter_config()
```



