# LLM Single Chip Simulation Framework

A comprehensive framework for simulating and optimizing Large Language Model (LLM) inference on single-chip hardware accelerators. This project provides cycle-accurate performance modeling, hardware-aware tiling optimization, and intelligent hardware recommendations for modern AI accelerators.

## 📋 TODO

### 🐛 Bug Fixes
- [ ] **Performance Calculation Bug**: Fix inaccurate performance calculation results for certain models
- [ ] **Memory Calculation Error**: Fix memory usage calculation deviation for large models
- [ ] **Tiling Optimization Bug**: Fix tiling strategy selection errors under specific hardware configurations
- [ ] **CLI Parameter Validation**: Improve command-line parameter validation and error handling
- [ ] **Model Configuration Parsing**: Fix parsing failures for certain Hugging Face model configurations

### 🚀 Core Feature Enhancements
- [ ] **Complete MoE Architecture Support**: 
  - [ ] Implement MoE routing mechanism simulation
  - [ ] Support dynamic expert selection
  - [ ] Optimize MoE memory access patterns
  - [ ] Add MoE-specific performance metrics
- [ ] **Cache Mechanism Optimization**:
  - [ ] Implement KV Cache simulation
  - [ ] Add attention caching strategies
  - [ ] Optimize cache hit rate calculation
  - [ ] Support different cache size configurations
- [ ] **Flash Attention Support**:
  - [ ] Complete Flash Attention algorithm implementation
  - [ ] Support different Flash Attention variants
  - [ ] Optimize memory access patterns

### 🔧 Hardware Simulation Enhancements
- [ ] **Multi-level Cache Hierarchy**:
  - [ ] Add L1/L2/L3 cache simulation
  - [ ] Implement cache coherence protocols
  - [ ] Support different cache replacement strategies
- [ ] **Parallel Computing Support**:
  - [ ] Multi-core parallel computing simulation
  - [ ] Data parallelism and model parallelism
  - [ ] Communication overhead modeling

### 📊 Performance Analysis Enhancements
- [ ] **Detailed Performance Analysis**:
  - [ ] Add more performance bottleneck analysis
  - [ ] Implement performance regression detection
  - [ ] Support performance comparison report generation
- [ ] **Visualization Improvements**:
  - [ ] Interactive performance charts
  - [ ] 3D hardware utilization visualization
  - [ ] Real-time performance monitoring interface
- [ ] **Benchmarking**:
  - [ ] Add standard benchmark test suites
  - [ ] Compare with real hardware performance
  - [ ] Automate performance regression testing

### 🎯 Model Support Extensions
- [ ] **More Model Architectures**:
- [ ] **Quantization Support**:
  - [ ] NVFP8/NVFP4 quantization simulation

### 🛠️ Development Tools Improvements
- [ ] **Test Coverage**:
  - [ ] Improve unit test coverage
  - [ ] Add integration tests
  - [ ] Automate performance testing
- [ ] **Documentation Enhancement**:
  - [ ] Auto-generate API documentation
  - [ ] Improve user guides
  - [ ] Expand tutorials and examples
- [ ] **CI/CD Optimization**:
  - [ ] Automated testing pipeline
  - [ ] Code quality checks
  - [ ] Automated release process

### 🔌 Compatibility
- [ ] **Hardware Platforms**:
  - [ ] Support more hardware architectures(H800/B300)
  - [ ] Real hardware validation
  - [ ] Hardware abstraction layer optimization

### 📈 Advanced Features
- [ ] **Auto-tuning**:
  - [ ] Automatic hardware configuration optimization
  - [ ] Intelligent tiling strategy selection
  - [ ] Adaptive performance tuning
- [ ] **Distributed Simulation**:
  - [ ] Multi-node simulation support
  - [ ] Network communication modeling
  - [ ] Load balancing strategies
- [ ] **Real-time Analysis**:
  - [ ] Real-time performance monitoring
  - [ ] Dynamic configuration adjustment
  - [ ] Online optimization recommendations



## 🚀 Features

- **Matrix Multiplication Accelerator (MMA) Simulation**: Cycle-accurate modeling of matrix multiplication operations
- **Hardware-Aware Tiling Optimization**: Automatic search for optimal tiling strategies based on hardware constraints
- **MoE (Mixture of Experts) Architecture Support**: Analysis of shared and routed experts in modern LLM architectures
- **Hardware Configuration Management**: Flexible hardware specifications with predefined presets
- **Performance Analysis & Visualization**: Detailed performance metrics and bottleneck analysis
- **Hardware Recommendation Engine**: Intelligent suggestions for optimal hardware configurations
- **Command-Line Interface**: Easy-to-use CLI for common analysis tasks
- **Hugging Face Integration**: Seamless support for models from Hugging Face Hub

## 📋 Quick Start

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/Gs-ygc/llm_sim_single_chip.git
cd llm_sim_single_chip

# Quick setup with automated script
./scripts/setup-env.sh  # Linux/Mac
# or
scripts\setup-env.bat   # Windows

# Manual setup - choose appropriate environment:
# CPU-only (recommended for most users):
conda env create -f environment-cpu.yml
conda activate llm-sim-single-chip-cpu

# GPU support (Linux with NVIDIA GPU):
conda env create -f environment-gpu.yml
conda activate llm-sim-single-chip-gpu

# Install package in development mode
pip install -e .
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/Gs-ygc/llm_sim_single_chip.git
cd llm_sim_single_chip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

#### Command Line Interface
```bash
# Analyze a model's performance
llm-sim analyze Qwen/Qwen3-1.7B

# Get hardware recommendations
llm-sim recommend Qwen/Qwen3-4B --use-case datacenter --target-tps 10.0

# Compare multiple models
llm-sim compare Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

# Compare hardware configurations
llm-sim compare-hardware Qwen/Qwen3-1.7B --hardware mobile datacenter research
```

#### Python API
```python
from llm_sim import ModelConfigLoader, MMAAnalyzer, HardwareConfig, HardwareRecommender

# Load model configuration
config_loader = ModelConfigLoader()
model_config = config_loader.load_config("Qwen/Qwen3-1.7B")

# Create hardware configuration
hw_config = HardwareConfig(
    sram_size=2*1024*1024,      # 2MB SRAM
    dram_bandwidth=32,           # 32 bytes/cycle
    mult_per_cycle=64*8*32,      # More compute units
)

# Analyze performance
analyzer = MMAAnalyzer(hw_config)
results = analyzer.analyze_model("Qwen/Qwen3-1.7B")

# Get hardware recommendations
recommender = HardwareRecommender()
recommendations = recommender.recommend_hardware(
    "Qwen/Qwen3-1.7B",
    target_tokens_per_second=5.0,
    use_case="datacenter"
)
```

## 🏗️ Project Structure

```
llm_sim_single_chip/
├── src/llm_sim/                    # Core framework
│   ├── __init__.py                 # Package initialization
│   ├── cli.py                      # Command-line interface
│   ├── config_loader.py            # Model configuration loading
│   ├── hardware_config.py          # Hardware specifications & presets
│   ├── inference_config.py         # Inference parameters
│   ├── mma_analyzer.py             # MMA performance analysis
│   ├── hardware_recommender.py     # Hardware recommendation engine
│   └── search_mma_tiling.py        # Tiling optimization algorithms
├── examples/                       # Example scripts
│   ├── qwen3_analysis.py          # Qwen3 model analysis
│   ├── complete_model_analysis.py  # Comprehensive analysis workflow
│   └── optimize_hardware_config.py # Hardware optimization
├── scripts/                        # Setup and utility scripts
│   ├── setup-env.sh               # Linux/Mac environment setup
│   └── setup-env.bat              # Windows environment setup
├── tests/                          # Test suite
├── docs/                           # Documentation
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## 🔧 Hardware Configurations

### Default Configuration
- **SRAM Size**: 1MB
- **DRAM Bandwidth**: 16 bytes/cycle
- **SRAM Bandwidth**: 256 bytes/cycle
- **Compute Units**: 32×8×32 multipliers/cycle
- **Data Types**: FP8 input, BF16 accumulation
- **Frequency**: 2GHz

### Predefined Presets

#### Mobile/Edge Configuration
```python
from llm_sim import HardwarePresets
hw_config = HardwarePresets.mobile_config()
# 512KB SRAM, 8 bytes/cycle DRAM, 1GHz frequency
```

#### Datacenter Configuration
```python
hw_config = HardwarePresets.datacenter_config()
# 4MB SRAM, 64 bytes/cycle DRAM, 3GHz frequency
```

#### Research Configuration
```python
hw_config = HardwarePresets.research_config()
# 8MB SRAM, 128 bytes/cycle DRAM, 2.5GHz frequency
```

## 📊 Supported Models

### Qwen3 Series
- **Qwen3-0.6B**: Lightweight model for edge deployment
- **Qwen3-1.7B**: Balanced performance and efficiency
- **Qwen3-4B**: High-performance model for datacenter
- **Qwen3-8B**: Large model for research applications
- **Qwen3-14B**: Very large model for advanced research
- **Qwen3-32B**: Massive model for cutting-edge research

### MoE (Mixture of Experts) Models
- **Qwen3-30B-A3B**: 30B parameter MoE with 3B active parameters
- **Qwen3-235B-A22B**: 235B parameter MoE with 22B active parameters
- **DeepSeek v3**: Advanced MoE architecture

### Custom Models
Support for any Hugging Face model with standard transformer architecture.

## 📈 Performance Metrics

### Key Metrics
- **MFU (Matrix Utilization Factor)**: Hardware utilization efficiency (0-1)
- **Tokens per Second**: Throughput performance
- **Per-Layer Time**: Detailed timing breakdown
- **Memory Usage**: SRAM and DRAM utilization
- **Bandwidth Requirements**: Memory bandwidth analysis

### Analysis Output
```
# HARDWARE RECOMMENDATIONS
========================================

#1 Datacenter Configuration
----------------------------------------
- Performance: 0.85/1.0
- Cost-effectiveness: 0.40/1.0
- Efficiency: 0.78/1.0
- Excellent performance with large memory capacity, suitable for datacenter deployment

## Hardware Specifications:
   SRAM: 4.0 MB
   DRAM Bandwidth: 192.0 GB/s
   Peak Compute: 12.3 TOPS

### Pros:
   - High throughput performance
   - Large SRAM for complex models
   - High compute capacity
   - Optimized for datacenter deployment

### Cons:
   - Higher cost compared to mobile configurations
```

## 🎯 Use Cases

### 1. Hardware Design
- Evaluate different hardware configurations
- Optimize memory hierarchy design
- Analyze compute unit requirements

### 2. Model Optimization
- Compare model architectures
- Analyze MoE vs dense models
- Optimize tiling strategies

### 3. Deployment Planning
- Select optimal hardware for specific models
- Estimate performance requirements
- Plan resource allocation

### 4. Research & Development
- Study hardware-software co-design
- Analyze performance bottlenecks
- Develop new optimization techniques

## 🔍 Advanced Features

### Tiling Optimization
```python
from llm_sim import MMAAnalyzer

analyzer = MMAAnalyzer()
# Automatic tiling optimization with verbose output
results = analyzer.analyze_model("Qwen/Qwen3-1.7B", verbose=True)
```

### Hardware Comparison
```python
from llm_sim import MMAAnalyzer, HardwarePresets

analyzer = MMAAnalyzer()
# Compare multiple hardware configurations
analyzer.compare_hardware_configs(
    "Qwen/Qwen3-1.7B",
    [HardwarePresets.mobile_config(), HardwarePresets.datacenter_config()]
)
```

### Model Comparison
```python
# Compare multiple models on the same hardware
analyzer.compare_models(
    ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
)
```

## 🛠️ Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mma_analyzer.py

# Run with coverage
pytest --cov=src/llm_sim
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Building Documentation
```bash
# Build Sphinx documentation
cd docs/
make html
```

## 📚 Examples

### Complete Model Analysis
```bash
# Run comprehensive analysis
python examples/complete_model_analysis.py
```

### Hardware Optimization
```bash
# Optimize hardware configurations
python examples/optimize_hardware_config.py
```

### Qwen3 Analysis
```bash
# Analyze Qwen3 model series
python examples/qwen3_analysis.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research in hardware-software co-design for LLM inference
- Inspired by modern AI accelerator architectures (TPU, GPU, specialized chips)
- Built with the Hugging Face ecosystem
- Thanks to the open-source community for foundational tools and libraries

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_sim_single_chip,
  title={LLM Single Chip Simulation Framework},
  author={Gs-ygc},
  year={2025},
  url={https://github.com/Gs-ygc/llm_sim_single_chip},
  note={A comprehensive framework for simulating and optimizing LLM inference on single-chip hardware accelerators}
}
```

## 🔗 Related Projects

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [MLPerf](https://mlcommons.org/en/inference/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Gs-y g c/llm_sim_single_chip/issues)
- **Discussions**: [GitHub Discussions](https://github.com/G/llm_sim_single_chip/discussions)
- **Documentation**: [Project Wiki](https://github.com/Gs-ygc/llm_sim_single_chip/wiki)

---

**Note**: This project focuses on hardware simulation and analysis, not running actual LLM inference. CPU-only environments are recommended for most users as they are faster to install, more compatible across platforms, and sufficient for simulation tasks.