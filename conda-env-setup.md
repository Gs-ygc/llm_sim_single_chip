# Conda Environment Setup Guide

This guide provides step-by-step instructions for setting up the LLM Single Chip Simulation environment using Conda.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed
- CUDA-compatible GPU (optional, for GPU acceleration)

## Quick Setup

### 1. Create Environment from YAML

```bash
# Clone the repository
git clone <repository-url>
cd llm_sim_single_chip

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate llm-sim-single-chip
```

### 2. Install Package in Development Mode

```bash
# Install the package in editable mode
pip install -e .
```

## Detailed Setup Options

### Option A: Using environment.yml (Recommended)

```bash
# Create environment with specific Python version
conda env create -f environment.yml

# Or create with a different name
conda env create -f environment.yml -n my-llm-sim

# Activate environment
conda activate llm-sim-single-chip
```

### Option B: Manual Environment Creation

```bash
# Create new environment
conda create -n llm-sim-single-chip python=3.10

# Activate environment
conda activate llm-sim-single-chip

# Add conda-forge channel
conda config --add channels conda-forge

# Install core packages
conda install numpy scipy pandas matplotlib seaborn
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers huggingface_hub -c huggingface

# Install development tools
conda install pytest black flake8 mypy jupyter

# Install remaining packages via pip
pip install -r requirements.txt
```

### Option C: GPU Support Setup

For CUDA 11.8:
```bash
conda create -n llm-sim-single-chip python=3.10
conda activate llm-sim-single-chip
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers huggingface_hub -c huggingface
pip install -r requirements.txt
```

For CUDA 12.1:
```bash
conda create -n llm-sim-single-chip python=3.10
conda activate llm-sim-single-chip
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers huggingface_hub -c huggingface
pip install -r requirements.txt
```

For CPU-only:
```bash
conda create -n llm-sim-single-chip python=3.10
conda activate llm-sim-single-chip
conda install pytorch cpuonly -c pytorch
conda install transformers huggingface_hub -c huggingface
pip install -r requirements.txt
```

## Environment Management

### List Environments
```bash
conda env list
# or
conda info --envs
```

### Activate/Deactivate Environment
```bash
# Activate
conda activate llm-sim-single-chip

# Deactivate
conda deactivate
```

### Update Environment
```bash
# Update from environment.yml
conda env update -f environment.yml

# Update specific packages
conda update pytorch transformers
```

### Remove Environment
```bash
conda env remove -n llm-sim-single-chip
```

## Verification

### Test Installation
```bash
# Activate environment
conda activate llm-sim-single-chip

# Test Python imports
python -c "import torch; import transformers; import numpy; print('All packages imported successfully!')"

# Test GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
python -m pytest tests/ -v

# Run example
python examples/qwen3_analysis.py
```

### Check Package Versions
```bash
conda list
# or for specific packages
conda list pytorch transformers numpy
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install matching PyTorch version
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **Channel Conflicts**
   ```bash
   # Clear conda cache
   conda clean --all
   
   # Reset channels
   conda config --remove-key channels
   conda config --add channels conda-forge
   ```

3. **Environment Creation Fails**
   ```bash
   # Try with mamba (faster solver)
   conda install mamba -c conda-forge
   mamba env create -f environment.yml
   ```

4. **Package Not Found**
   ```bash
   # Search for packages
   conda search package-name
   
   # Install from specific channel
   conda install package-name -c channel-name
   ```

### Performance Optimization

1. **Use Mamba for Faster Solves**
   ```bash
   conda install mamba -c conda-forge
   mamba env create -f environment.yml
   ```

2. **Enable Conda Parallel Downloads**
   ```bash
   conda config --set download_threads 4
   ```

3. **Use Local Cache**
   ```bash
   conda config --set use_only_tar_bz2 true
   ```

## Development Workflow

### Daily Usage
```bash
# Activate environment
conda activate llm-sim-single-chip

# Work on your code
# ...

# Deactivate when done
conda deactivate
```

### Adding New Dependencies
```bash
# Add to environment.yml
# Then update environment
conda env update -f environment.yml

# Or install directly
conda install new-package
# or
pip install new-package
```

### Sharing Environment
```bash
# Export exact environment
conda env export > environment-exact.yml

# Export with build strings
conda env export --no-builds > environment-no-builds.yml
```

## Integration with IDEs

### VS Code
1. Install Python extension
2. Select the conda environment as interpreter
3. Use `Ctrl+Shift+P` → "Python: Select Interpreter"

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add Interpreter → Conda Environment
3. Select existing environment or create new one

### Jupyter Notebooks
```bash
# Install kernel
conda activate llm-sim-single-chip
python -m ipykernel install --user --name llm-sim-single-chip --display-name "LLM Sim Single Chip"
```

## Best Practices

1. **Always use environment.yml** for reproducible builds
2. **Pin major versions** in environment.yml for stability
3. **Use separate environments** for different projects
4. **Regularly update** packages for security and features
5. **Document any manual installations** in the README
6. **Test environment setup** on clean systems regularly



