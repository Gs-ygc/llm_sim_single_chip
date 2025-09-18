#!/bin/bash
# Setup script for LLM Single Chip Simulation environment

set -e  # Exit on any error

echo "> Setting up LLM Single Chip Simulation environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "x Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if mamba is available (faster solver)
if command -v mamba &> /dev/null; then
    echo "- Mamba found - using for faster environment creation"
    SOLVER="mamba"
else
    echo "!  Mamba not found - using conda (slower)"
    echo "   Install mamba for faster solves: conda install mamba -c conda-forge"
    SOLVER="conda"
fi

# Detect platform and choose appropriate environment file
PLATFORM=$(uname -s)
ARCH=$(uname -m)

if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "🍎 Detected macOS ARM64 - using CPU-only environment"
    ENV_FILE="environment-cpu.yml"
elif [[ "$PLATFORM" == "Linux" ]]; then
    echo "🐧 Detected Linux - checking for CUDA support..."
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected - using GPU environment"
        ENV_FILE="environment-gpu.yml"
    else
        echo "💻 No NVIDIA GPU detected - using CPU-only environment"
        ENV_FILE="environment-cpu.yml"
    fi
else
    echo "💻 Using default CPU-only environment"
    ENV_FILE="environment-cpu.yml"
fi

# Create environment
echo "📦 Creating conda environment from $ENV_FILE..."
if [ "$SOLVER" = "mamba" ]; then
    mamba env create -f $ENV_FILE
else
    conda env create -f $ENV_FILE
fi

# Activate environment
echo "> Activating environment..."
eval "$(conda shell.bash hook)"

# Determine environment name based on the file used
if [[ "$ENV_FILE" == "environment-gpu.yml" ]]; then
    ENV_NAME="llm-sim-single-chip-gpu"
else
    ENV_NAME="llm-sim-single-chip-cpu"
fi

conda activate $ENV_NAME

# Install package in development mode
echo "# Installing package in development mode..."
pip install -e .

# Verify installation
echo "- Verifying installation..."
python -c "
import torch
import transformers
import numpy
import pandas
import matplotlib
print('- All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# Run basic tests
echo "> Running basic tests..."
python -m pytest tests/test_config_loader.py -v

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the example analysis:"
echo "  python examples/qwen3_analysis.py"
echo ""
echo "To run the original tiling analysis:"
echo "  python search_mma_tiling.py"
echo ""
echo "Environment details:"
echo "  Platform: $PLATFORM $ARCH"
echo "  Environment file: $ENV_FILE"
echo "  Environment name: $ENV_NAME"
