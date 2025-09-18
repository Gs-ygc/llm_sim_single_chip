@echo off
REM Setup script for LLM Single Chip Simulation environment (Windows)

echo > Setting up LLM Single Chip Simulation environment...

REM Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo x Conda is not installed. Please install Miniconda or Anaconda first.
    echo    Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check if mamba is available (faster solver)
where mamba >nul 2>nul
if %errorlevel% equ 0 (
    echo - Mamba found - using for faster environment creation
    set SOLVER=mamba
) else (
    echo !  Mamba not found - using conda (slower)
    echo    Install mamba for faster solves: conda install mamba -c conda-forge
    set SOLVER=conda
)

REM Create environment
echo 📦 Creating conda environment from environment.yml...
if "%SOLVER%"=="mamba" (
    mamba env create -f environment.yml
) else (
    conda env create -f environment.yml
)

REM Activate environment
echo > Activating environment...
call conda activate llm-sim-single-chip

REM Install package in development mode
echo # Installing package in development mode...
pip install -e .

REM Verify installation
echo - Verifying installation...
python -c "import torch; import transformers; import numpy; import pandas; import matplotlib; print('- All core packages imported successfully!'); print(f'PyTorch version: {torch.__version__}'); print(f'Transformers version: {transformers.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only')"

REM Run basic tests
echo > Running basic tests...
python -m pytest tests/test_config_loader.py -v

echo.
echo 🎉 Environment setup complete!
echo.
echo To activate the environment in the future, run:
echo   conda activate llm-sim-single-chip
echo.
echo To run the example analysis:
echo   python examples/qwen3_analysis.py
echo.
echo To run the original tiling analysis:
echo   python search_mma_tiling.py
echo.
pause



