#!/usr/bin/env python3
"""Example demonstrating ModelScope integration for LLM analysis.

This example shows how to use the llm_sim framework with models from
ModelScope (魔塔社区) instead of Hugging Face.
"""

import os
from llm_sim import ModelConfigLoader, MMAAnalyzer, HardwareConfig, HardwarePresets
from llm_sim.model_source import ModelSource


def example_basic_modelscope():
    """Basic example using ModelScope."""
    print("=" * 80)
    print("Example 1: Loading Model from ModelScope")
    print("=" * 80)
    
    # Method 1: Specify source when creating loader
    config_loader = ModelConfigLoader(model_source="modelscope")
    
    # Load a model from ModelScope (note: model names may differ from HuggingFace)
    # Example: "qwen/Qwen3-1.7B" on ModelScope
    try:
        model_config = config_loader.load_config("qwen/Qwen3-1.7B")
        config_loader.print_model_info("qwen/Qwen3-1.7B")
    except Exception as e:
        print(f"Note: ModelScope integration requires 'modelscope' package: {e}")
        print("Install with: pip install modelscope")


def example_analyze_with_modelscope():
    """Analyze model performance using ModelScope."""
    print("\n" + "=" * 80)
    print("Example 2: Analyzing Model from ModelScope")
    print("=" * 80)
    
    # Create hardware configuration
    hw_config = HardwarePresets.mobile_config()
    
    # Create analyzer with ModelScope source
    try:
        analyzer = MMAAnalyzer(hw_config, model_source="modelscope")
        
        # Analyze a model from ModelScope
        results = analyzer.analyze_model("qwen/Qwen3-1.7B")
        analyzer.print_analysis_summary(results)
    except Exception as e:
        print(f"Note: ModelScope integration requires 'modelscope' package: {e}")
        print("Install with: pip install modelscope")


def example_switch_sources():
    """Example showing how to switch between sources."""
    print("\n" + "=" * 80)
    print("Example 3: Switching Between Model Sources")
    print("=" * 80)
    
    config_loader = ModelConfigLoader(model_source="huggingface")
    
    # Load from Hugging Face
    print("\n> Loading from Hugging Face:")
    try:
        config_loader.print_model_info("Qwen/Qwen3-1.7B")
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
    
    # Switch to ModelScope
    print("\n> Switching to ModelScope:")
    try:
        config_loader.switch_source("modelscope")
        config_loader.print_model_info("qwen/Qwen3-1.7B")
    except Exception as e:
        print(f"Note: ModelScope integration requires 'modelscope' package: {e}")
        print("Install with: pip install modelscope")


def example_environment_variable():
    """Example using environment variable to set default source."""
    print("\n" + "=" * 80)
    print("Example 4: Using Environment Variable")
    print("=" * 80)
    
    # Set environment variable (would normally be set in shell)
    os.environ["LLM_SIM_MODEL_SOURCE"] = "modelscope"
    
    # Create loader without specifying source (uses environment variable)
    config_loader = ModelConfigLoader()
    print(f"Current source: {config_loader.current_source}")
    
    # Clean up
    del os.environ["LLM_SIM_MODEL_SOURCE"]


def example_compare_sources():
    """Compare the same model from different sources."""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Model Sources")
    print("=" * 80)
    
    hw_config = HardwarePresets.xsai_config()
    
    # Analyze from Hugging Face
    print("\n> Analyzing from Hugging Face:")
    try:
        analyzer_hf = MMAAnalyzer(hw_config, model_source="huggingface")
        results_hf = analyzer_hf.analyze_model("Qwen/Qwen3-1.7B")
        print(f"Tokens per second: {results_hf['tokens_per_second']:.2f}")
        print(f"MFU: {results_hf.get('average_mfu', 0):.3f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Analyze from ModelScope
    print("\n> Analyzing from ModelScope:")
    try:
        analyzer_ms = MMAAnalyzer(hw_config, model_source="modelscope")
        results_ms = analyzer_ms.analyze_model("qwen/Qwen3-1.7B")
        print(f"Tokens per second: {results_ms['tokens_per_second']:.2f}")
        print(f"MFU: {results_ms.get('average_mfu', 0):.3f}")
    except Exception as e:
        print(f"Note: ModelScope integration requires 'modelscope' package: {e}")
        print("Install with: pip install modelscope")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ModelScope Integration Examples")
    print("=" * 80)
    print("\nThese examples demonstrate how to use ModelScope (魔塔社区)")
    print("as an alternative to Hugging Face for loading model configurations.")
    print("\nNote: To use ModelScope, you need to install it first:")
    print("  pip install modelscope")
    print("\n")
    
    # Run examples
    example_basic_modelscope()
    example_analyze_with_modelscope()
    example_switch_sources()
    example_environment_variable()
    example_compare_sources()
    
    print("\n" + "=" * 80)
    print("CLI Usage Examples:")
    print("=" * 80)
    print("\n# Analyze a model from ModelScope:")
    print("llm-sim analyze qwen/Qwen3-1.7B --source modelscope")
    print("\n# Compare models from ModelScope:")
    print("llm-sim compare qwen/Qwen3-0.6B qwen/Qwen3-1.7B --source modelscope")
    print("\n# Use environment variable to set default source:")
    print("export LLM_SIM_MODEL_SOURCE=modelscope")
    print("llm-sim analyze qwen/Qwen3-1.7B")
    print("\n")


if __name__ == "__main__":
    main()
