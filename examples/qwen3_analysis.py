#!/usr/bin/env python3
"""Example script for analyzing Qwen3 models with the LLM simulation framework."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_sim import ModelConfigLoader, HardwareConfig, HardwarePresets


def analyze_qwen3_models():
    """Analyze different Qwen3 model configurations."""
    
    print("=== Qwen3 Model Analysis ===\n")
    
    # Initialize components
    config_loader = ModelConfigLoader()
    
    # Models to analyze
    models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B", 
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
    ]
    
    # Hardware configurations to test
    hardware_configs = {
        "Default": HardwareConfig(),
        "Mobile": HardwarePresets.mobile_config(),
        "Datacenter": HardwarePresets.datacenter_config(),
    }
    
    # Analyze each model
    for model_name in models:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing: {model_name}")
            print('='*60)
            
            # Load model configuration
            config_loader.print_model_info(model_name)
            
            # Analyze with different hardware configurations
            for hw_name, hw_config in hardware_configs.items():
                print(f"\n--- Hardware: {hw_name} ---")
                hw_config.print_summary()
                
                # Here you would run the actual MMA analysis
                # For now, just show the configuration
                print(f"Ready to analyze {model_name} with {hw_name} hardware")
                
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print('='*60)


def compare_model_scaling():
    """Compare how different model sizes scale."""
    
    print("\n=== Model Scaling Analysis ===\n")
    
    config_loader = ModelConfigLoader()
    hw_config = HardwareConfig()
    
    models = [
        ("Qwen/Qwen3-0.6B", "0.6B"),
        ("Qwen/Qwen3-1.7B", "1.7B"),
        ("Qwen/Qwen3-4B", "4B"),
        ("Qwen/Qwen3-8B", "8B"),
    ]
    
    print(f"{'Model':<15} {'Params':<10} {'Hidden':<8} {'Layers':<8} {'Heads':<8} {'Est. Size':<12}")
    print("-" * 80)
    
    for model_name, size_name in models:
        try:
            info = config_loader.get_model_info(model_name)
            print(f"{size_name:<15} {info.get('estimated_params', 0):<10,} "
                  f"{info['hidden_size']:<8} {info['num_hidden_layers']:<8} "
                  f"{info['num_attention_heads']:<8} {info.get('estimated_params', 0)/1e6:<12.1f}M")
        except Exception as e:
            print(f"{size_name:<15} Error: {e}")


def analyze_moe_vs_dense():
    """Compare MoE vs Dense architectures."""
    
    print("\n=== MoE vs Dense Architecture Comparison ===\n")
    
    config_loader = ModelConfigLoader()
    
    # Dense models
    dense_models = [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B", 
        "Qwen/Qwen3-8B",
    ]
    
    # MoE models (if available)
    moe_models = [
        "Qwen/Qwen3-30B-A3B",  # MoE with 3B active parameters
        "Qwen/Qwen3-235B-A22B",  # MoE with 22B active parameters
    ]
    
    print("Dense Models:")
    print(f"{'Model':<20} {'Total Params':<15} {'Hidden Size':<12} {'Layers':<8}")
    print("-" * 60)
    
    for model_name in dense_models:
        try:
            info = config_loader.get_model_info(model_name)
            print(f"{model_name.split('/')[-1]:<20} {info.get('estimated_params', 0):<15,} "
                  f"{info['hidden_size']:<12} {info['num_hidden_layers']:<8}")
        except Exception as e:
            print(f"{model_name:<20} Error: {e}")
    
    print("\nMoE Models:")
    print(f"{'Model':<20} {'Total Params':<15} {'Active Params':<15} {'Hidden Size':<12}")
    print("-" * 70)
    
    for model_name in moe_models:
        try:
            info = config_loader.get_model_info(model_name)
            # For MoE models, we might need to extract active parameters differently
            print(f"{model_name.split('/')[-1]:<20} {info.get('estimated_params', 0):<15,} "
                  f"{'N/A':<15} {info['hidden_size']:<12}")
        except Exception as e:
            print(f"{model_name:<20} Error: {e}")


def main():
    """Main function to run all analyses."""
    
    print("LLM Single Chip Simulation - Qwen3 Analysis")
    print("=" * 50)
    
    try:
        # Run different analyses
        analyze_qwen3_models()
        compare_model_scaling()
        analyze_moe_vs_dense()
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



