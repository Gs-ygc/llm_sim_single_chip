#!/usr/bin/env python3
"""Command-line interface for LLM Single Chip Simulation."""

import argparse
import sys
from typing import Optional
from .mma_analyzer import MMAAnalyzer
from .hardware_recommender import HardwareRecommender
from .hardware_config import HardwareConfig, HardwarePresets
from .inference_config import InferenceConfig, InferencePresets


def analyze_model_cmd(args):
    """Analyze a single model."""
    print(f"> Analyzing model: {args.model}")
    
    # Create analyzer
    hw_config = HardwareConfig()
    if args.hardware:
        if args.hardware == "xsai":
            hw_config = HardwarePresets.xsai_config()
        elif args.hardware == "mobile":
            hw_config = HardwarePresets.mobile_config()
        elif args.hardware == "datacenter":
            hw_config = HardwarePresets.datacenter_config()
        elif args.hardware == "research":
            hw_config = HardwarePresets.research_config()
    
    # Create inference config (can be extended to support different inference presets)
    inf_config = InferenceConfig()
    
    # Create analyzer with model source
    model_source = getattr(args, 'source', None)
    analyzer = MMAAnalyzer(hw_config, inf_config, model_source=model_source)
    results = analyzer.analyze_model(args.model, verbose=args.verbose, explain=args.explain)
    analyzer.print_analysis_summary(results, explain=args.explain)


def recommend_hardware_cmd(args):
    """Recommend hardware for a model."""
    print(f"# Recommending hardware for: {args.model}")
    
    model_source = getattr(args, 'source', None)
    recommender = HardwareRecommender(model_source=model_source)
    recommendations = recommender.recommend_hardware(
        model_name=args.model,
        target_tokens_per_second=args.target_tps,
        use_case=args.use_case,
        max_cost=args.max_cost
    )
    
    recommender.print_recommendations(recommendations)


def compare_models_cmd(args):
    """Compare multiple models."""
    print(f"> Comparing models: {', '.join(args.models)}")
    
    model_source = getattr(args, 'source', None)
    recommender = HardwareRecommender(model_source=model_source)
    comparison_results = []
    
    for model_name in args.models:
        print(f"\n- Analyzing {model_name}...")
        recommendations = recommender.recommend_hardware(
            model_name=model_name,
            use_case=args.use_case
        )
        
        best_rec = recommendations[0]
        analyzer = MMAAnalyzer(best_rec.hardware_config, model_source=model_source)
        results = analyzer.analyze_model(model_name)
        
        comparison_results.append({
            "model": model_name,
            "recommended_hardware": best_rec.config_name,
            "tokens_per_second": results["tokens_per_second"],
            "per_layer_time_ms": results["per_layer_time_ms"],
            "efficiency_score": best_rec.efficiency_score,
            "cost_score": best_rec.cost_score
        })
    
    # Print comparison table
    print(f"\n- Comparison Results:")
    print("-" * 100)
    print(f"{'Model':<20} {'Hardware':<15} {'Tokens/s':<10} {'Layer Time':<12} {'Efficiency':<12} {'Cost Score':<12}")
    print("-" * 100)
    
    for result in comparison_results:
        print(f"{result['model'].split('/')[-1]:<20} "
              f"{result['recommended_hardware']:<15} "
              f"{result['tokens_per_second']:<10.2f} "
              f"{result['per_layer_time_ms']:<12.2f} "
              f"{result['efficiency_score']:<12.2f} "
              f"{result['cost_score']:<12.2f}")
    
    # Find best model for each metric
    best_performance = max(comparison_results, key=lambda x: x['tokens_per_second'])
    best_efficiency = max(comparison_results, key=lambda x: x['efficiency_score'])
    best_cost = max(comparison_results, key=lambda x: x['cost_score'])
    
    print(f"\n# Best Models:")
    print(f"   > Best Performance: {best_performance['model']} ({best_performance['tokens_per_second']:.2f} tokens/s)")
    print(f"   - Best Efficiency: {best_efficiency['model']} ({best_efficiency['efficiency_score']:.2f})")
    print(f"   - Best Cost-Effectiveness: {best_cost['model']} ({best_cost['cost_score']:.2f})")


def compare_hardware_cmd(args):
    """Compare different hardware configurations for a model."""
    print(f"> Comparing hardware configurations for model: {args.model}")
    
    # Get hardware configurations
    hardware_configs = {}
    if args.hardware:
        for hw_name in args.hardware:
            if hw_name == "xsai":
                hardware_configs[hw_name] = HardwarePresets.xsai_config()
            elif hw_name == "mobile":
                hardware_configs[hw_name] = HardwarePresets.mobile_config()
            elif hw_name == "datacenter":
                hardware_configs[hw_name] = HardwarePresets.datacenter_config()
            elif hw_name == "research":
                hardware_configs[hw_name] = HardwarePresets.research_config()
    
    if not hardware_configs:
        # Default to comparing mobile and xsai
        hardware_configs = {
            "mobile": HardwarePresets.mobile_config(),
            "xsai": HardwarePresets.xsai_config()
        }
    
    # Set baseline
    baseline = args.baseline if args.baseline else list(hardware_configs.keys())[0]
    
    # Create analyzer and run comparison
    model_source = getattr(args, 'source', None)
    analyzer = MMAAnalyzer(model_source=model_source)
    comparison = analyzer.compare_hardware_configs(
        model_name=args.model,
        hardware_configs=hardware_configs,
        baseline_hardware=baseline,
        verbose=args.verbose
    )
    
    analyzer.print_hardware_comparison(comparison)
    analyzer.print_comparison_summary(comparison, "hardware")


def compare_combinations_cmd(args):
    """Compare different model-hardware combinations."""
    print(f"> Comparing model-hardware combinations")
    
    # Parse combinations from arguments
    combinations = []
    for combo_str in args.combinations:
        parts = combo_str.split(':')
        if len(parts) != 2:
            print(f"x Invalid combination format: {combo_str}. Use 'model:hardware'")
            return
        
        model_name, hw_name = parts
        
        # Get hardware config
        if hw_name == "xsai":
            hw_config = HardwarePresets.xsai_config()
        elif hw_name == "mobile":
            hw_config = HardwarePresets.mobile_config()
        elif hw_name == "datacenter":
            hw_config = HardwarePresets.datacenter_config()
        elif hw_name == "research":
            hw_config = HardwarePresets.research_config()
        else:
            print(f"x Unknown hardware configuration: {hw_name}")
            return
        
        combinations.append((model_name, hw_name, hw_config))
    
    if not combinations:
        print("x No valid combinations provided")
        return
    
    # Set baseline
    baseline = None
    if args.baseline:
        baseline_parts = args.baseline.split(':')
        if len(baseline_parts) == 2:
            baseline = (baseline_parts[0], baseline_parts[1])
    
    # Create analyzer and run comparison
    model_source = getattr(args, 'source', None)
    analyzer = MMAAnalyzer(model_source=model_source)
    comparison = analyzer.compare_models_and_hardware(
        model_hardware_pairs=combinations,
        baseline_pair=baseline,
        verbose=args.verbose
    )
    
    analyzer.print_model_hardware_comparison(comparison)
    analyzer.print_comparison_summary(comparison, "model_hardware")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Single Chip Simulation - Hardware Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a model with default hardware (ModelScope - default)
  llm-sim analyze qwen/Qwen3-1.7B
  
  # Analyze with specific hardware
  llm-sim analyze qwen/Qwen3-1.7B --hardware mobile
  
  # Analyze a model from Hugging Face
  llm-sim --source huggingface analyze Qwen/Qwen3-1.7B
  
  # Get hardware recommendations
  llm-sim recommend qwen/Qwen3-1.7B --target-tps 10 --use-case mobile
  
  # Compare multiple models
  llm-sim compare qwen/Qwen3-0.6B qwen/Qwen3-1.7B qwen/Qwen3-4B
  
  # Compare hardware configurations for a model
  llm-sim compare-hardware qwen/Qwen3-1.7B --hardware mobile xsai --baseline mobile
  
  # Compare model-hardware combinations
  llm-sim compare-combinations qwen/Qwen3-1.7B:mobile qwen/Qwen3-1.7B:xsai qwen/Qwen3-4B:datacenter

Environment Variables:
  LLM_SIM_MODEL_SOURCE  Override default model source (default: modelscope)
        """
    )
    
    # Global arguments
    parser.add_argument('--source', choices=['huggingface', 'modelscope'],
                       help='Model source (default: modelscope or from LLM_SIM_MODEL_SOURCE env)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a model')
    analyze_parser.add_argument('model', help='Model name (e.g., qwen/Qwen3-1.7B for ModelScope)')
    analyze_parser.add_argument('--hardware', choices=['xsai', 'mobile', 'datacenter', 'research'], 
                               help='Hardware configuration to use')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', 
                               help='Show detailed analysis')
    analyze_parser.add_argument('--explain', action='store_true', 
                               help='Show detailed diagnostics for performance differences')
    analyze_parser.set_defaults(func=analyze_model_cmd)
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend hardware for a model')
    recommend_parser.add_argument('model', help='Model name (e.g., Qwen/Qwen3-1.7B)')
    recommend_parser.add_argument('--target-tps', type=float, 
                                help='Target tokens per second')
    recommend_parser.add_argument('--use-case', 
                                 choices=['mobile', 'edge', 'datacenter', 'research', 'general'],
                                 default='general', help='Use case')
    recommend_parser.add_argument('--max-cost', 
                                 choices=['low', 'medium', 'high'],
                                 help='Maximum cost constraint')
    recommend_parser.set_defaults(func=recommend_hardware_cmd)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('models', nargs='+', 
                               help='Model names to compare')
    compare_parser.add_argument('--use-case', 
                               choices=['mobile', 'edge', 'datacenter', 'research', 'general'],
                               default='general', help='Use case')
    compare_parser.set_defaults(func=compare_models_cmd)
    
    # Compare hardware command
    compare_hw_parser = subparsers.add_parser('compare-hardware', help='Compare hardware configurations for a model')
    compare_hw_parser.add_argument('model', help='Model name (e.g., Qwen/Qwen3-1.7B)')
    compare_hw_parser.add_argument('--hardware', nargs='+', 
                                  choices=['xsai', 'mobile', 'datacenter', 'research'],
                                  help='Hardware configurations to compare (default: mobile, xsai)')
    compare_hw_parser.add_argument('--baseline', 
                                  choices=['xsai', 'mobile', 'datacenter', 'research'],
                                  help='Baseline hardware for comparison (default: first hardware)')
    compare_hw_parser.add_argument('--verbose', '-v', action='store_true', 
                                  help='Show detailed analysis')
    compare_hw_parser.set_defaults(func=compare_hardware_cmd)
    
    # Compare combinations command
    compare_combo_parser = subparsers.add_parser('compare-combinations', help='Compare model-hardware combinations')
    compare_combo_parser.add_argument('combinations', nargs='+', 
                                     help='Model:hardware combinations (e.g., Qwen/Qwen3-1.7B:mobile)')
    compare_combo_parser.add_argument('--baseline', 
                                     help='Baseline combination in format model:hardware')
    compare_combo_parser.add_argument('--verbose', '-v', action='store_true', 
                                     help='Show detailed analysis')
    compare_combo_parser.set_defaults(func=compare_combinations_cmd)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nx Analysis interrupted by user.")
    except Exception as e:
        print(f"\nx Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
