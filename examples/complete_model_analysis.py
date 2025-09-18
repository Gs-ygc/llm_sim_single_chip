#!/usr/bin/env python3
"""Complete model analysis with hardware recommendations."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_sim import (
    ModelConfigLoader, 
    HardwareConfig, 
    HardwarePresets,
    MMAAnalyzer,
    HardwareRecommender
)


def analyze_model_complete(model_name: str, 
                          target_tokens_per_second: float = None,
                          use_case: str = "general",
                          max_cost: str = None,
                          verbose: bool = False):
    """Perform complete analysis of a model with hardware recommendations.
    
    Args:
        model_name: Hugging Face model identifier
        target_tokens_per_second: Target performance (optional)
        use_case: Use case ("mobile", "edge", "datacenter", "research", "general")
        max_cost: Maximum cost ("low", "medium", "high")
        verbose: Whether to show detailed information
    """
    
    print("> LLM Single Chip Simulation - Complete Model Analysis")
    print("=" * 80)
    print(f"- Model: {model_name}")
    print(f"# Use Case: {use_case}")
    if target_tokens_per_second:
        print(f"- Target Performance: {target_tokens_per_second:.1f} tokens/second")
    if max_cost:
        print(f"- Max Cost: {max_cost}")
    print("=" * 80)
    
    # Step 1: Load model configuration
    print("\n- Step 1: Loading Model Configuration")
    print("-" * 50)
    config_loader = ModelConfigLoader()
    config_loader.print_model_info(model_name)
    
    # Step 2: Hardware recommendations
    print("\n> Step 2: Hardware Configuration Analysis")
    print("-" * 50)
    recommender = HardwareRecommender()
    recommendations = recommender.recommend_hardware(
        model_name=model_name,
        target_tokens_per_second=target_tokens_per_second,
        max_cost=max_cost,
        use_case=use_case
    )
    
    # Step 3: Detailed analysis with recommended hardware
    print("\n# Step 3: Detailed Performance Analysis")
    print("-" * 50)
    
    # Analyze with top 3 recommendations
    top_recommendations = recommendations[:3]
    
    for i, rec in enumerate(top_recommendations, 1):
        print(f"\n# Analysis #{i}: {rec.config_name} Configuration")
        print("=" * 60)
        
        analyzer = MMAAnalyzer(rec.hardware_config)
        results = analyzer.analyze_model(model_name, verbose=verbose)
        
        # Print summary
        analyzer.print_analysis_summary(results)
        
        # Additional insights
        print(f"\n> Key Insights:")
        if results["tokens_per_second"] >= 10:
            print("   - Excellent performance - suitable for real-time applications")
        elif results["tokens_per_second"] >= 5:
            print("   - Good performance - suitable for interactive applications")
        elif results["tokens_per_second"] >= 2:
            print("   ! Moderate performance - suitable for batch processing")
        else:
            print("   x Limited performance - consider hardware upgrade")
        
        # Memory analysis
        avg_mfu = sum(op["mfu"] for op in results["operations"]) / len(results["operations"])
        if avg_mfu >= 0.5:
            print("   - High hardware utilization efficiency")
        elif avg_mfu >= 0.3:
            print("   ! Moderate hardware utilization efficiency")
        else:
            print("   x Low hardware utilization efficiency")
    
    # Step 4: Final recommendations
    print("\n# Step 4: Final Recommendations")
    print("-" * 50)
    recommender.print_recommendations(recommendations)
    
    # Step 5: Implementation guidance
    print("\n- Step 5: Implementation Guidance")
    print("-" * 50)
    best_rec = recommendations[0]
    
    print(f"# Recommended Configuration: {best_rec.config_name}")
    print(f"- Reasoning: {best_rec.reasoning}")
    
    print(f"\n# Hardware Specifications:")
    hw_info = best_rec.hardware_config.get_memory_hierarchy_info()
    compute_info = best_rec.hardware_config.get_compute_info()
    print(f"   - SRAM Size: {hw_info['sram_size_mb']:.1f} MB")
    print(f"   - DRAM Bandwidth: {hw_info['dram_bandwidth_gb_per_sec']:.1f} GB/s")
    print(f"   - Peak Compute: {compute_info['peak_compute_ops_per_sec']/1e12:.1f} TOPS")
    print(f"   - Frequency: {compute_info['frequency_ghz']:.1f} GHz")
    
    print(f"\n- Expected Performance:")
    analyzer = MMAAnalyzer(best_rec.hardware_config)
    results = analyzer.analyze_model(model_name)
    print(f"   - Tokens per Second: {results['tokens_per_second']:.2f}")
    print(f"   - Per Layer Time: {results['per_layer_time_ms']:.2f} ms")
    print(f"   - Per Forward Time: {results['per_forward_time_ms']:.2f} ms")
    
    print(f"\n- Next Steps:")
    print(f"   1. Implement the recommended hardware configuration")
    print(f"   2. Use the provided tiling strategies for each operation")
    print(f"   3. Monitor actual performance vs. predicted performance")
    print(f"   4. Optimize based on real-world usage patterns")
    
    return recommendations, results


def compare_models(models: list, use_case: str = "general"):
    """Compare multiple models for a given use case.
    
    Args:
        models: List of model names to compare
        use_case: Use case for comparison
    """
    print(f"\n> Model Comparison for {use_case} Use Case")
    print("=" * 80)
    
    recommender = HardwareRecommender()
    comparison_results = []
    
    for model_name in models:
        print(f"\n- Analyzing {model_name}...")
        recommendations = recommender.recommend_hardware(
            model_name=model_name,
            use_case=use_case
        )
        
        best_rec = recommendations[0]
        analyzer = MMAAnalyzer(best_rec.hardware_config)
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


def main():
    """Main function with example usage."""
    
    # Example 1: Complete analysis of Qwen3-1.7B
    print("Example 1: Complete Analysis of Qwen3-1.7B")
    analyze_model_complete(
        model_name="Qwen/Qwen3-1.7B",
        target_tokens_per_second=5.0,
        use_case="general",
        verbose=False
    )
    
    # Example 2: Mobile deployment analysis
    print("\n" + "="*80)
    print("Example 2: Mobile Deployment Analysis")
    analyze_model_complete(
        model_name="Qwen/Qwen3-0.6B",
        use_case="mobile",
        max_cost="low",
        verbose=False
    )
    
    # Example 3: Model comparison
    print("\n" + "="*80)
    print("Example 3: Model Comparison")
    compare_models(
        models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
        use_case="general"
    )


if __name__ == "__main__":
    main()



