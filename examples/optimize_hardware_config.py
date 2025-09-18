#!/usr/bin/env python3
"""Hardware configuration optimization and visualization."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from llm_sim import HardwareConfig, InferenceConfig, MMAAnalyzer

def create_hardware_config(sram_size_mb, dram_bandwidth, sram_bandwidth):
    """Create a hardware configuration with given parameters."""
    return HardwareConfig(
        sram_size=sram_size_mb * 1024 * 1024,  # Convert MB to bytes
        dram_bandwidth=dram_bandwidth,
        sram_bandwidth=sram_bandwidth,
        mult_per_cycle=32 * 8 * 32,  # Keep compute constant
        bytes_per_element=1,  # FP8
        bytes_per_acc=2,      # BF16
        tile_quants=(64, 256, 64),
        freq=2 * 1000 * 1000 * 1000,  # 2GHz
    )

def evaluate_configuration(sram_size_mb, dram_bandwidth, sram_bandwidth, model_name="Qwen/Qwen3-30B-A3B"):
    """Evaluate a hardware configuration and return performance metrics."""
    try:
        hw_config = create_hardware_config(sram_size_mb, dram_bandwidth, sram_bandwidth)
        inf_config = InferenceConfig()
        analyzer = MMAAnalyzer(hw_config, inf_config)
        
        results = analyzer.analyze_model(model_name)
        
        return {
            'sram_size_mb': sram_size_mb,
            'dram_bandwidth': dram_bandwidth,
            'sram_bandwidth': sram_bandwidth,
            'tokens_per_second': results['tokens_per_second'],
            'per_layer_time_ms': results['per_layer_time_ms'],
            'per_forward_time_ms': results['per_forward_time_ms'],
            'total_cycles': results['total_cycles'],
            'success': True
        }
    except Exception as e:
        return {
            'sram_size_mb': sram_size_mb,
            'dram_bandwidth': dram_bandwidth,
            'sram_bandwidth': sram_bandwidth,
            'tokens_per_second': 0,
            'per_layer_time_ms': float('inf'),
            'per_forward_time_ms': float('inf'),
            'total_cycles': float('inf'),
            'success': False,
            'error': str(e)
        }

def optimize_hardware_configs():
    """Search for optimal hardware configurations."""
    print("> Searching for optimal hardware configurations...")
    
    # Define parameter ranges
    sram_sizes = [0.5, 1.0, 2.0, 4.0, 8.0]  # MB
    dram_bandwidths = [8, 16, 32, 64, 128]  # bytes/cycle
    sram_bandwidths = [16, 32, 64, 128, 256]  # bytes/cycle
    
    results = []
    total_combinations = len(sram_sizes) * len(dram_bandwidths) * len(sram_bandwidths)
    current = 0
    
    print(f"Total combinations to test: {total_combinations}")
    
    for sram_size, dram_bw, sram_bw in product(sram_sizes, dram_bandwidths, sram_bandwidths):
        current += 1
        print(f"Testing {current}/{total_combinations}: SRAM={sram_size}MB, DRAM={dram_bw}, SRAM_BW={sram_bw}")
        
        result = evaluate_configuration(sram_size, dram_bw, sram_bw)
        results.append(result)
        
        if result['success']:
            print(f"  - Tokens/s: {result['tokens_per_second']:.1f}")
        else:
            print(f"  x Failed: {result.get('error', 'Unknown error')}")
    
    return results

def create_visualizations(results_df):
    """Create visualization plots for the optimization results."""
    # Filter successful results
    success_df = results_df[results_df['success'] == True].copy()
    
    if len(success_df) == 0:
        print("x No successful configurations found!")
        return
    
    print(f"- Creating visualizations for {len(success_df)} successful configurations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hardware Configuration Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. SRAM Size vs Performance
    ax1 = axes[0, 0]
    sram_performance = success_df.groupby('sram_size_mb')['tokens_per_second'].agg(['mean', 'max', 'min']).reset_index()
    ax1.plot(sram_performance['sram_size_mb'], sram_performance['mean'], 'o-', label='Average', linewidth=2, markersize=8)
    ax1.plot(sram_performance['sram_size_mb'], sram_performance['max'], 's-', label='Maximum', linewidth=2, markersize=8)
    ax1.fill_between(sram_performance['sram_size_mb'], sram_performance['min'], sram_performance['max'], alpha=0.3)
    ax1.set_xlabel('SRAM Size (MB)')
    ax1.set_ylabel('Tokens per Second')
    ax1.set_title('SRAM Size vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. DRAM Bandwidth vs Performance
    ax2 = axes[0, 1]
    dram_performance = success_df.groupby('dram_bandwidth')['tokens_per_second'].agg(['mean', 'max', 'min']).reset_index()
    ax2.plot(dram_performance['dram_bandwidth'], dram_performance['mean'], 'o-', label='Average', linewidth=2, markersize=8)
    ax2.plot(dram_performance['dram_bandwidth'], dram_performance['max'], 's-', label='Maximum', linewidth=2, markersize=8)
    ax2.fill_between(dram_performance['dram_bandwidth'], dram_performance['min'], dram_performance['max'], alpha=0.3)
    ax2.set_xlabel('DRAM Bandwidth (bytes/cycle)')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('DRAM Bandwidth vs Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SRAM Bandwidth vs Performance
    ax3 = axes[1, 0]
    sram_bw_performance = success_df.groupby('sram_bandwidth')['tokens_per_second'].agg(['mean', 'max', 'min']).reset_index()
    ax3.plot(sram_bw_performance['sram_bandwidth'], sram_bw_performance['mean'], 'o-', label='Average', linewidth=2, markersize=8)
    ax3.plot(sram_bw_performance['sram_bandwidth'], sram_bw_performance['max'], 's-', label='Maximum', linewidth=2, markersize=8)
    ax3.fill_between(sram_bw_performance['sram_bandwidth'], sram_bw_performance['min'], sram_bw_performance['max'], alpha=0.3)
    ax3.set_xlabel('SRAM Bandwidth (bytes/cycle)')
    ax3.set_ylabel('Tokens per Second')
    ax3.set_title('SRAM Bandwidth vs Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D Scatter plot of all parameters
    ax4 = axes[1, 1]
    scatter = ax4.scatter(success_df['sram_size_mb'], success_df['dram_bandwidth'], 
                         c=success_df['tokens_per_second'], s=success_df['sram_bandwidth']*2, 
                         alpha=0.7, cmap='viridis')
    ax4.set_xlabel('SRAM Size (MB)')
    ax4.set_ylabel('DRAM Bandwidth (bytes/cycle)')
    ax4.set_title('3D Parameter Space (Color=Tokens/s, Size=SRAM_BW)')
    plt.colorbar(scatter, ax=ax4, label='Tokens per Second')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hardware_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed plots
    create_detailed_plots(success_df)

def create_detailed_plots(success_df):
    """Create additional detailed analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance distribution
    ax1 = axes[0, 0]
    ax1.hist(success_df['tokens_per_second'], bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(success_df['tokens_per_second'].mean(), color='red', linestyle='--', label=f'Mean: {success_df["tokens_per_second"].mean():.1f}')
    ax1.axvline(success_df['tokens_per_second'].max(), color='green', linestyle='--', label=f'Max: {success_df["tokens_per_second"].max():.1f}')
    ax1.set_xlabel('Tokens per Second')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Performance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Latency vs Performance
    ax2 = axes[0, 1]
    ax2.scatter(success_df['per_forward_time_ms'], success_df['tokens_per_second'], alpha=0.7)
    ax2.set_xlabel('Forward Time (ms)')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Latency vs Performance Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # 3. Bandwidth ratio analysis
    ax3 = axes[1, 0]
    success_df['bandwidth_ratio'] = success_df['sram_bandwidth'] / success_df['dram_bandwidth']
    ax3.scatter(success_df['bandwidth_ratio'], success_df['tokens_per_second'], alpha=0.7)
    ax3.set_xlabel('SRAM/DRAM Bandwidth Ratio')
    ax3.set_ylabel('Tokens per Second')
    ax3.set_title('Bandwidth Ratio vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top 10 configurations
    ax4 = axes[1, 1]
    top_10 = success_df.nlargest(10, 'tokens_per_second')
    config_labels = [f"SRAM:{row['sram_size_mb']}MB\nDRAM:{row['dram_bandwidth']}\nSRAM_BW:{row['sram_bandwidth']}" 
                    for _, row in top_10.iterrows()]
    bars = ax4.bar(range(len(top_10)), top_10['tokens_per_second'])
    ax4.set_xlabel('Configuration Rank')
    ax4.set_ylabel('Tokens per Second')
    ax4.set_title('Top 10 Configurations')
    ax4.set_xticks(range(len(top_10)))
    ax4.set_xticklabels([f"#{i+1}" for i in range(len(top_10))])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_optimization_summary(results_df):
    """Print a summary of the optimization results."""
    success_df = results_df[results_df['success'] == True]
    
    if len(success_df) == 0:
        print("x No successful configurations found!")
        return
    
    print("\n" + "="*80)
    print("# HARDWARE OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"- Total configurations tested: {len(results_df)}")
    print(f"- Successful configurations: {len(success_df)}")
    print(f"x Failed configurations: {len(results_df) - len(success_df)}")
    print(f"- Success rate: {len(success_df)/len(results_df)*100:.1f}%")
    
    # Performance statistics
    print(f"\n> Performance Statistics:")
    print(f"   Average Tokens/s: {success_df['tokens_per_second'].mean():.1f}")
    print(f"   Maximum Tokens/s: {success_df['tokens_per_second'].max():.1f}")
    print(f"   Minimum Tokens/s: {success_df['tokens_per_second'].min():.1f}")
    print(f"   Standard deviation: {success_df['tokens_per_second'].std():.1f}")
    
    # Top 5 configurations
    print(f"\n# Top 5 Configurations:")
    top_5 = success_df.nlargest(5, 'tokens_per_second')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   #{i}: {row['tokens_per_second']:.1f} tokens/s")
        print(f"        SRAM: {row['sram_size_mb']}MB, DRAM: {row['dram_bandwidth']}, SRAM_BW: {row['sram_bandwidth']}")
        print(f"        Latency: {row['per_forward_time_ms']:.1f}ms")
    
    # Parameter sensitivity analysis
    print(f"\n- Parameter Sensitivity Analysis:")
    for param in ['sram_size_mb', 'dram_bandwidth', 'sram_bandwidth']:
        correlation = success_df[param].corr(success_df['tokens_per_second'])
        print(f"   {param}: correlation = {correlation:.3f}")

def main():
    """Main optimization function."""
    print("> Starting Hardware Configuration Optimization")
    print("="*60)
    
    # Run optimization
    results = optimize_hardware_configs()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print_optimization_summary(results_df)
    
    # Create visualizations
    create_visualizations(results_df)
    
    # Save results to CSV
    results_df.to_csv('hardware_optimization_results.csv', index=False)
    print(f"\n- Results saved to 'hardware_optimization_results.csv'")
    print(f"- Plots saved as 'hardware_optimization_results.png' and 'detailed_performance_analysis.png'")

if __name__ == "__main__":
    main()
