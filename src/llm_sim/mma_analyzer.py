"""Matrix Multiplication Accelerator (MMA) analyzer for LLM models."""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .hardware_config import HardwareConfig
from .inference_config import InferenceConfig
from .config_loader import ModelConfigLoader


@dataclass
class MMAConfig:
    """Configuration for a matrix multiplication operation."""
    
    name: str
    bs: int  # batch size
    m: int   # matrix dimension M
    k: int   # matrix dimension K  
    n: int   # matrix dimension N
    weight_bytes: float
    fa_qk: bool = False  # Flash Attention QK
    fa_pv: bool = False  # Flash Attention PV
    
    def __str__(self):
        return f"{self.name}\t bs={self.bs}\t m={self.m}\t k={self.k}\t n={self.n}"


@dataclass
class TilingResult:
    """Result of tiling optimization."""
    
    tile_m: int
    tile_k: int
    tile_n: int
    total_cycles: float
    mfu: float
    bw_bound_cycles: float
    compute_bound_cycles: float
    still_policy: str
    still_reuse_factor: int
    memory_usage: float
    sram_bw_per_cycle: float


class MMAAnalyzer:
    """Analyzer for Matrix Multiplication Accelerator performance."""
    
    def __init__(self, hardware_config: Optional[HardwareConfig] = None, 
                 inference_config: Optional[InferenceConfig] = None,
                 model_source: Optional[str] = None):
        """Initialize MMA analyzer.
        
        Args:
            hardware_config: Hardware configuration. If None, uses default.
            inference_config: Inference configuration. If None, uses default.
            model_source: Model source ('huggingface' or 'modelscope'). If None, uses default.
        """
        self.hw_config = hardware_config or HardwareConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.config_loader = ModelConfigLoader(model_source=model_source)
        
        # Extract hardware parameters
        self.sram_size = self.hw_config.sram_size
        self.dram_per_cycle = self.hw_config.dram_bandwidth
        self.sram_per_cycle = getattr(self.hw_config, 'sram_bandwidth', None) or (self.hw_config.dram_bandwidth * 8)
        self.mult_per_cycle = self.hw_config.mult_per_cycle
        self.tile_quants = self.hw_config.tile_quants
        self.bytes_per_element = self.hw_config.bytes_per_element
        self.bytes_per_acc = self.hw_config.bytes_per_acc
        self.freq = self.hw_config.freq
        
        # Extract inference parameters
        self.mtp_num = self.inference_config.mtp_num
        self.sd_mal = self.inference_config.sd_mal
    
    def create_model_configs(self, model_name: str) -> List[MMAConfig]:
        """Create MMA configurations for a given model.
        
        Args:
            model_name: Model identifier (from Hugging Face or ModelScope)
            
        Returns:
            List of MMA configurations for the model
        """
        model_info = self.config_loader.get_model_info(model_name)
        
        hidden_dim = model_info["hidden_size"]
        intermediate_size = model_info["intermediate_size"]
        num_q_heads = model_info["num_attention_heads"]
        num_kv_heads = model_info["num_key_value_heads"]
        head_dim = model_info["head_dim"]
        n_layers = model_info["num_hidden_layers"]
        
        # Calculate derived parameters
        group_size = num_q_heads // num_kv_heads
        fa_len_tile = self.inference_config.fa_len_tile
        total_context_len = self.inference_config.total_context_len
        n_fa_chunks = self.inference_config.get_flash_attention_chunks()
        flash_attention_batch_size = self.inference_config.get_flash_attention_batch_size(num_kv_heads)
        
        configs = [
            # Feed-forward network layers
            MMAConfig("up proj", self.inference_config.default_batch_size, self.mtp_num, hidden_dim, intermediate_size, 0.5, 0, 0),
            MMAConfig("down proj", self.inference_config.default_batch_size, self.mtp_num, intermediate_size, hidden_dim, 0.5, 0, 0),
            
            # Attention projection layers
            MMAConfig("q_proj", self.inference_config.default_batch_size, self.mtp_num, hidden_dim, num_q_heads * head_dim, 0.5, 0, 0),
            MMAConfig("kv_proj", self.inference_config.default_batch_size, self.mtp_num, hidden_dim, num_kv_heads * head_dim, 0.5, 0, 0),
            
            # Flash Attention layers
            MMAConfig("fa-qk", flash_attention_batch_size, self.mtp_num * group_size, 
                     max(256, head_dim), fa_len_tile, 1, fa_qk=True, fa_pv=False),
            MMAConfig("fa-pv", flash_attention_batch_size, self.mtp_num * group_size, 
                     fa_len_tile, head_dim, 1, fa_qk=False, fa_pv=True),
            
            # Output projection
            MMAConfig("o_proj", self.inference_config.default_batch_size, self.mtp_num, num_q_heads * head_dim, hidden_dim, 0.5, 0, 0),
        ]
        
        return configs, n_layers
    
    def print_model_matrix_sizes(self, model_name: str):
        """Print matrix sizes for all MMA operations in a given model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Matrix sizes for model: {model_name}")
        print("=" * 80)
        
        # Print model information
        self._print_model_info(model_name)
        
        # Get model configurations
        configs, n_layers = self.create_model_configs(model_name)
        
        print(f"Matrix Multiplication Operations (per layer):")
        print("-" * 80)
        print(f"{'Operation':<15} {'Batch Size':<12} {'M':<8} {'K':<8} {'N':<8} {'Total Elements':<15} {'Memory (MB)':<12}")
        print("-" * 80)
        
        total_elements = 0
        total_memory_mb = 0
        
        for config in configs:
            # Calculate matrix dimensions
            m, k, n = config.m, config.k, config.n
            elements = m * k + k * n  # A matrix + B matrix elements
            memory_mb = elements * self.bytes_per_element / (1024 * 1024)
            
            total_elements += elements
            total_memory_mb += memory_mb
            
            print(f"{config.name:<15} {config.bs:<12} {m:<8} {k:<8} {n:<8} {elements:<15,} {memory_mb:<12.2f}")
        
        print("-" * 80)
        print(f"{'TOTAL (per layer)':<15} {'':<12} {'':<8} {'':<8} {'':<8} {total_elements:<15,} {total_memory_mb:<12.2f}")
        print(f"{'TOTAL (all layers)':<15} {'':<12} {'':<8} {'':<8} {'':<8} {total_elements * n_layers:<15,} {total_memory_mb * n_layers:<12.2f}")
        
        print(f"\nModel Summary:")
        print(f"- Number of layers: {n_layers}")
        print(f"- Operations per layer: {len(configs)}")
        print(f"- Total operations: {len(configs) * n_layers}")
        print(f"- Memory per layer: {total_memory_mb:.2f} MB")
        print(f"- Total model memory: {total_memory_mb * n_layers:.2f} MB")
        print("=" * 80)
    
    def calculate_memory_requirements(self, m: int, k: int, n: int, 
                                    tile_m: int, tile_k: int, tile_n: int) -> Tuple[float, float, float, float]:
        """Calculate memory requirements for tiled matrices.
        
        Args:
            m, k, n: Matrix dimensions
            tile_m, tile_k, tile_n: Tile dimensions
            
        Returns:
            Tuple of (total_memory, a_memory, b_memory, c_memory)
        """
        # Memory for tiled A
        a_memory = tile_m * tile_k * self.bytes_per_element
        
        # Memory for tiled B
        b_memory = tile_k * tile_n * self.bytes_per_element
        
        # Memory for tiled C
        c_memory = tile_m * tile_n * self.bytes_per_acc
        
        total_memory = a_memory + b_memory + c_memory
        return total_memory, a_memory, b_memory, c_memory
    
    def compute_per_tile_cycles(self, config: MMAConfig, tile_m: int, tile_k: int, tile_n: int) -> Tuple[float, float, float, str, int]:
        """Compute cycles per tile for a given configuration.
        
        Args:
            config: MMA configuration
            tile_m, tile_k, tile_n: Tile dimensions
            
        Returns:
            Tuple of (mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor)
        """
        m, k, n = config.m, config.k, config.n
        num_tile_m = np.ceil(m / tile_m)
        num_tile_k = np.ceil(k / tile_k)
        num_tile_n = np.ceil(n / tile_n)
        weight_bytes = config.weight_bytes
        fa_qk = config.fa_qk
        fa_pv = config.fa_pv

        A_bytes = self.bytes_per_element
        B_bytes = weight_bytes
        C_bytes = self.bytes_per_acc

        if fa_qk:
            C_bytes = 0
        elif fa_pv:
            A_bytes = 0

        # C-still policy
        dram_bytes_c = (
            tile_m * tile_k * A_bytes +
            tile_k * tile_n * B_bytes +
            tile_m * tile_n * C_bytes / num_tile_k
        )
        # Estimate SRAM traffic: amount actually moved between SRAM and regs each tile
        # We approximate as A+B+C once per tile (can be refined per policy)
        sram_bytes_c = (
            tile_m * tile_k * A_bytes +
            tile_k * tile_n * B_bytes +
            tile_m * tile_n * C_bytes
        )
        dram_cycles_c = dram_bytes_c / self.dram_per_cycle
        sram_cycles_c = sram_bytes_c / self.sram_per_cycle
        bw_bound_cycles = max(dram_cycles_c, sram_cycles_c)
        still_policy = "c_still"
        still_reuse_factor = num_tile_k
        
        # A-still policy
        a_still_dram_bytes = (
            tile_m * tile_k * A_bytes / num_tile_n +
            tile_k * tile_n * B_bytes +
            tile_m * tile_n * C_bytes
        )
        a_still_sram_bytes = (
            tile_m * tile_k * A_bytes +
            tile_k * tile_n * B_bytes +
            tile_m * tile_n * C_bytes
        )
        a_still_bw_bound_cycles = max(a_still_dram_bytes / self.dram_per_cycle,
                                      a_still_sram_bytes / self.sram_per_cycle)

        if a_still_bw_bound_cycles < bw_bound_cycles:
            bw_bound_cycles = a_still_bw_bound_cycles
            still_policy = "a_still"
            still_reuse_factor = num_tile_n

        # B-still policy
        b_still_dram_bytes = (
            tile_m * tile_k * A_bytes +
            tile_k * tile_n * B_bytes / num_tile_m +
            tile_m * tile_n * C_bytes
        )
        b_still_sram_bytes = (
            tile_m * tile_k * A_bytes +
            tile_k * tile_n * B_bytes +
            tile_m * tile_n * C_bytes
        )
        b_still_bw_bound_cycles = max(b_still_dram_bytes / self.dram_per_cycle,
                                      b_still_sram_bytes / self.sram_per_cycle)

        if b_still_bw_bound_cycles < bw_bound_cycles:
            bw_bound_cycles = b_still_bw_bound_cycles
            still_policy = "b_still"
            still_reuse_factor = num_tile_m

        # Compute cycles
        ideal_compute_cycles = tile_m * tile_k * tile_n / self.mult_per_cycle
        # ideal_compute_cycles = ideal_compute_cycles + tile_m*tile_n/8
        mn_min = min(tile_m, tile_n)
        if mn_min > 64:
            compute_bound_cycles = tile_m * tile_k * tile_n / self.mult_per_cycle
        else:
            discounted_mult_per_cycle = self.mult_per_cycle * mn_min / 64
            compute_bound_cycles = tile_m * tile_k * tile_n / discounted_mult_per_cycle

        max_cycles = max(bw_bound_cycles, compute_bound_cycles)
        mfu = ideal_compute_cycles / max_cycles

        return mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor
    
    def calculate_bandwidth_requirements(self, config: MMAConfig, tile_m: int, tile_k: int, tile_n: int) -> Tuple[float, float, float, float, str, int]:
        """Calculate bandwidth requirements for a configuration.
        
        Args:
            config: MMA configuration
            tile_m, tile_k, tile_n: Tile dimensions
            
        Returns:
            Tuple of (total_cycles, mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor)
        """
        m, k, n = config.m, config.k, config.n
        num_tile_m = (m + tile_m - 1) // tile_m
        num_tile_k = (k + tile_k - 1) // tile_k
        num_tile_n = (n + tile_n - 1) // tile_n
        
        mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor = self.compute_per_tile_cycles(
            config, tile_m, tile_k, tile_n
        )
        total_cycles = max(bw_bound_cycles, compute_bound_cycles) * num_tile_m * num_tile_n * num_tile_k
        
        return total_cycles, mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor
    
    def find_optimal_tiling(self, config: MMAConfig, verbose: bool = False) -> Tuple[Optional[Tuple[int, int, int]], Dict[str, Any]]:
        """Find optimal tiling for a given configuration.
        
        Args:
            config: MMA configuration
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (best_tiling, best_info)
        """
        best_tiling = None
        best_total_cycles = float('inf')
        best_info = {}
        
        # Try different tile sizes
        for tile_m in np.arange(self.tile_quants[0], config.m + self.tile_quants[0], self.tile_quants[0]):
            for tile_k in np.arange(self.tile_quants[1], config.k + self.tile_quants[1], self.tile_quants[1]):
                for tile_n in np.arange(self.tile_quants[2], config.n + self.tile_quants[2], self.tile_quants[2]):
                    # Skip if tile sizes are larger than matrix dimensions
                    if tile_m > config.m or tile_k > config.k or tile_n > config.n:
                        if verbose:
                            print(f"Skipping tile sizes: {tile_m}, {tile_k}, {tile_n} for {config}")
                        continue
                    
                    # Calculate memory requirements
                    total_memory, a_memory, b_memory, c_memory = self.calculate_memory_requirements(
                        config.m, config.k, config.n, tile_m, tile_k, tile_n
                    )
                    
                    # Skip if doesn't fit in SRAM
                    if total_memory > self.sram_size:
                        if verbose:
                            print(f"Skipping tile sizes: {tile_m}, {tile_k}, {tile_n} for {config} because it doesn't fit in SRAM")
                        continue
                    
                    # Calculate bandwidth requirements
                    total_cycles, mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor = self.calculate_bandwidth_requirements(
                        config, tile_m, tile_k, tile_n
                    )
                    total_cycles *= config.bs

                    # Update best tiling if this is better
                    if verbose:
                        print(f"tile_m: {tile_m}, tile_k: {tile_k}, tile_n: {tile_n}, total_cycles: {total_cycles}, still_policy: {still_policy}")
                    if total_cycles < best_total_cycles:
                        if verbose:
                            print(f"update best total cycles: {best_total_cycles} -> {total_cycles}")
                        best_total_cycles = total_cycles
                        best_tiling = (tile_m, tile_k, tile_n)
                        best_info = {
                            "total_cycles": total_cycles,
                            "mfu": mfu,
                            "bw_bound_cycles": bw_bound_cycles,
                            "compute_bound_cycles": compute_bound_cycles,
                            "still_policy": still_policy,
                            "still_reuse_factor": still_reuse_factor,
                        }
                        if verbose:
                            print(best_info)
        
        return best_tiling, best_info
    
    def _print_model_info(self, model_name: str):
        """Print model configuration information."""
        try:
            model_info = self.config_loader.get_model_info(model_name)
            print("- Model Configuration:")
            print(f"  Model Name: {model_name}")
            print(f"  Hidden Size: {model_info['hidden_size']:,}")
            print(f"  Intermediate Size: {model_info['intermediate_size']:,}")
            print(f"  Attention Heads: {model_info['num_attention_heads']}")
            print(f"  Key-Value Heads: {model_info['num_key_value_heads']}")
            print(f"  Head Dimension: {model_info['head_dim']}")
            print(f"  Number of Layers: {model_info['num_hidden_layers']}")
            print(f"  Vocabulary Size: {model_info['vocab_size']:,}")
            print(f"  Max Position Embeddings: {model_info['max_position_embeddings']:,}")
            if 'estimated_params' in model_info:
                print(f"  Estimated Parameters: {model_info['estimated_params']:,}")
            print()
        except Exception as e:
            print(f"!  Could not load model info: {e}")
            print()
    
    def _print_hardware_info(self):
        """Print hardware configuration information."""
        print("# Hardware Configuration:")
        print(f"  SRAM Size: {self.sram_size / (1024*1024):.1f} MB")
        print(f"  DRAM Bandwidth: {self.dram_per_cycle} bytes/cycle")
        print(f"  SRAM Bandwidth: {self.sram_per_cycle} bytes/cycle")
        print(f"  Compute Units: {self.mult_per_cycle:,} multipliers/cycle")
        print(f"  Frequency: {self.freq / (1000**3):.1f} GHz")
        print(f"  Data Types: {self.bytes_per_element*8}-bit input, {self.bytes_per_acc*8}-bit accumulation")
        print(f"  Tile Quantization: {self.tile_quants}")
        print()
        
        print("#  Inference Configuration:")
        print(f"  Context Length: {self.inference_config.total_context_len:,}")
        print(f"  Flash Attention Tile: {self.inference_config.fa_len_tile}")
        print(f"  FA Chunks: {self.inference_config.get_flash_attention_chunks()}")
        print(f"  Default Batch Size: {self.inference_config.default_batch_size}")
        print(f"  Multi-Token Parallelism: {self.mtp_num}")
        print(f"  Speculative Decode MAL: {self.sd_mal}")
        print()
    
    def analyze_model(self, model_name: str, verbose: bool = False, explain: bool = False) -> Dict[str, Any]:
        """Analyze a complete model and return performance metrics.
        
        Args:
            model_name: Hugging Face model identifier
            verbose: Whether to print detailed information
            explain: Whether to show detailed diagnostics
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing model: {model_name}")
        print("-" * 80)
        
        # Print model information
        self._print_model_info(model_name)
        
        # Print hardware information
        self._print_hardware_info()
        
        # Get model configurations
        configs, n_layers = self.create_model_configs(model_name)
        
        results = {
            "model_name": model_name,
            "n_layers": n_layers,
            "operations": [],
            "total_cycles": 0,
            "total_time_ms": 0,
            "tokens_per_second": 0,
        }
        
        total_cycles = 0
        
        for config in configs:
            best_tiling, best_info = self.find_optimal_tiling(config, verbose)
            
            if best_tiling:
                tile_m, tile_k, tile_n = best_tiling
                time_ms = best_info['total_cycles'] / self.freq * 1000
                total_cycles += best_info['total_cycles']
                
                operation_result = {
                    "name": config.name,
                    "dimensions": f"M={config.m}, K={config.k}, N={config.n}",
                    "tiling": f"M={tile_m}, K={tile_k}, N={tile_n}",
                    "time_ms": time_ms,
                    "mfu": best_info['mfu'],
                    "still_policy": best_info['still_policy'],
                    "cycles": best_info['total_cycles'],
                    "bw_bound_cycles": best_info['bw_bound_cycles'],
                    "compute_bound_cycles": best_info['compute_bound_cycles'],
                }
                
                results["operations"].append(operation_result)
                
                if verbose:
                    print(f"{config.name.ljust(20)} {time_ms:.4f} ms, MFU: {best_info['mfu']:.4f}")
                    print(f"  Dimensions: M={config.m}, K={config.k}, N={config.n}")
                    print(f"  Tiling: M={tile_m}, K={tile_k}, N={tile_n}")
                    print(f"  Still policy: {best_info['still_policy']}")
            else:
                print(f"x No valid tiling found for {config.name}")
        
        # Calculate overall metrics
        results["total_cycles"] = total_cycles
        results["total_time_ms"] = total_cycles / self.freq * 1000
        results["per_layer_time_ms"] = results["total_time_ms"] / n_layers
        results["per_forward_time_ms"] = results["total_time_ms"] * n_layers
        results["tokens_per_second"] = 1000 / (results["per_forward_time_ms"]) * self.sd_mal
        
        return results
    
    def print_analysis_summary(self, results: Dict[str, Any], explain: bool = False):
        """Print a summary of the analysis results.
        
        Args:
            results: Analysis results from analyze_model()
            explain: Whether to show detailed diagnostics
        """
        print(f"\n{'='*80}")
        print(f"Analysis Summary for {results['model_name']}")
        print(f"{'='*80}")
        
        print(f"Model Layers: {results['n_layers']}")
        print(f"Total Cycles: {results['total_cycles']:,.0f}")
        print(f"Per Layer Time: {results['per_layer_time_ms']:.4f} ms")
        print(f"Per Forward Time: {results['per_forward_time_ms']:.4f} ms")
        print(f"Tokens per Second: {results['tokens_per_second']:.4f}")
        
        print(f"\nOperation Breakdown:")
        print(f"{'Operation':<20} {'Time (ms)':<12} {'MFU':<8} {'Still Policy':<12}")
        print("-" * 60)
        
        for op in results["operations"]:
            print(f"{op['name']:<20} {op['time_ms']:<12.4f} {op['mfu']:<8.4f} {op['still_policy']:<12}")
        
        print(f"{'='*80}")
        
        # Add detailed diagnostics if explain is requested
        if explain:
            self._print_detailed_diagnostics(results)
    
    def _print_detailed_diagnostics(self, results: Dict[str, Any]):
        """Print detailed diagnostics for the analysis results."""
        print(f"\n> Detailed Diagnostics:")
        print("-" * 80)
        
        # Find the slowest operations
        operations = results["operations"]
        sorted_ops = sorted(operations, key=lambda x: x['time_ms'], reverse=True)
        
        print(f"- Performance Bottlenecks (Top 3):")
        for i, op in enumerate(sorted_ops[:3]):
            bound_type = "bandwidth" if op['bw_bound_cycles'] > op['compute_bound_cycles'] else "compute"
            print(f"  {i+1}. {op['name']}: {op['time_ms']:.4f} ms ({bound_type} bound)")
            print(f"     - Dimensions: {op['dimensions']}")
            print(f"     - Tiling: {op['tiling']}")
            print(f"     - MFU: {op['mfu']:.4f}")
            print(f"     - Still Policy: {op['still_policy']}")
            print(f"     - Bandwidth Cycles: {op['bw_bound_cycles']:.0f}")
            print(f"     - Compute Cycles: {op['compute_bound_cycles']:.0f}")
            print()
        
        # Memory usage analysis
        print(f"- Memory Usage Analysis:")
        max_memory_usage = 0
        memory_usage_details = []
        
        for op in operations:
            # Estimate memory usage based on tiling
            tiling_parts = op['tiling'].split(', ')
            tile_m = int(tiling_parts[0].split('=')[1])
            tile_k = int(tiling_parts[1].split('=')[1])
            tile_n = int(tiling_parts[2].split('=')[1])
            
            # Calculate memory for A, B, C matrices
            a_memory = tile_m * tile_k * self.bytes_per_element
            b_memory = tile_k * tile_n * 0.5  # weight_bytes
            c_memory = tile_m * tile_n * self.bytes_per_acc
            
            total_memory = a_memory + b_memory + c_memory
            max_memory_usage = max(max_memory_usage, total_memory)
            memory_usage_details.append((op['name'], total_memory))
            
            if total_memory > self.sram_size * 0.8:  # If using >80% of SRAM
                print(f"  !  {op['name']}: {total_memory/1024:.1f} KB (high memory usage)")
        
        print(f"  - Peak memory usage: {max_memory_usage/1024:.1f} KB")
        print(f"  # SRAM capacity: {self.sram_size/1024:.1f} KB")
        print(f"  - Memory utilization: {max_memory_usage/self.sram_size*100:.1f}%")
        
        # Show memory usage breakdown
        if len(memory_usage_details) > 0:
            print(f"  - Memory usage by operation:")
            for name, memory in sorted(memory_usage_details, key=lambda x: x[1], reverse=True)[:3]:
                print(f"     - {name}: {memory/1024:.1f} KB")
        print()
        
        # Hardware utilization analysis
        print(f"- Hardware Utilization Analysis:")
        avg_mfu = sum(op['mfu'] for op in operations) / len(operations)
        print(f"  - Average MFU: {avg_mfu:.4f}")
        print(f"  - Peak Compute: {self.mult_per_cycle:,} multipliers/cycle")
        print(f"  - Frequency: {self.freq/(1000**3):.1f} GHz")
        print(f"  - Peak Performance: {self.mult_per_cycle * self.freq / 1e12:.2f} TOPS")
        print()
        
        # Bandwidth analysis
        print(f"- Bandwidth Analysis:")
        print(f"  - DRAM Bandwidth: {self.dram_per_cycle} bytes/cycle ({self.dram_per_cycle * self.freq / (1024**3):.1f} GB/s)")
        print(f"  - SRAM Bandwidth: {self.sram_per_cycle} bytes/cycle ({self.sram_per_cycle * self.freq / (1024**3):.1f} GB/s)")
        print(f"  - Bandwidth Ratio: {self.sram_per_cycle / self.dram_per_cycle:.1f}x")
        print()
        
        # Recommendations
        print(f"> Optimization Recommendations:")
        if avg_mfu < 0.3:
            print(f"  # Low MFU ({avg_mfu:.3f}) - Consider increasing compute units or optimizing tiling")
        if max_memory_usage > self.sram_size * 0.9:
            print(f"  # High memory usage - Consider larger SRAM or more aggressive tiling")
        
        bandwidth_bound_ops = [op for op in operations if op['bw_bound_cycles'] > op['compute_bound_cycles']]
        if len(bandwidth_bound_ops) > len(operations) * 0.6:
            print(f"  # Most operations are bandwidth-bound - Consider increasing memory bandwidth")
        
        print(f"  # Consider using --verbose flag for detailed tiling search process")
        print("-" * 80)
    
    def compare_hardware_configs(self, model_name: str, hardware_configs: Dict[str, HardwareConfig], 
                                baseline_hardware: str = None, verbose: bool = False) -> Dict[str, Any]:
        """Compare performance of a model across different hardware configurations.
        
        Args:
            model_name: Hugging Face model identifier
            hardware_configs: Dictionary mapping hardware names to HardwareConfig objects
            baseline_hardware: Name of baseline hardware for comparison (if None, uses first config)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing comparison results
        """
        if not hardware_configs:
            raise ValueError("At least one hardware configuration is required")
        
        if baseline_hardware is None:
            baseline_hardware = list(hardware_configs.keys())[0]
        
        if baseline_hardware not in hardware_configs:
            raise ValueError(f"Baseline hardware '{baseline_hardware}' not found in configurations")
        
        print(f"> Comparing hardware configurations for model: {model_name}")
        print(f"- Baseline: {baseline_hardware}")
        print("-" * 80)
        
        # Analyze model with each hardware configuration
        results = {}
        baseline_results = None
        
        for hw_name, hw_config in hardware_configs.items():
            print(f"Analyzing with {hw_name} hardware...")
            analyzer = MMAAnalyzer(hw_config)
            model_results = analyzer.analyze_model(model_name, verbose=False)
            results[hw_name] = model_results
            
            if hw_name == baseline_hardware:
                baseline_results = model_results
        
        # Create comparison summary
        comparison = {
            "model_name": model_name,
            "baseline_hardware": baseline_hardware,
            "hardware_results": results,
            "comparison_summary": self._create_comparison_summary(results, baseline_hardware)
        }
        
        return comparison
    
    def _create_comparison_summary(self, results: Dict[str, Dict], baseline_hardware: str) -> Dict[str, Any]:
        """Create a summary comparing all hardware configurations against baseline."""
        baseline = results[baseline_hardware]
        summary = {
            "overall_metrics": {},
            "operation_comparisons": []
        }
        
        # Compare overall metrics
        baseline_tps = baseline["tokens_per_second"]
        baseline_layer_time = baseline["per_layer_time_ms"]
        baseline_total_time = baseline["per_forward_time_ms"]
        
        for hw_name, result in results.items():
            if hw_name == baseline_hardware:
                continue
                
            tps_change = ((result["tokens_per_second"] - baseline_tps) / baseline_tps) * 100
            layer_time_change = ((result["per_layer_time_ms"] - baseline_layer_time) / baseline_layer_time) * 100
            total_time_change = ((result["per_forward_time_ms"] - baseline_total_time) / baseline_total_time) * 100
            
            summary["overall_metrics"][hw_name] = {
                "tokens_per_second": result["tokens_per_second"],
                "tps_change_percent": tps_change,
                "per_layer_time_ms": result["per_layer_time_ms"],
                "layer_time_change_percent": layer_time_change,
                "per_forward_time_ms": result["per_forward_time_ms"],
                "total_time_change_percent": total_time_change
            }
        
        # Compare individual operations
        baseline_ops = {op["name"]: op for op in baseline["operations"]}
        
        for hw_name, result in results.items():
            if hw_name == baseline_hardware:
                continue
                
            for op in result["operations"]:
                op_name = op["name"]
                if op_name in baseline_ops:
                    baseline_op = baseline_ops[op_name]
                    time_change = ((op["time_ms"] - baseline_op["time_ms"]) / baseline_op["time_ms"]) * 100
                    mfu_change = ((op["mfu"] - baseline_op["mfu"]) / baseline_op["mfu"]) * 100
                    
                    summary["operation_comparisons"].append({
                        "hardware": hw_name,
                        "operation": op_name,
                        "time_ms": op["time_ms"],
                        "time_change_percent": time_change,
                        "mfu": op["mfu"],
                        "mfu_change_percent": mfu_change,
                        "still_policy": op["still_policy"],
                        "baseline_time_ms": baseline_op["time_ms"],
                        "baseline_mfu": baseline_op["mfu"]
                    })
        
        return summary
    
    def print_hardware_comparison(self, comparison: Dict[str, Any]):
        """Print a formatted comparison of hardware configurations.
        
        Args:
            comparison: Comparison results from compare_hardware_configs()
        """
        model_name = comparison["model_name"]
        baseline_hw = comparison["baseline_hardware"]
        results = comparison["hardware_results"]
        summary = comparison["comparison_summary"]
        
        print(f"\n{'='*120}")
        print(f"> Hardware Performance Comparison")
        print(f"Model: {model_name}")
        print(f"Baseline: {baseline_hw}")
        print(f"{'='*120}")
        
        # Helpers for colored arrows and merged change field
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"
        def fmt_change(val: float, invert: bool = False) -> str:
            # invert=True means lower is better (latency)
            good = (val > 0) if not invert else (val < 0)
            # Arrow indicates direction (sign) regardless of goodness
            arrow = "^" if val >= 0 else "v"
            color = GREEN if good else RED
            return f" {color}{arrow}{val:+.1f}%{RESET}"
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        def pad(text: str, width: int) -> str:
            visible = ansi_re.sub("", text)
            pad_len = max(0, width - len(visible))
            return text + " " * pad_len

        # Print overall metrics comparison with merged change fields
        print(f"\n- Overall Performance Summary:")
        print(f"{'Hardware':<12} {'Tokens/s':<22} {'Layer Time (ms)':<28} {'Forward Time (ms)':<28}")
        print("-" * 120)
        
        # Print baseline first
        baseline = results[baseline_hw]
        baseline_tokens = f"{baseline['tokens_per_second']:.2f} (baseline)"
        baseline_layer = f"{baseline['per_layer_time_ms']:.4f} (baseline)"
        baseline_forward = f"{baseline['per_forward_time_ms']:.4f} (baseline)"
        print(f"{baseline_hw:<12} "
              f"{pad(baseline_tokens,22)} "
              f"{pad(baseline_layer,28)} "
              f"{pad(baseline_forward,28)}")
        
        # Print other configurations with better formatting
        for hw_name, metrics in summary["overall_metrics"].items():
            tps_field = f"{metrics['tokens_per_second']:.2f}" + fmt_change(metrics['tps_change_percent'], invert=False)
            layer_field = f"{metrics['per_layer_time_ms']:.4f}" + fmt_change(metrics['layer_time_change_percent'], invert=True)
            total_field = f"{metrics['per_forward_time_ms']:.4f}" + fmt_change(metrics['total_time_change_percent'], invert=True)
            print(f"{hw_name:<12} {pad(tps_field,22)} {pad(layer_field,28)} {pad(total_field,28)}")
        
        # Print operation-level comparison in a more organized way
        print(f"\n# Detailed Operation Analysis:")
        print(f"{'Operation':<12} {'Hardware':<10} {'Time (ms)':<22} {'MFU':<18} {'Policy':<10}")
        print("-" * 120)
        
        # Group operations by name
        operations = {}
        for comp in summary["operation_comparisons"]:
            op_name = comp["operation"]
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(comp)
        
        for op_name in sorted(operations.keys()):
            # Print baseline first
            baseline_op = next(op for op in results[baseline_hw]["operations"] if op["name"] == op_name)
            base_time = f"{baseline_op['time_ms']:.4f} (baseline)"
            base_mfu = f"{baseline_op['mfu']:.4f} (baseline)"
            print(f"{op_name:<12} {baseline_hw:<10} "
                  f"{pad(base_time,22)} {pad(base_mfu,18)} {baseline_op['still_policy']:<10}")
            
            # Print comparisons
            for comp in operations[op_name]:
                time_field = f"{comp['time_ms']:.4f}" + fmt_change(comp['time_change_percent'], invert=True)
                mfu_field = f"{comp['mfu']:.4f}" + fmt_change(comp['mfu_change_percent'], invert=False)
                print(f"{'':<12} {comp['hardware']:<10} {pad(time_field,22)} {pad(mfu_field,18)} {comp['still_policy']:<10}")
        
        print(f"{'='*120}")
    
    def print_comparison_summary(self, comparison: Dict[str, Any], comparison_type: str = "hardware"):
        """Print a concise summary of the comparison results.
        
        Args:
            comparison: Comparison results
            comparison_type: Type of comparison ("hardware" or "model_hardware")
        """
        if comparison_type == "hardware":
            model_name = comparison["model_name"]
            baseline_hw = comparison["baseline_hardware"]
            summary = comparison["comparison_summary"]
            
            print(f"\n# Quick Summary for {model_name}:")
            print(f"Baseline: {baseline_hw}")
            print("-" * 80)
            
            for hw_name, metrics in summary["overall_metrics"].items():
                tps_change = metrics["tps_change_percent"]
                time_change = metrics["total_time_change_percent"]
                
                if tps_change > 0:
                    perf_emoji = ">"
                    perf_desc = f"{tps_change:+.1f}% faster"
                else:
                    perf_emoji = "<"
                    perf_desc = f"{tps_change:+.1f}% slower"
                
                if time_change < 0:
                    time_desc = f"{abs(time_change):.1f}% less latency"
                else:
                    time_desc = f"{time_change:+.1f}% more latency"
                
                print(f"{perf_emoji} {hw_name:<12}: {perf_desc}, {time_desc}")
            
            # Add a visual performance comparison (scaled to max TPS)
            print(f"\n- Performance Visualization (bars scaled to max Tokens/s):")
            print("=" * 60)
            
            # Collect TPS values
            baseline_results = comparison["hardware_results"][baseline_hw]
            baseline_tps = baseline_results["tokens_per_second"]
            all_items = [(baseline_hw, baseline_tps, True)] + [
                (hw_name, m["tokens_per_second"], False) for hw_name, m in summary["overall_metrics"].items()
            ]
            max_tps = max(tps for _, tps, _ in all_items) or 1.0
            
            for name, tps, is_baseline in sorted(all_items, key=lambda x: -x[1]):
                bar_length = max(1, int(tps / max_tps * 20))
                bar = "#" * bar_length + "." * (20 - bar_length)
                if is_baseline:
                    rel = "1.00x vs baseline"
                else:
                    rel_speed = tps / baseline_tps if baseline_tps > 0 else 0
                    rel = f"{rel_speed:.2f}x vs baseline"
                suffix = " (baseline)" if is_baseline else ""
                print(f"{name:<12}: {bar} {tps:>7.1f} tok/s   {rel}{suffix}")
        
        elif comparison_type == "model_hardware":
            baseline_model = comparison["baseline_model"]
            baseline_hw = comparison["baseline_hardware"]
            summary = comparison["comparison_summary"]
            
            print(f"\n# Quick Summary:")
            print(f"Baseline: {baseline_model} on {baseline_hw}")
            print("-" * 80)
            
            for combo_name, metrics in summary["overall_metrics"].items():
                model_name = metrics["model_name"].split('/')[-1]
                hw_name = metrics["hardware_name"]
                tps_change = metrics["tps_change_percent"]
                time_change = metrics["total_time_change_percent"]
                
                if tps_change > 0:
                    perf_emoji = ">"
                    perf_desc = f"{tps_change:+.1f}% faster"
                else:
                    perf_emoji = "<"
                    perf_desc = f"{tps_change:+.1f}% slower"
                
                if time_change < 0:
                    time_desc = f"{abs(time_change):.1f}% less latency"
                else:
                    time_desc = f"{time_change:+.1f}% more latency"
                
                print(f"{perf_emoji} {model_name} on {hw_name:<10}: {perf_desc}, {time_desc}")
            
            # Add a visual performance comparison (scaled to max TPS)
            print(f"\n- Performance Visualization (bars scaled to max Tokens/s):")
            print("=" * 60)
            
            # Get baseline TPS from the comparison results
            baseline_key = f"{baseline_model.split('/')[-1]}_{baseline_hw}"
            baseline_results = comparison["combinations"][baseline_key]["results"]
            baseline_tps = baseline_results["tokens_per_second"]
            
            # Collect TPS values
            items = []
            for _, metrics in summary["overall_metrics"].items():
                model_name = metrics["model_name"].split('/')[-1]
                hw_name = metrics["hardware_name"]
                tps = metrics["tokens_per_second"]
                items.append((f"{model_name} on {hw_name}", tps, False))
            items.append((f"{baseline_model.split('/')[-1]} on {baseline_hw}", baseline_tps, True))
            max_tps = max(tps for _, tps, _ in items) or 1.0
            
            for name, tps, is_baseline in sorted(items, key=lambda x: -x[1]):
                bar_length = max(1, int(tps / max_tps * 20))
                bar = "#" * bar_length + "." * (20 - bar_length)
                if is_baseline:
                    rel = "1.00x vs baseline"
                else:
                    rel_speed = tps / baseline_tps if baseline_tps > 0 else 0
                    rel = f"{rel_speed:.2f}x vs baseline"
                suffix = " (baseline)" if is_baseline else ""
                print(f"{name:<28}: {bar} {tps:>7.1f} tok/s   {rel}{suffix}")
    
    def compare_models_and_hardware(self, model_hardware_pairs: List[Tuple[str, str, HardwareConfig]], 
                                  baseline_pair: Tuple[str, str] = None, verbose: bool = False) -> Dict[str, Any]:
        """Compare different model-hardware combinations.
        
        Args:
            model_hardware_pairs: List of (model_name, hardware_name, hardware_config) tuples
            baseline_pair: (model_name, hardware_name) tuple for baseline comparison
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing comparison results
        """
        if not model_hardware_pairs:
            raise ValueError("At least one model-hardware pair is required")
        
        if baseline_pair is None:
            baseline_pair = (model_hardware_pairs[0][0], model_hardware_pairs[0][1])
        
        print(f"- Baseline: {baseline_pair[0]} on {baseline_pair[1]}")
        print("-" * 80)
        
        # Analyze each model-hardware combination
        results = {}
        baseline_results = None
        
        for model_name, hw_name, hw_config in model_hardware_pairs:
            combo_name = f"{model_name.split('/')[-1]}_{hw_name}"
            print(f"Analyzing {model_name} on {hw_name} hardware...")
            
            analyzer = MMAAnalyzer(hw_config)
            model_results = analyzer.analyze_model(model_name, verbose=False)
            results[combo_name] = {
                "model_name": model_name,
                "hardware_name": hw_name,
                "results": model_results
            }
            
            if model_name == baseline_pair[0] and hw_name == baseline_pair[1]:
                baseline_results = model_results
        
        # Create comparison summary
        comparison = {
            "baseline_model": baseline_pair[0],
            "baseline_hardware": baseline_pair[1],
            "combinations": results,
            "comparison_summary": self._create_model_hardware_comparison_summary(results, baseline_pair)
        }
        
        return comparison
    
    def _create_model_hardware_comparison_summary(self, results: Dict, baseline_pair: Tuple[str, str]) -> Dict[str, Any]:
        """Create a summary comparing all model-hardware combinations against baseline."""
        baseline_key = f"{baseline_pair[0].split('/')[-1]}_{baseline_pair[1]}"
        baseline = results[baseline_key]["results"]
        
        summary = {
            "overall_metrics": {},
            "operation_comparisons": [],
            "diagnostics": {}
        }
        
        # Compare overall metrics
        baseline_tps = baseline["tokens_per_second"]
        baseline_layer_time = baseline["per_layer_time_ms"]
        baseline_total_time = baseline["per_forward_time_ms"]
        
        for combo_name, combo_data in results.items():
            if combo_name == baseline_key:
                continue
                
            result = combo_data["results"]
            tps_change = ((result["tokens_per_second"] - baseline_tps) / baseline_tps) * 100
            layer_time_change = ((result["per_layer_time_ms"] - baseline_layer_time) / baseline_layer_time) * 100
            total_time_change = ((result["per_forward_time_ms"] - baseline_total_time) / baseline_total_time) * 100
            
            summary["overall_metrics"][combo_name] = {
                "model_name": combo_data["model_name"],
                "hardware_name": combo_data["hardware_name"],
                "tokens_per_second": result["tokens_per_second"],
                "tps_change_percent": tps_change,
                "per_layer_time_ms": result["per_layer_time_ms"],
                "layer_time_change_percent": layer_time_change,
                "per_forward_time_ms": result["per_forward_time_ms"],
                "total_time_change_percent": total_time_change
            }

            # Diagnostics: attribute differences
            diag = {
                "top_slowest_ops": [],
                "policy_changes": [],
                "mfu_changes": [],
            }
            # Build op maps
            base_ops = {op["name"]: op for op in baseline["operations"]}
            for op in result["operations"]:
                name = op["name"]
                if name in base_ops:
                    b = base_ops[name]
                    time_diff = op["time_ms"] - b["time_ms"]
                    mfu_diff = op["mfu"] - b["mfu"]
                    # Track slowest deltas
                    diag["top_slowest_ops"].append({
                        "operation": name,
                        "delta_ms": time_diff,
                        "new_time_ms": op["time_ms"],
                        "base_time_ms": b["time_ms"],
                        "bound": "bandwidth" if op.get("bw_bound_cycles",0) >= op.get("compute_bound_cycles",0) else "compute"
                    })
                    # Policy change
                    if op["still_policy"] != b["still_policy"]:
                        diag["policy_changes"].append({
                            "operation": name,
                            "base_policy": b["still_policy"],
                            "new_policy": op["still_policy"]
                        })
                    # MFU change
                    if abs(mfu_diff) > 1e-6:
                        diag["mfu_changes"].append({
                            "operation": name,
                            "delta_mfu": mfu_diff,
                            "new_mfu": op["mfu"],
                            "base_mfu": b["mfu"]
                        })
            # Sort slowest ops by absolute regression first (desc)
            diag["top_slowest_ops"].sort(key=lambda x: x["delta_ms"], reverse=True)
            summary["diagnostics"][combo_name] = diag
        
        return summary
    
    def print_model_hardware_comparison(self, comparison: Dict[str, Any]):
        """Print a formatted comparison of model-hardware combinations.
        
        Args:
            comparison: Comparison results from compare_models_and_hardware()
        """
        baseline_model = comparison["baseline_model"]
        baseline_hw = comparison["baseline_hardware"]
        results = comparison["combinations"]
        summary = comparison["comparison_summary"]
        
        print(f"\n{'='*140}")
        print(f"> Model-Hardware Combination Comparison")
        print(f"Baseline: {baseline_model} on {baseline_hw}")
        print(f"{'='*140}")
        
        # Print overall metrics comparison in a cleaner format
        print(f"\n- Performance Summary:")
        print(f"{'Model':<18} {'Hardware':<10} {'Tokens/s':<10} {'Change':<12} {'Layer Time (ms)':<15} {'Change':<12} {'Forward Time (ms)':<18} {'Change':<12}")
        print("-" * 140)
        
        # Print baseline first
        baseline_key = f"{baseline_model.split('/')[-1]}_{baseline_hw}"
        baseline = results[baseline_key]["results"]
        print(f"{baseline_model.split('/')[-1]:<18} {baseline_hw:<10} {baseline['tokens_per_second']:<10.2f} {'(baseline)':<12} "
              f"{baseline['per_layer_time_ms']:<15.4f} {'(baseline)':<12} "
              f"{baseline['per_forward_time_ms']:<18.4f} {'(baseline)':<12}")
        
        # Print other combinations with better formatting
        for combo_name, metrics in summary["overall_metrics"].items():
            tps_arrow = "^" if metrics["tps_change_percent"] > 0 else "v"
            layer_arrow = "^" if metrics["layer_time_change_percent"] < 0 else "v"  # Lower is better
            total_arrow = "^" if metrics["total_time_change_percent"] < 0 else "v"  # Lower is better
            
            tps_change_str = f"{tps_arrow}{metrics['tps_change_percent']:+.1f}%"
            layer_change_str = f"{layer_arrow}{metrics['layer_time_change_percent']:+.1f}%"
            total_change_str = f"{total_arrow}{metrics['total_time_change_percent']:+.1f}%"
            
            print(f"{metrics['model_name'].split('/')[-1]:<18} {metrics['hardware_name']:<10} "
                  f"{metrics['tokens_per_second']:<10.2f} {tps_change_str:<12} "
                  f"{metrics['per_layer_time_ms']:<15.4f} {layer_change_str:<12} "
                  f"{metrics['per_forward_time_ms']:<18.4f} {total_change_str:<12}")
        
        print(f"{'='*140}")

        # Diagnostics summary to attribute differences
        print(f"\n# Diagnostics (why performance differs):")
        print("-" * 140)
        diags = summary.get("diagnostics", {})
        if not diags:
            print("No diagnostics available.")
            return
        for combo_name, diag in diags.items():
            model_name = summary["overall_metrics"][combo_name]["model_name"].split('/')[-1]
            hw_name = summary["overall_metrics"][combo_name]["hardware_name"]
            print(f"- {model_name} on {hw_name}")
            # Top slowest ops
            slow_ops = diag.get("top_slowest_ops", [])[:5]
            if slow_ops:
                print("  - Top time regressions (ms, bound):")
                for s in slow_ops:
                    sign = "+" if s["delta_ms"] >= 0 else ""
                    print(f"    - {s['operation']:<10} {sign}{s['delta_ms']:.4f} ms (new {s['new_time_ms']:.4f}, base {s['base_time_ms']:.4f}) [{s['bound']}]")
            # Policy changes
            policies = diag.get("policy_changes", [])
            if policies:
                print("  - Still policy changes:")
                for p in policies:
                    print(f"    - {p['operation']:<10} {p['base_policy']} -> {p['new_policy']}")
            # MFU changes
            mfus = diag.get("mfu_changes", [])[:5]
            if mfus:
                print("  - MFU changes:")
                for m in mfus:
                    sign = "+" if m["delta_mfu"] >= 0 else ""
                    print(f"    - {m['operation']:<10} {sign}{m['delta_mfu']:.4f} (new {m['new_mfu']:.4f}, base {m['base_mfu']:.4f})")


# Example usage
if __name__ == "__main__":
    # Create analyzer with default hardware
    analyzer = MMAAnalyzer()
    
    # Analyze Qwen3-1.7B
    results = analyzer.analyze_model("Qwen/Qwen3-1.7B", verbose=True)
    analyzer.print_analysis_summary(results)
