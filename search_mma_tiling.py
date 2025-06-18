import numpy as np

class MMAConfig:
    def __init__(self, name, bs, m, k, n):
        self.name = name
        self.bs = bs
        self.m = m
        self.k = k
        self.n = n

    def __str__(self):
        return f"{self.name}\t bs={self.bs}\t m={self.m}\t k={self.k}\t n={self.n}"

ds_v3_bs32_configs = [
    MMAConfig("up proj shared", 1, 64, 7168, 4096),
    MMAConfig("down proj shared", 1, 64, 2048, 7168),
    MMAConfig("up proj routed", 2, 512, 7168, 4096),
    MMAConfig("down proj route", 2, 512, 2048, 7168),
    MMAConfig("q_up", 1, 64, 128, 24576),
    MMAConfig("Q @ W_{UV}^T", 128, 64, 128, 512),
    MMAConfig("kv_nope_mm_W_KV_down", 1, 64, 7168, 576),
    MMAConfig("qK", 32, 256, 576, 8192),
    MMAConfig("PV", 32, 256, 8192, 512),
    MMAConfig("O @ W_{UV}", 128, 64, 512, 128),
    MMAConfig("O proj", 1, 64, 16384, 7168),
]


mtp_num = 64
hidden_dim = 5120
num_q_heads = 64
num_kv_heads = 8
group_size = num_q_heads // num_kv_heads
head_dim = 128
FA_len_tile = 512
qwen3_bs1_mtp64_configs = [
    # MMAConfig("up proj", 1, mtp_num, hidden_dim, 25600 * 2),
    # MMAConfig("down proj", 1, mtp_num, 25600, hidden_dim),

    MMAConfig("q_proj", 1, mtp_num*8, hidden_dim, num_q_heads * head_dim),
    # MMAConfig("kv_proj", 1, mtp_num*8, hidden_dim, num_kv_heads * head_dim),
    # MMAConfig("qk", 1, mtp_num * group_size, head_dim, FA_len_tile),
    # MMAConfig("pv", 1, mtp_num * group_size, FA_len_tile, head_dim),
    # MMAConfig("o_proj", 1, mtp_num, num_q_heads * head_dim, hidden_dim),
]

configs = qwen3_bs1_mtp64_configs

SRAM_size = 2*1024*1024
DRAM_per_cycle = 24
mult_per_cycle = 32*8*32

tile_quants = (64, 128, 64)
bytes_per_element = 1  #fp8
bytes_per_acc = 2  # bf16

def calculate_memory_requirements(m, k, n, tile_m, tile_k, tile_n):
    # Calculate memory needed for tiled matrices
    # A: m x k matrix
    # B: k x n matrix
    # C: m x n matrix
    
    # Memory for tiled A
    a_memory = tile_m * tile_k * bytes_per_element
    
    # Memory for tiled B
    b_memory = tile_k * tile_n * bytes_per_element
    
    # Memory for tiled C
    c_memory = tile_m * tile_n * bytes_per_acc
    
    total_memory = a_memory + b_memory + c_memory
    return total_memory, a_memory, b_memory, c_memory

def compute_per_tile_cycles(tile_m, tile_k, tile_n, still_policy):
    pass

def calculate_bandwidth_requirements(m, k, n, tile_m, tile_k, tile_n):
    # Calculate number of tiles
    num_tiles_m = (m + tile_m - 1) // tile_m
    num_tiles_k = (k + tile_k - 1) // tile_k
    num_tiles_n = (n + tile_n - 1) // tile_n
    
    # Calculate memory bandwidth for each matrix
    
    # keep C still
    # A matrix: read once for each tile_k
    a_bandwidth = m * k * bytes_per_element * num_tiles_n
    # B matrix: read once for each tile_m
    b_bandwidth = k * n * bytes_per_element * num_tiles_m
    # C matrix: read and write for each tile
    c_bandwidth = m * n * bytes_per_acc
    c_still_bandwidth = a_bandwidth + b_bandwidth + c_bandwidth

    # keep A still
    a_bandwidth = m * k * bytes_per_element
    # B matrix: read once for each tile_m
    b_bandwidth = k * n * bytes_per_element * num_tiles_m
    # C matrix: read and write for each tile
    c_bandwidth = m * n * bytes_per_acc * num_tiles_k * 2
    a_still_bandwidth = a_bandwidth + b_bandwidth + c_bandwidth

    # keep B still
    a_bandwidth = m * k * bytes_per_element * num_tiles_n
    b_bandwidth = k * n * bytes_per_element
    c_bandwidth = m * n * bytes_per_acc * num_tiles_k * 2
    b_still_bandwidth = a_bandwidth + b_bandwidth + c_bandwidth

    print(f"m: {m}, k: {k}, n: {n}, tile_m: {tile_m}, tile_k: {tile_k}, tile_n: {tile_n}")
    print(f"c_still_bandwidth: {c_still_bandwidth}, a_still_bandwidth: {a_still_bandwidth}, b_still_bandwidth: {b_still_bandwidth}")

    return a_still_bandwidth, b_still_bandwidth, c_still_bandwidth

    

def find_optimal_tiling(config):
    best_tiling = None
    best_reuse = 0
    best_bandwidth = float('inf')
    best_sram_cost = float('inf')
    
    # Try different tile sizes
    for tile_m in np.arange(tile_quants[0], config.m + tile_quants[0], tile_quants[0]):
        for tile_k in np.arange(tile_quants[1], config.k + tile_quants[1], tile_quants[1]):
            for tile_n in np.arange(tile_quants[2], config.n + tile_quants[2], tile_quants[2]):
                # Skip if tile sizes are larger than matrix dimensions
                if tile_m > config.m or tile_k > config.k or tile_n > config.n:
                    # print(f"Skipping tile sizes: {tile_m}, {tile_k}, {tile_n} for {config}")
                    continue
                
                # Calculate memory requirements
                total_memory, a_memory, b_memory, c_memory = calculate_memory_requirements(
                    config.m, config.k, config.n, tile_m, tile_k, tile_n
                )
                
                # Skip if doesn't fit in SRAM
                if total_memory > SRAM_size:
                    # print(f"Skipping tile sizes: {tile_m}, {tile_k}, {tile_n} for {config} because it doesn't fit in SRAM")
                    continue
                
                # Calculate bandwidth requirements
                a_still_bandwidth, b_still_bandwidth, c_still_bandwidth = calculate_bandwidth_requirements(
                    config.m, config.k, config.n, tile_m, tile_k, tile_n
                )

                bandwidth = min(a_still_bandwidth, b_still_bandwidth, c_still_bandwidth)
                best_policy = "c_still" if best_bandwidth == c_still_bandwidth else "a_still" if best_bandwidth == a_still_bandwidth else "b_still"
                
                # Calculate reuse factor (higher is better)
                reuse_factor = (config.m * config.k * config.n) / bandwidth
                
                # Update best tiling if this is better
                if bandwidth < best_bandwidth:
                    print(f"update best bandwidth: {best_bandwidth} -> {bandwidth}")
                    best_reuse = reuse_factor
                    best_bandwidth = bandwidth
                    best_tiling = (tile_m, tile_k, tile_n)
                    best_policy = best_policy
                    best_sram_cost = total_memory
                else:
                    print(f"skip: {bandwidth} > {best_bandwidth}")
    
    return best_tiling, best_reuse, best_bandwidth, best_policy, best_sram_cost

# Analyze each configuration
print("Analyzing tiling strategies for each configuration:")
print("-" * 80)
for config in configs:
    best_tiling, best_reuse, best_bandwidth, best_policy, best_sram_cost = find_optimal_tiling(config)
    if best_tiling:
        print(f"\nConfiguration: {config}")
        print(f"Best tiling: M={best_tiling[0]}, K={best_tiling[1]}, N={best_tiling[2]}")
        print(f"Reuse factor: {best_reuse:.2f}")
        print(f"Total bandwidth: {best_bandwidth/1024/1024:.2f} MB")
        print(f"Best policy: {best_policy}")
        print(f"SRAM cost: {best_sram_cost/1024:.2f} KB")

        num_tile_m = np.ceil(config.m / best_tiling[0])
        num_tile_k = np.ceil(config.k / best_tiling[1])
        num_tile_n = np.ceil(config.n / best_tiling[2])

        if best_policy == "c_still":
            tile_memory_size = \
                best_tiling[0] * best_tiling[1] * bytes_per_element + \
                best_tiling[1] * best_tiling[2] * bytes_per_element + \
                best_tiling[0] * best_tiling[2] * bytes_per_acc / num_tile_k
            bw_bound_cycles = tile_memory_size / DRAM_per_cycle
        elif best_policy == "a_still":
            tile_memory_size = \
                best_tiling[0] * best_tiling[1] * bytes_per_element / num_tile_n + \
                best_tiling[1] * best_tiling[2] * bytes_per_element + \
                best_tiling[0] * best_tiling[2] * bytes_per_acc
            bw_bound_cycles = tile_memory_size / DRAM_per_cycle
        elif best_policy == "b_still":
            tile_memory_size = \
                best_tiling[0] * best_tiling[1] * bytes_per_element + \
                best_tiling[1] * best_tiling[2] * bytes_per_element / num_tile_m + \
                best_tiling[0] * best_tiling[2] * bytes_per_acc
            bw_bound_cycles = tile_memory_size / DRAM_per_cycle

        ideal_compute_cycles = best_tiling[0] * best_tiling[1] * best_tiling[2] / mult_per_cycle
        mn_min = min(best_tiling[0], best_tiling[2])
        if mn_min > 64:
            compute_bound_cycles = best_tiling[0] * best_tiling[1] * best_tiling[2] / mult_per_cycle
        else:
            discounted_mult_per_cycle = mult_per_cycle * mn_min / 64
            compute_bound_cycles = best_tiling[0] * best_tiling[1] * best_tiling[2] / discounted_mult_per_cycle

        max_cycles = max(bw_bound_cycles, ideal_compute_cycles)
        mfu = ideal_compute_cycles / max_cycles
        print(f"MFU: {mfu:.2f}, BW bound cycles: {bw_bound_cycles:.2f}, ideal compute cycles: {ideal_compute_cycles:.2f}, compute bound cycles: {compute_bound_cycles:.2f}")
            

    else:
        print(f"\nConfiguration: {config}")
        print("No valid tiling found that fits in SRAM")
