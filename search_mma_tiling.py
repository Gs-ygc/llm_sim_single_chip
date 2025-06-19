import numpy as np
from pprint import pprint

class MMAConfig:
    def __init__(self, name, bs, m, k, n, weight_bytes, fa_qk, fa_pv):
        self.name = name
        self.bs = bs
        self.m = m
        self.k = k
        self.n = n
        self.weight_bytes = weight_bytes
        self.fa_qk = fa_qk
        self.fa_pv = fa_pv

    def __str__(self):
        return f"{self.name}\t bs={self.bs}\t m={self.m}\t k={self.k}\t n={self.n}"

ds_v3_bs32_configs = [
    MMAConfig("up proj shared", 1, 64, 7168, 4096, 1, 0, 0),
    MMAConfig("down proj shared", 1, 64, 2048, 7168, 1, 0, 0),
    MMAConfig("up proj routed", 2, 512, 7168, 4096, 1, 0, 0),
    MMAConfig("down proj route", 2, 512, 2048, 7168, 1, 0, 0),
    MMAConfig("q_up", 1, 64, 128, 24576, 1, 0, 0),
    MMAConfig("Q @ W_{UV}^T", 128, 64, 128, 512, 1, 0, 0),
    MMAConfig("kv_nope_mm_W_KV_down", 1, 64, 7168, 576, 1, 0, 0),
    MMAConfig("qK", 32, 256, 576, 8192, 1, fa_qk=True, fa_pv=False),
    MMAConfig("PV", 32, 256, 8192, 512, 1, fa_qk=False, fa_pv=True),
    MMAConfig("O @ W_{UV}", 128, 64, 512, 128, 1, 0, 0),
    MMAConfig("O proj", 1, 64, 16384, 7168, 1, 0, 0),
]



mtp_num = 64
sd_mal = 3.5

def get_qwen3_bs1_mtp64_configs():
    hidden_dim = 5120
    num_q_heads = 64
    num_kv_heads = 8
    group_size = num_q_heads // num_kv_heads
    head_dim = 128
    FA_len_tile = 512
    total_context_len = 8192
    n_fa_chunks = total_context_len // FA_len_tile
    n_layers = 64
    qwen3_bs1_mtp64_configs = [
        # MMAConfig("up proj", 1, mtp_num, hidden_dim, 25600 * 2, 0.5, 0, 0),
        # MMAConfig("down proj", 1, mtp_num, 25600, hidden_dim, 0.5, 0, 0),

        # MMAConfig("q_proj", 1, mtp_num, hidden_dim, num_q_heads * head_dim, 0.5, 0, 0),
        # MMAConfig("kv_proj", 1, mtp_num, hidden_dim, num_kv_heads * head_dim, 0.5, 0, 0),

        # MMAConfig("qk", num_kv_heads * n_fa_chunks, mtp_num * group_size, head_dim, FA_len_tile, 1, fa_qk=True, fa_pv=False),
        # MMAConfig("pv", num_kv_heads * n_fa_chunks, mtp_num * group_size, FA_len_tile, head_dim, 1, fa_qk=False, fa_pv=True),

        # MMAConfig("o_proj", 1, mtp_num, num_q_heads * head_dim, hidden_dim, 0.5, 0, 0),
        MMAConfig("ideal", 1, 2048, 2048, 2048, 1, 0, 0),
    ]
    return qwen3_bs1_mtp64_configs, n_layers

configs, n_layers = get_qwen3_bs1_mtp64_configs()

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

def compute_per_tile_cycles(config, tile_m, tile_k, tile_n):
    m, k, n = config.m, config.k, config.n
    num_tile_m = np.ceil(m / tile_m)
    num_tile_k = np.ceil(k / tile_k)
    num_tile_n = np.ceil(n / tile_n)
    weight_bytes = config.weight_bytes
    fa_qk = config.fa_qk
    fa_pv = config.fa_pv

    A_bytes = bytes_per_element
    B_bytes = weight_bytes
    C_bytes = bytes_per_acc

    if fa_qk:
        C_bytes = 0
    elif fa_pv:
        A_bytes = 0

    tile_memory_size = \
        tile_m * tile_k * A_bytes + \
        tile_k * tile_n * B_bytes + \
        tile_m * tile_n * C_bytes / num_tile_k
    bw_bound_cycles = tile_memory_size / DRAM_per_cycle
    still_policy = "c_still"
    still_reuse_factor = num_tile_k
    
    a_still_tile_memory_size = \
        tile_m * tile_k * A_bytes / num_tile_n + \
        tile_k * tile_n * B_bytes + \
        tile_m * tile_n * C_bytes
    a_still_bw_bound_cycles = a_still_tile_memory_size / DRAM_per_cycle

    if a_still_bw_bound_cycles < bw_bound_cycles:
        bw_bound_cycles = a_still_bw_bound_cycles
        still_policy = "a_still"
        still_reuse_factor = num_tile_n

    b_still_tile_memory_size = \
        tile_m * tile_k * A_bytes + \
        tile_k * tile_n * B_bytes / num_tile_m + \
        tile_m * tile_n * C_bytes
    b_still_bw_bound_cycles = b_still_tile_memory_size / DRAM_per_cycle

    if b_still_bw_bound_cycles < bw_bound_cycles:
        bw_bound_cycles = b_still_bw_bound_cycles
        still_policy = "b_still"
        still_reuse_factor = num_tile_m

    ideal_compute_cycles = tile_m * tile_k * tile_n / mult_per_cycle
    mn_min = min(tile_m, tile_n)
    if mn_min > 64:
        compute_bound_cycles = tile_m * tile_k * tile_n / mult_per_cycle
    else:
        discounted_mult_per_cycle = mult_per_cycle * mn_min / 64
        compute_bound_cycles = tile_m * tile_k * tile_n / discounted_mult_per_cycle

    max_cycles = max(bw_bound_cycles, compute_bound_cycles)
    mfu = ideal_compute_cycles / max_cycles

    return mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor


def calculate_bandwidth_requirements(config, tile_m, tile_k, tile_n):
    # Calculate number of tiles
    m, k, n = config.m, config.k, config.n
    num_tile_m = (m + tile_m - 1) // tile_m
    num_tile_k = (k + tile_k - 1) // tile_k
    num_tile_n = (n + tile_n - 1) // tile_n
    
    mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor = compute_per_tile_cycles(config, tile_m, tile_k, tile_n)
    total_cycles = max(bw_bound_cycles, compute_bound_cycles) * num_tile_m*num_tile_n*num_tile_k
    
    return total_cycles, mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor

def compute_in_sram_info(global_still_policy, tile_m, tile_k, tile_n):
    reg_m, reg_k, reg_n = tile_quants

    print(f"\ntile_m: {tile_m}, tile_k: {tile_k}, tile_n: {tile_n}")
    print(f"reg_m: {reg_m}, reg_k: {reg_k}, reg_n: {reg_n}")

    if tile_m == reg_m and tile_k == reg_k:
        local_still_policy = "a_still"
    elif tile_m == reg_m and tile_n == reg_n:
        local_still_policy = "c_still"
    elif tile_k == reg_k and tile_n == reg_n:
        local_still_policy = "b_still"
    else:
        raise ValueError(f"Invalid tile sizes: {tile_m}, {tile_k}, {tile_n}")

    load_c_to_reg_before_mma = False
    if global_still_policy == "c_still" and local_still_policy != "c_still":
        load_c_to_reg_before_mma = True

    macro_op_count = 0
    if local_still_policy == "a_still":
        bytes_a = reg_m * reg_k * bytes_per_element / (tile_n / reg_n)
        bytes_b = reg_k * reg_n * bytes_per_element
        bytes_c = reg_m * reg_n * bytes_per_acc
        macro_op_count = tile_n / reg_n
    elif local_still_policy == "b_still":
        bytes_a = reg_m * reg_k * bytes_per_element
        bytes_b = reg_k * reg_n * bytes_per_element / (tile_m / reg_m)
        bytes_c = reg_m * reg_n * bytes_per_acc
        macro_op_count = tile_m / reg_m
    else:
        assert local_still_policy == "c_still"
        bytes_a = reg_m * reg_k * bytes_per_element
        bytes_b = reg_k * reg_n * bytes_per_element
        bytes_c = reg_m * reg_n * bytes_per_acc / (tile_k / reg_k)
        macro_op_count = tile_k / reg_k

    load_c_bytes = reg_m * reg_n * bytes_per_acc if load_c_to_reg_before_mma else 0

    byte_loaded_from_sram = (bytes_a + bytes_b + bytes_c + load_c_bytes) * macro_op_count

    c_count_factor = 2 if load_c_to_reg_before_mma else 1

    if global_still_policy == "c_still":
        dram_usage = tile_m * tile_k * bytes_per_element + tile_k * tile_n * bytes_per_element
    elif global_still_policy == "a_still":
        dram_usage = tile_k * tile_n * bytes_per_element + tile_m * tile_n * bytes_per_acc
    elif global_still_policy == "b_still":
        dram_usage = tile_m * tile_k * bytes_per_element + tile_m * tile_n * bytes_per_acc
    else:
        raise ValueError(f"Invalid global still policy: {global_still_policy}")

    memory_cycles = dram_usage / DRAM_per_cycle

    sram_bw_per_cycle = byte_loaded_from_sram / memory_cycles

    print(f"Load A bytes: {bytes_a}, Load B bytes: {bytes_b}, Store C bytes: {bytes_c}, Load C bytes: {load_c_bytes}, Memory usage: {dram_usage}, Memory cycles: {memory_cycles}, SRAM BW per cycle: {sram_bw_per_cycle}")

    return {
        "bytes_a": bytes_a,
        "bytes_b": bytes_b,
        "bytes_c": bytes_c,
        "load_c_bytes": load_c_bytes,
        "memory_usage": dram_usage,
        "sram_bw_per_cycle": sram_bw_per_cycle,
    }

def find_optimal_tiling(config):
    best_tiling = None
    best_total_cycles = float('inf')
    best_info = {}
    
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
                total_cycles, mfu, bw_bound_cycles, compute_bound_cycles, still_policy, still_reuse_factor = calculate_bandwidth_requirements(
                    config, tile_m, tile_k, tile_n
                )
                total_cycles *= config.bs

                # Update best tiling if this is better
                if total_cycles < best_total_cycles:
                    # print(f"update best total cycles: {best_total_cycles} -> {total_cycles}")
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
                    # pprint(best_info)
    
    return best_tiling, best_info

# Analyze each configuration
print("Analyzing tiling strategies for each configuration:")
print("-" * 80)
cycles = 0
freq = 2 * 1000 * 1000 * 1000
for config in configs:
    best_tiling, best_info = find_optimal_tiling(config)
    if best_tiling:
        # print(f"\nConfiguration: {config}")
        # print(f"Best tiling: M={best_tiling[0]}, K={best_tiling[1]}, N={best_tiling[2]}")
        # print(f"Total cycles: {best_info['total_cycles']}")
        # print(f"MFU: {best_info['mfu']:.2f}, BW bound cycles: {best_info['bw_bound_cycles']:.2f}, ideal compute cycles: {best_info['compute_bound_cycles']:.2f}, compute bound cycles: {best_info['compute_bound_cycles']:.2f}")
        # print(f"Still policy: {best_info['still_policy']}, still reuse factor: {best_info['still_reuse_factor']}")

        time_ms = best_info['total_cycles'] / freq * 1000
        cycles += best_info['total_cycles']
        mfu = best_info['mfu']
        print(f"{config.name.ljust(20)} {time_ms:.4f} ms, MFU: {mfu:.4f}", end=',   ')
        print(f"M={config.m}, K={config.k}, N={config.n},".ljust(30), end='')
        print(f"Tiling: M={best_tiling[0]}, K={best_tiling[1]}, N={best_tiling[2]},".ljust(30), end='')
        print(f"Still policy: {best_info['still_policy']}")
        in_sram_info = compute_in_sram_info(best_info['still_policy'], best_tiling[0], best_tiling[1], best_tiling[2])

    else:
        print(f"\nConfiguration: {config}")
        print("No valid tiling found that fits in SRAM")

print(f"Total cycles: {cycles}")
layer_time = cycles / freq * 1000
print(f"Per layer time: {layer_time} ms")
print(f"Per forward time: {n_layers * layer_time} ms")
print(f"With spec decode MAL={sd_mal}, Token per second: {1000 / (n_layers * layer_time) * sd_mal}")