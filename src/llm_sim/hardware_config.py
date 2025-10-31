"""Hardware configuration for MMA simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareConfig:
    """Hardware configuration for MMA simulation.
    
    This class defines the hardware specifications for the matrix multiplication
    accelerator simulation, including memory hierarchy, compute units, and data types.
    """
    
    # Memory specifications
    sram_size: int = 1 * 1024 * 1024  # 1MB SRAM
    dram_bandwidth: int = 16  # bytes per cycle
    sram_bandwidth: int = 256  # bytes per cycle (SRAM/NOC effective bandwidth)
    
    # Compute specifications
    mult_per_cycle: int = 32 * 8 * 32  # multipliers per cycle
    
    # Data type specifications
    bytes_per_element: int = 1  # FP8 input
    bytes_per_acc: int = 2  # BF16 accumulation
    
    # Tiling constraints
    tile_quants: tuple = (64, 256, 64)  # (reg_m, reg_k, reg_n)
    
    # Additional parameters
    freq: float = 2 * 1000 * 1000 * 1000  # 2GHz frequency
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sram_size <= 0:
            raise ValueError("SRAM size must be positive")
        if self.dram_bandwidth <= 0:
            raise ValueError("DRAM bandwidth must be positive")
        if self.sram_bandwidth <= 0:
            raise ValueError("SRAM bandwidth must be positive")
        if self.mult_per_cycle <= 0:
            raise ValueError("Multipliers per cycle must be positive")
        if len(self.tile_quants) != 3:
            raise ValueError("Tile quants must have exactly 3 elements")
        if self.freq <= 0:
            raise ValueError("Frequency must be positive")
    
    def get_memory_hierarchy_info(self) -> dict:
        """Get memory hierarchy information.
        
        Returns:
            Dictionary containing memory hierarchy details
        """
        return {
            "sram_size_bytes": self.sram_size,
            "sram_size_mb": self.sram_size / (1024 * 1024),
            "dram_bandwidth_bytes_per_cycle": self.dram_bandwidth,
            "dram_bandwidth_gb_per_sec": self.dram_bandwidth * self.freq / (1024**3),
            "sram_bandwidth_bytes_per_cycle": self.sram_bandwidth,
            "sram_bandwidth_gb_per_sec": self.sram_bandwidth * self.freq / (1024**3),
        }
    
    def get_compute_info(self) -> dict:
        """Get compute unit information.
        
        Returns:
            Dictionary containing compute unit details
        """
        return {
            "multipliers_per_cycle": self.mult_per_cycle,
            "frequency_hz": self.freq,
            "frequency_ghz": self.freq / (1000**3),
            "peak_compute_ops_per_sec": self.mult_per_cycle * self.freq,
        }
    
    def get_data_type_info(self) -> dict:
        """Get data type information.
        
        Returns:
            Dictionary containing data type details
        """
        return {
            "input_bytes_per_element": self.bytes_per_element,
            "accumulation_bytes_per_element": self.bytes_per_acc,
            "input_precision": "FP8" if self.bytes_per_element == 1 else "Unknown",
            "accumulation_precision": "BF16" if self.bytes_per_acc == 2 else "Unknown",
        }
    
    def get_tiling_info(self) -> dict:
        """Get tiling configuration information.
        
        Returns:
            Dictionary containing tiling details
        """
        return {
            "register_tile_m": self.tile_quants[0],
            "register_tile_k": self.tile_quants[1], 
            "register_tile_n": self.tile_quants[2],
        }
    
    def print_summary(self):
        """Print hardware configuration summary."""
        print("\n=== Hardware Configuration Summary ===")
        
        # Memory hierarchy
        mem_info = self.get_memory_hierarchy_info()
        print(f"SRAM Size: {mem_info['sram_size_mb']:.1f} MB")
        print(f"DRAM Bandwidth: {mem_info['dram_bandwidth_gb_per_sec']:.1f} GB/s")
        
        # Compute units
        compute_info = self.get_compute_info()
        print(f"Peak Compute: {compute_info['peak_compute_ops_per_sec']/1e12:.1f} TOPS")
        print(f"Frequency: {compute_info['frequency_ghz']:.1f} GHz")
        
        # Data types
        dtype_info = self.get_data_type_info()
        print(f"Input Precision: {dtype_info['input_precision']}")
        print(f"Accumulation Precision: {dtype_info['accumulation_precision']}")
        
        # Tiling
        tile_info = self.get_tiling_info()
        print(f"Register Tiles: {tile_info['register_tile_m']}x{tile_info['register_tile_k']}x{tile_info['register_tile_n']}")
        print("=" * 40)


# Predefined hardware configurations
class HardwarePresets:
    """Predefined hardware configurations for different scenarios."""
    
    @staticmethod
    def xsai_config() -> HardwareConfig:
        """XSAI reference configuration (same as current default)."""
        return HardwareConfig(
            sram_size=1 * 1024 * 1024,    # 1MB
            dram_bandwidth=128,            # 16 bytes/cycle
            sram_bandwidth=128,           # 32 bytes/cycle (on-chip is wider)
            mult_per_cycle=32 * 8 * 32,   # same compute as default
            bytes_per_element=1,          # FP8
            bytes_per_acc=2,              # BF16
            tile_quants=(64, 256, 64),
            freq=2 * 1000 * 1000 * 1000,  # 2GHz
        )
    
    @staticmethod
    def mobile_config() -> HardwareConfig:
        """Mobile/edge device configuration."""
        return HardwareConfig(
            sram_size=512 * 1024,  # 512KB
            dram_bandwidth=8,      # 8 bytes/cycle
            sram_bandwidth=16,     # narrower on-chip
            mult_per_cycle=16 * 4 * 16,  # Smaller compute
            freq=1 * 1000 * 1000 * 1000,  # 1GHz
        )
    
    @staticmethod
    def datacenter_config() -> HardwareConfig:
        """Datacenter/server configuration."""
        return HardwareConfig(
            sram_size=4 * 1024 * 1024,  # 4MB
            dram_bandwidth=64,           # 64 bytes/cycle
            sram_bandwidth=512,          # 512 bytes/cycle
            mult_per_cycle=64 * 16 * 64,  # Larger compute
            freq=3 * 1000 * 1000 * 1000,  # 3GHz
        )
    
    @staticmethod
    def research_config() -> HardwareConfig:
        """Research/experimental configuration."""
        return HardwareConfig(
            sram_size=8 * 1024 * 1024,   # 8MB
            dram_bandwidth=128,           # 128 bytes/cycle
            sram_bandwidth=1024,          # 1024 bytes/cycle
            mult_per_cycle=128 * 32 * 128,  # Very large compute
            freq=2.5 * 1000 * 1000 * 1000,  # 2.5GHz
        )


# Example usage
if __name__ == "__main__":
    # Default configuration
    default_hw = HardwareConfig()
    default_hw.print_summary()
    
    # Mobile configuration
    mobile_hw = HardwarePresets.mobile_config()
    mobile_hw.print_summary()
    
    # Datacenter configuration
    datacenter_hw = HardwarePresets.datacenter_config()
    datacenter_hw.print_summary()
