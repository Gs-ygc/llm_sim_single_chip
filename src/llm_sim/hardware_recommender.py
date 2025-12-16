"""Hardware configuration recommender for LLM models."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .hardware_config import HardwareConfig, HardwarePresets
from .mma_analyzer import MMAAnalyzer


@dataclass
class HardwareRecommendation:
    """Hardware configuration recommendation."""
    
    config_name: str
    hardware_config: HardwareConfig
    performance_score: float
    cost_score: float
    efficiency_score: float
    reasoning: str
    pros: List[str]
    cons: List[str]


class HardwareRecommender:
    """Recommends optimal hardware configurations for LLM models."""
    
    def __init__(self, model_source: Optional[str] = None):
        """Initialize hardware recommender.
        
        Args:
            model_source: Model source ('huggingface' or 'modelscope'). If None, uses default.
        """
        self.presets = HardwarePresets()
        self.model_source = model_source
    
    def analyze_with_hardware(self, model_name: str, hardware_config: HardwareConfig) -> Dict:
        """Analyze model performance with specific hardware configuration.
        
        Args:
            model_name: Model identifier (from Hugging Face or ModelScope)
            hardware_config: Hardware configuration to test
            
        Returns:
            Analysis results
        """
        analyzer = MMAAnalyzer(hardware_config, model_source=self.model_source)
        return analyzer.analyze_model(model_name)
    
    def recommend_hardware(self, model_name: str, 
                          target_tokens_per_second: Optional[float] = None,
                          max_cost: Optional[str] = None,
                          use_case: str = "general") -> List[HardwareRecommendation]:
        """Recommend hardware configurations for a model.
        
        Args:
            model_name: Model identifier (from Hugging Face or ModelScope)
            target_tokens_per_second: Target performance (optional)
            max_cost: Maximum cost category ("low", "medium", "high")
            use_case: Use case ("mobile", "edge", "datacenter", "research")
            
        Returns:
            List of hardware recommendations sorted by suitability
        """
        print(f"> Analyzing hardware requirements for {model_name}")
        print(f"- Use case: {use_case}")
        if target_tokens_per_second:
            print(f"- Target performance: {target_tokens_per_second:.2f} tokens/second")
        print("-" * 60)
        
        # Define hardware configurations to test
        configs_to_test = self._get_configs_to_test(use_case, max_cost)
        
        recommendations = []
        
        for config_name, hw_config in configs_to_test:
            print(f"\n> Testing {config_name} configuration...")
            
            # Analyze performance
            results = self.analyze_with_hardware(model_name, hw_config)
            
            # Calculate scores
            performance_score = self._calculate_performance_score(results, target_tokens_per_second)
            cost_score = self._calculate_cost_score(config_name, hw_config)
            efficiency_score = self._calculate_efficiency_score(results, hw_config)
            
            # Generate reasoning
            reasoning, pros, cons = self._generate_recommendation_reasoning(
                config_name, results, hw_config, target_tokens_per_second, use_case
            )
            
            recommendation = HardwareRecommendation(
                config_name=config_name,
                hardware_config=hw_config,
                performance_score=performance_score,
                cost_score=cost_score,
                efficiency_score=efficiency_score,
                reasoning=reasoning,
                pros=pros,
                cons=cons
            )
            
            recommendations.append(recommendation)
        
        # Sort by overall suitability
        recommendations.sort(key=lambda x: (
            x.performance_score * 0.4 + 
            x.efficiency_score * 0.4 + 
            x.cost_score * 0.2
        ), reverse=True)
        
        return recommendations
    
    def _get_configs_to_test(self, use_case: str, max_cost: Optional[str]) -> List[Tuple[str, HardwareConfig]]:
        """Get hardware configurations to test based on use case and cost constraints."""
        configs = []
        
        # Always include mobile config for comparison
        configs.append(("Mobile", self.presets.mobile_config()))
        
        # Add configurations based on use case
        if use_case in ["mobile", "edge"]:
            configs.append(("Default", HardwareConfig()))
        elif use_case in ["datacenter", "server"]:
            configs.append(("Default", HardwareConfig()))
            configs.append(("Datacenter", self.presets.datacenter_config()))
        elif use_case in ["research", "experimental"]:
            configs.append(("Default", HardwareConfig()))
            configs.append(("Datacenter", self.presets.datacenter_config()))
            configs.append(("Research", self.presets.research_config()))
        else:  # general
            configs.append(("Default", HardwareConfig()))
            configs.append(("Datacenter", self.presets.datacenter_config()))
        
        # Filter by cost if specified
        if max_cost == "low":
            configs = [c for c in configs if c[0] in ["Mobile", "Default"]]
        elif max_cost == "medium":
            configs = [c for c in configs if c[0] in ["Mobile", "Default", "Datacenter"]]
        
        return configs
    
    def _calculate_performance_score(self, results: Dict, target_tokens_per_second: Optional[float]) -> float:
        """Calculate performance score (0-1, higher is better)."""
        actual_tps = results["tokens_per_second"]
        
        if target_tokens_per_second:
            # Score based on how close to target
            if actual_tps >= target_tokens_per_second:
                return 1.0
            else:
                return actual_tps / target_tokens_per_second
        else:
            # Score based on absolute performance (normalized)
            # Assume 100 tokens/second is excellent
            return min(actual_tps / 100.0, 1.0)
    
    def _calculate_cost_score(self, config_name: str, hw_config: HardwareConfig) -> float:
        """Calculate cost score (0-1, higher is better for cost-effectiveness)."""
        # Rough cost estimation based on hardware specs
        cost_factors = {
            "Mobile": 1.0,      # Lowest cost
            "Default": 0.8,    # Medium-low cost
            "Datacenter": 0.4, # Medium-high cost
            "Research": 0.2,   # Highest cost
        }
        
        base_score = cost_factors.get(config_name, 0.5)
        
        # Adjust based on hardware specs
        sram_factor = min(hw_config.sram_size / (4 * 1024 * 1024), 1.0)  # Normalize to 4MB
        compute_factor = min(hw_config.mult_per_cycle / (128 * 32 * 128), 1.0)  # Normalize to high-end
        
        # Lower specs = higher cost score (more cost-effective)
        adjusted_score = base_score * (1.0 - sram_factor * 0.3) * (1.0 - compute_factor * 0.3)
        
        return max(adjusted_score, 0.1)  # Minimum score
    
    def _calculate_efficiency_score(self, results: Dict, hw_config: HardwareConfig) -> float:
        """Calculate efficiency score (0-1, higher is better)."""
        # Calculate average MFU across operations
        if not results["operations"]:
            return 0.0
        
        avg_mfu = sum(op["mfu"] for op in results["operations"]) / len(results["operations"])
        
        # Calculate memory efficiency
        total_memory_usage = sum(
            self._estimate_memory_usage(op["dimensions"], op["tiling"]) 
            for op in results["operations"]
        )
        memory_efficiency = min(hw_config.sram_size / total_memory_usage, 1.0) if total_memory_usage > 0 else 1.0
        
        # Combine MFU and memory efficiency
        efficiency_score = avg_mfu * 0.7 + memory_efficiency * 0.3
        
        return efficiency_score
    
    def _estimate_memory_usage(self, dimensions: str, tiling: str) -> float:
        """Estimate memory usage for an operation."""
        # Parse dimensions and tiling
        # This is a simplified estimation
        return 1024 * 1024  # 1MB placeholder
    
    def _generate_recommendation_reasoning(self, config_name: str, results: Dict, 
                                         hw_config: HardwareConfig, 
                                         target_tokens_per_second: Optional[float],
                                         use_case: str) -> Tuple[str, List[str], List[str]]:
        """Generate reasoning for hardware recommendation."""
        actual_tps = results["tokens_per_second"]
        per_layer_time = results["per_layer_time_ms"]
        
        pros = []
        cons = []
        
        # Performance analysis
        if actual_tps >= 10:
            pros.append("High throughput performance")
        elif actual_tps >= 5:
            pros.append("Good throughput performance")
        else:
            cons.append("Limited throughput performance")
        
        # Memory analysis
        if hw_config.sram_size >= 2 * 1024 * 1024:
            pros.append("Large SRAM for complex models")
        elif hw_config.sram_size < 1024 * 1024:
            cons.append("Limited SRAM may constrain model size")
        
        # Compute analysis
        if hw_config.mult_per_cycle >= 64 * 16 * 64:
            pros.append("High compute capacity")
        elif hw_config.mult_per_cycle < 16 * 4 * 16:
            cons.append("Limited compute capacity")
        
        # Use case specific analysis
        if use_case == "mobile" and config_name == "Mobile":
            pros.append("Optimized for mobile deployment")
        elif use_case == "datacenter" and config_name == "Datacenter":
            pros.append("Optimized for datacenter deployment")
        elif use_case == "research" and config_name == "Research":
            pros.append("Maximum performance for research")
        
        # Target performance analysis
        if target_tokens_per_second:
            if actual_tps >= target_tokens_per_second:
                pros.append(f"Meets target performance ({target_tokens_per_second:.1f} tokens/s)")
            else:
                cons.append(f"Below target performance ({target_tokens_per_second:.1f} tokens/s)")
        
        # Generate reasoning
        reasoning_parts = []
        
        if actual_tps >= 10:
            reasoning_parts.append("Excellent performance")
        elif actual_tps >= 5:
            reasoning_parts.append("Good performance")
        else:
            reasoning_parts.append("Moderate performance")
        
        if hw_config.sram_size >= 2 * 1024 * 1024:
            reasoning_parts.append("with large memory capacity")
        elif hw_config.sram_size < 1024 * 1024:
            reasoning_parts.append("with compact memory footprint")
        
        if config_name == "Mobile":
            reasoning_parts.append("suitable for mobile/edge deployment")
        elif config_name == "Datacenter":
            reasoning_parts.append("suitable for datacenter deployment")
        elif config_name == "Research":
            reasoning_parts.append("suitable for research applications")
        else:
            reasoning_parts.append("suitable for general applications")
        
        reasoning = ", ".join(reasoning_parts) + "."
        
        return reasoning, pros, cons
    
    def print_recommendations(self, recommendations: List[HardwareRecommendation]):
        """Print hardware recommendations in a formatted way."""
        print(f"\n{'='*80}")
        print("# HARDWARE RECOMMENDATIONS")
        print(f"{'='*80}")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n#{i} {rec.config_name} Configuration")
            print("-" * 40)
            
            # Hardware specs
            hw_info = rec.hardware_config.get_memory_hierarchy_info()
            compute_info = rec.hardware_config.get_compute_info()
            
            print(f"- Performance: {rec.performance_score:.2f}/1.0")
            print(f"- Cost-effectiveness: {rec.cost_score:.2f}/1.0")
            print(f"- Efficiency: {rec.efficiency_score:.2f}/1.0")
            print(f"- {rec.reasoning}")
            
            print(f"\n## Hardware Specifications:")
            print(f"   SRAM: {hw_info['sram_size_mb']:.1f} MB")
            print(f"   DRAM Bandwidth: {hw_info['dram_bandwidth_gb_per_sec']:.1f} GB/s")
            print(f"   Peak Compute: {compute_info['peak_compute_ops_per_sec']/1e12:.1f} TOPS")
            
            if rec.pros:
                print(f"\n### Pros:")
                for pro in rec.pros:
                    print(f"   - {pro}")
            
            if rec.cons:
                print(f"\n### Cons:")
                for con in rec.cons:
                    print(f"   - {con}")
        
        print(f"\n{'='*80}")
        print("> Recommendation: Use the #1 configuration for optimal balance of performance, cost, and efficiency.")
        print(f"{'='*80}")


# Example usage
if __name__ == "__main__":
    recommender = HardwareRecommender()
    
    # Recommend hardware for Qwen3-1.7B
    recommendations = recommender.recommend_hardware(
        "Qwen/Qwen3-1.7B",
        target_tokens_per_second=5.0,
        use_case="general"
    )
    
    recommender.print_recommendations(recommendations)



