#!/usr/bin/env python3
"""
Print Matrix Sizes for LLM Models

This script prints detailed matrix sizes and memory requirements for LLM models.

Usage:
    python examples/print_matrix_sizes.py <model_name>
    
Examples:
    python examples/print_matrix_sizes.py Qwen/Qwen3-8B-FP8
    python examples/print_matrix_sizes.py Qwen/Qwen3-1.7B
    python examples/print_matrix_sizes.py meta-llama/Llama-3.1-8B
"""

import sys
import os
import argparse

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_sim.mma_analyzer import MMAAnalyzer


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Print matrix sizes for LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/print_matrix_sizes.py Qwen/Qwen3-8B-FP8
  python examples/print_matrix_sizes.py Qwen/Qwen3-1.7B
  python examples/print_matrix_sizes.py meta-llama/Llama-3.1-8B
        """
    )
    
    parser.add_argument(
        "model_name", 
        help="Hugging Face model identifier (e.g., Qwen/Qwen3-8B-FP8)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer and print matrix sizes
    analyzer = MMAAnalyzer()
    analyzer.print_model_matrix_sizes(args.model_name)


if __name__ == "__main__":
    main()
