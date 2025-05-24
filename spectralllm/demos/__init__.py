"""
SpectralLLM Demos and Benchmarks
================================

Simplified demo and benchmarking functionality for SpectralLLM.
"""

from .complexity_demo import complexity_benchmark, embedding_efficiency_test, quick_benchmark
from .interactive_demo import interactive_demo, quick_demo, show_complexity_comparison, architecture_overview

__all__ = [
    "complexity_benchmark",
    "embedding_efficiency_test", 
    "quick_benchmark",
    "interactive_demo",
    "quick_demo",
    "show_complexity_comparison",
    "architecture_overview"
] 