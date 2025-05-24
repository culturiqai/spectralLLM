"""
SpectralLLM Utilities
===================

Utility functions and classes for SpectralLLM training and optimization.
"""

from .tokenizer import SimpleTokenizer

try:
    from .mps_optimizations import optimize_for_mps, setup_mps_optimizations, MPSWaveletTransform, MPSOptimizedAttention
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False

try:
    from .performance import (
        PerformanceProfiler, 
        MemoryTracker, 
        BenchmarkSuite,
        get_profiler,
        profile_function,
        profile_block,
        analyze_memory_patterns,
        time_operation,
        compare_operations
    )
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

__all__ = [
    'SimpleTokenizer'
]

if MPS_AVAILABLE:
    __all__.extend(['optimize_for_mps', 'setup_mps_optimizations', 'MPSWaveletTransform', 'MPSOptimizedAttention'])

if PERFORMANCE_AVAILABLE:
    __all__.extend([
        'PerformanceProfiler', 
        'MemoryTracker', 
        'BenchmarkSuite',
        'get_profiler',
        'profile_function',
        'profile_block',
        'analyze_memory_patterns',
        'time_operation',
        'compare_operations'
    ]) 