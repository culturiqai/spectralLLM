#!/usr/bin/env python3
"""
Performance Profiling and Analysis for SpectralLLM
=================================================

Comprehensive performance monitoring, profiling, and analysis tools.
Includes memory tracking, timing analysis, device utilization monitoring,
and detailed performance reports.

Based on the reference implementation with all production features.
"""

import torch
import time
import psutil
import gc
import os
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from functools import wraps
import math
import threading
from datetime import datetime
import warnings

# Try to import additional profiling tools
try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False

try:
    import nvidia_ml_py3 as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class PerformanceProfiler:
    """
    Comprehensive performance profiler for SpectralLLM operations.
    
    Features:
    - Memory usage tracking (CUDA/MPS/CPU)
    - Execution time monitoring
    - Device utilization analysis  
    - Custom metric collection
    - Report generation
    """
    
    def __init__(self, enable_memory_tracking: bool = True, 
                 enable_gpu_tracking: bool = True,
                 max_history: int = 1000):
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_gpu_tracking = enable_gpu_tracking
        self.max_history = max_history
        
        # Storage for metrics
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.gpu_data = defaultdict(list)
        self.custom_metrics = defaultdict(list)
        
        # Recent history with efficient circular buffer
        self.recent_timings = deque(maxlen=max_history)
        self.recent_memory = deque(maxlen=max_history)
        
        # Session tracking
        self.session_start = time.time()
        self.operation_stack = []
        
        # Threading for continuous monitoring
        self._monitoring = False
        self._monitor_thread = None
        
        # Initialize GPU monitoring if available
        self._init_gpu_monitoring()
        
        # Initialize memory tracking
        if self.enable_memory_tracking and HAS_TRACEMALLOC:
            tracemalloc.start()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring based on available hardware"""
        self.gpu_type = None
        self.gpu_available = False
        
        # Check for CUDA
        if torch.cuda.is_available():
            self.gpu_type = 'cuda'
            self.gpu_available = True
            if HAS_NVML:
                try:
                    nvml.nvmlInit()
                    self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                except Exception as e:
                    warnings.warn(f"Could not initialize NVML: {e}")
        
        # Check for MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            self.gpu_type = 'mps'
            self.gpu_available = True
        
        # CPU only
        else:
            self.gpu_type = 'cpu'
            self.gpu_available = False
    
    @contextmanager
    def profile_operation(self, operation_name: str, **kwargs):
        """
        Context manager for profiling operations with comprehensive metrics
        
        Args:
            operation_name: Name of the operation being profiled
            **kwargs: Additional metadata to store with the profile
        """
        # Record operation start
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        start_gpu = self._get_gpu_utilization()
        
        # Push to operation stack
        self.operation_stack.append({
            'name': operation_name,
            'start_time': start_time,
            'start_memory': start_memory,
            'metadata': kwargs
        })
        
        try:
            yield self
        finally:
            # Record operation end
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_gpu = self._get_gpu_utilization()
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = {}
            for key in start_memory:
                memory_delta[key] = end_memory.get(key, 0) - start_memory.get(key, 0)
            
            # Store results
            profile_data = {
                'operation': operation_name,
                'duration': duration,
                'memory_delta': memory_delta,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'gpu_utilization': end_gpu,
                'timestamp': time.time(),
                'metadata': kwargs
            }
            
            # Add to collections
            self.timing_data[operation_name].append(duration)
            self.memory_data[operation_name].append(memory_delta)
            self.recent_timings.append(profile_data)
            
            # Pop from operation stack
            if self.operation_stack:
                self.operation_stack.pop()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive memory usage statistics"""
        memory_stats = {}
        
        # CPU memory
        process = psutil.Process()
        memory_stats['cpu_rss'] = process.memory_info().rss / 1024 / 1024  # MB
        memory_stats['cpu_vms'] = process.memory_info().vms / 1024 / 1024  # MB
        memory_stats['cpu_percent'] = process.memory_percent()
        
        # Python memory tracking
        if HAS_TRACEMALLOC and self.enable_memory_tracking:
            try:
                current, peak = tracemalloc.get_traced_memory()
                memory_stats['python_current'] = current / 1024 / 1024  # MB
                memory_stats['python_peak'] = peak / 1024 / 1024  # MB
            except Exception:
                pass
        
        # GPU memory
        if self.gpu_available and self.enable_gpu_tracking:
            if self.gpu_type == 'cuda':
                try:
                    memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_stats['gpu_cached'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                except Exception:
                    pass
            elif self.gpu_type == 'mps':
                try:
                    memory_stats['mps_allocated'] = torch.mps.current_allocated_memory() / 1024 / 1024  # MB
                except Exception:
                    pass
        
        return memory_stats
    
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization statistics"""
        gpu_stats = {}
        
        if not self.gpu_available or not self.enable_gpu_tracking:
            return gpu_stats
        
        if self.gpu_type == 'cuda' and HAS_NVML:
            try:
                # Get utilization
                util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_stats['gpu_utilization'] = util.gpu
                gpu_stats['memory_utilization'] = util.memory
                
                # Get temperature
                temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                gpu_stats['temperature'] = temp
                
                # Get power
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
                    gpu_stats['power_usage'] = power
                except Exception:
                    pass
                    
            except Exception as e:
                # NVML calls can fail, continue gracefully
                warnings.warn(f"GPU monitoring failed: {e}")
        
        elif self.gpu_type == 'mps':
            # MPS doesn't have direct utilization APIs
            # Use memory allocation as a proxy
            try:
                allocated = torch.mps.current_allocated_memory()
                gpu_stats['mps_allocated'] = allocated / 1024 / 1024  # MB
            except Exception:
                pass
        
        return gpu_stats
    
    def add_custom_metric(self, metric_name: str, value: float, **metadata):
        """Add a custom metric to the profiler"""
        self.custom_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time(),
            'metadata': metadata
        })
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring in background thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    # Collect system metrics
                    memory_usage = self._get_memory_usage()
                    gpu_usage = self._get_gpu_utilization()
                    
                    # Store in recent history
                    self.recent_memory.append({
                        'timestamp': time.time(),
                        'memory': memory_usage,
                        'gpu': gpu_usage
                    })
                    
                    time.sleep(interval)
                except Exception as e:
                    warnings.warn(f"Monitoring thread error: {e}")
                    time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a specific operation"""
        if operation_name not in self.timing_data:
            return {}
        
        timings = self.timing_data[operation_name]
        
        if not timings:
            return {}
        
        stats = {
            'count': len(timings),
            'total_time': sum(timings),
            'mean_time': sum(timings) / len(timings),
            'min_time': min(timings),
            'max_time': max(timings),
            'median_time': sorted(timings)[len(timings) // 2],
        }
        
        # Calculate standard deviation
        mean = stats['mean_time']
        variance = sum((t - mean) ** 2 for t in timings) / len(timings)
        stats['std_time'] = math.sqrt(variance)
        
        # Percentiles
        sorted_timings = sorted(timings)
        stats['p95_time'] = sorted_timings[int(0.95 * len(sorted_timings))]
        stats['p99_time'] = sorted_timings[int(0.99 * len(sorted_timings))]
        
        # Memory statistics if available
        if operation_name in self.memory_data:
            memory_deltas = self.memory_data[operation_name]
            if memory_deltas:
                # Calculate memory statistics for each memory type
                memory_stats = {}
                for memory_type in memory_deltas[0].keys():
                    values = [delta.get(memory_type, 0) for delta in memory_deltas]
                    memory_stats[f'{memory_type}_mean'] = sum(values) / len(values)
                    memory_stats[f'{memory_type}_max'] = max(values)
                    memory_stats[f'{memory_type}_min'] = min(values)
                
                stats['memory_stats'] = memory_stats
        
        return stats
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary report"""
        report = {
            'session_info': {
                'start_time': self.session_start,
                'duration': time.time() - self.session_start,
                'gpu_type': self.gpu_type,
                'gpu_available': self.gpu_available,
            },
            'operations': {},
            'system_info': self._get_system_info(),
            'peak_memory': self._get_peak_memory_usage(),
        }
        
        # Add operation statistics
        for operation_name in self.timing_data:
            report['operations'][operation_name] = self.get_operation_stats(operation_name)
        
        # Add custom metrics summary
        if self.custom_metrics:
            report['custom_metrics'] = {}
            for metric_name, data in self.custom_metrics.items():
                values = [item['value'] for item in data]
                if values:
                    report['custom_metrics'][metric_name] = {
                        'count': len(values),
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1]
                    }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'available_memory': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'total_memory': psutil.virtual_memory().total / 1024 / 1024,  # MB
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'torch_version': torch.__version__,
        }
    
    def _get_peak_memory_usage(self) -> Dict[str, float]:
        """Get peak memory usage across all operations"""
        peak_memory = {}
        
        for memory_data in self.recent_memory:
            if 'memory' in memory_data:
                for key, value in memory_data['memory'].items():
                    if key not in peak_memory or value > peak_memory[key]:
                        peak_memory[key] = value
        
        return peak_memory
    
    def save_report(self, filepath: str, include_raw_data: bool = False):
        """Save performance report to JSON file"""
        report = self.get_summary_report()
        
        if include_raw_data:
            report['raw_data'] = {
                'timing_data': dict(self.timing_data),
                'memory_data': [dict(m) for m in self.recent_memory],
                'custom_metrics': dict(self.custom_metrics)
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.timing_data.clear()
        self.memory_data.clear()
        self.gpu_data.clear()
        self.custom_metrics.clear()
        self.recent_timings.clear()
        self.recent_memory.clear()
        self.session_start = time.time()
        
        # Reset GPU memory tracking
        if self.gpu_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_continuous_monitoring()


# Global profiler instance
_global_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_function(operation_name: Optional[str] = None, include_memory: bool = True):
    """
    Decorator for profiling function execution
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        include_memory: Whether to track memory usage
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with profiler.profile_operation(op_name, include_memory=include_memory):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def profile_block(operation_name: str, **metadata):
    """Context manager for profiling code blocks"""
    profiler = get_profiler()
    with profiler.profile_operation(operation_name, **metadata):
        yield


class MemoryTracker:
    """Dedicated memory tracking utility for detailed analysis"""
    
    def __init__(self):
        self.checkpoints = {}
        self.peak_usage = {}
        
    def checkpoint(self, name: str):
        """Create a memory checkpoint"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure operations are complete
        
        memory_info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info['cpu'] = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        elif torch.backends.mps.is_available():
            try:
                memory_info['mps_allocated'] = torch.mps.current_allocated_memory() / 1024 / 1024  # MB
            except:
                pass
        
        self.checkpoints[name] = memory_info
        
        # Update peak usage
        for key, value in memory_info.items():
            if key not in self.peak_usage or value > self.peak_usage[key]:
                self.peak_usage[key] = value
    
    def get_delta(self, start_checkpoint: str, end_checkpoint: str) -> Dict[str, float]:
        """Get memory delta between two checkpoints"""
        if start_checkpoint not in self.checkpoints or end_checkpoint not in self.checkpoints:
            return {}
        
        start = self.checkpoints[start_checkpoint]
        end = self.checkpoints[end_checkpoint]
        
        delta = {}
        for key in start:
            if key in end:
                delta[key] = end[key] - start[key]
        
        return delta
    
    def print_summary(self):
        """Print memory usage summary"""
        print("\n=== Memory Usage Summary ===")
        print(f"Peak Usage:")
        for key, value in self.peak_usage.items():
            print(f"  {key}: {value:.2f} MB")
        
        print(f"\nCheckpoints ({len(self.checkpoints)}):")
        for name, info in self.checkpoints.items():
            print(f"  {name}:")
            for key, value in info.items():
                print(f"    {key}: {value:.2f} MB")


class BenchmarkSuite:
    """Comprehensive benchmarking suite for SpectralLLM components"""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or get_profiler()
        self.results = {}
    
    def benchmark_operation(self, operation_func: Callable, 
                          operation_name: str,
                          num_runs: int = 10,
                          warmup_runs: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """
        Benchmark a specific operation with multiple runs
        
        Args:
            operation_func: Function to benchmark
            operation_name: Name for the benchmark
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not counted)
            **kwargs: Arguments to pass to the operation function
        """
        print(f"Benchmarking {operation_name}...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                operation_func(**kwargs)
            except Exception as e:
                print(f"Warmup failed: {e}")
                return {}
        
        # Clear memory and reset stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Benchmark runs
        timings = []
        memory_usage = []
        
        for run in range(num_runs):
            memory_before = self.profiler._get_memory_usage()
            
            start_time = time.perf_counter()
            try:
                result = operation_func(**kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Benchmark run {run} failed: {e}")
                continue
            end_time = time.perf_counter()
            
            memory_after = self.profiler._get_memory_usage()
            
            timings.append(end_time - start_time)
            
            # Calculate memory delta
            memory_delta = {}
            for key in memory_before:
                memory_delta[key] = memory_after.get(key, 0) - memory_before.get(key, 0)
            memory_usage.append(memory_delta)
        
        if not timings:
            return {}
        
        # Calculate statistics
        benchmark_results = {
            'operation': operation_name,
            'num_runs': len(timings),
            'mean_time': sum(timings) / len(timings),
            'min_time': min(timings),
            'max_time': max(timings),
            'std_time': math.sqrt(sum((t - sum(timings)/len(timings))**2 for t in timings) / len(timings)),
            'throughput': 1.0 / (sum(timings) / len(timings)),  # ops/second
        }
        
        # Memory statistics
        if memory_usage:
            memory_stats = {}
            for memory_type in memory_usage[0].keys():
                values = [usage.get(memory_type, 0) for usage in memory_usage]
                memory_stats[f'{memory_type}_mean'] = sum(values) / len(values)
                memory_stats[f'{memory_type}_max'] = max(values)
                memory_stats[f'{memory_type}_min'] = min(values)
            
            benchmark_results['memory_stats'] = memory_stats
        
        self.results[operation_name] = benchmark_results
        return benchmark_results
    
    def run_model_benchmark(self, model, input_tensor: torch.Tensor, 
                          batch_sizes: List[int] = [1, 4, 8],
                          sequence_lengths: List[int] = [128, 512, 1024]) -> Dict[str, Any]:
        """Run comprehensive model benchmarks across different configurations"""
        
        print("Running comprehensive model benchmark...")
        benchmark_results = {}
        
        original_batch_size = input_tensor.size(0)
        original_seq_length = input_tensor.size(1)
        
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                config_name = f"batch_{batch_size}_seq_{seq_length}"
                print(f"  Testing {config_name}...")
                
                # Create appropriately sized input
                test_input = torch.randn(batch_size, seq_length, input_tensor.size(-1), 
                                       device=input_tensor.device, dtype=input_tensor.dtype)
                
                def model_forward():
                    with torch.no_grad():
                        return model(test_input)
                
                results = self.benchmark_operation(
                    model_forward, 
                    config_name,
                    num_runs=5,
                    warmup_runs=2
                )
                
                if results:
                    benchmark_results[config_name] = results
        
        return benchmark_results
    
    def print_results(self):
        """Print formatted benchmark results"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        for operation_name, results in self.results.items():
            print(f"\n{operation_name}:")
            print(f"  Runs: {results['num_runs']}")
            print(f"  Mean time: {results['mean_time']*1000:.2f} ms")
            print(f"  Min time: {results['min_time']*1000:.2f} ms")
            print(f"  Max time: {results['max_time']*1000:.2f} ms")
            print(f"  Std dev: {results['std_time']*1000:.2f} ms")
            print(f"  Throughput: {results['throughput']:.2f} ops/sec")
            
            if 'memory_stats' in results:
                print(f"  Memory usage:")
                for key, value in results['memory_stats'].items():
                    print(f"    {key}: {value:.2f} MB")


def analyze_memory_patterns(profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Analyze memory usage patterns and identify potential issues"""
    
    analysis = {
        'memory_leaks': [],
        'high_usage_operations': [],
        'recommendations': []
    }
    
    # Check for operations with high memory usage
    for operation_name in profiler.timing_data:
        stats = profiler.get_operation_stats(operation_name)
        if 'memory_stats' in stats:
            # Check for operations using more than 500MB
            for memory_type, value in stats['memory_stats'].items():
                if 'mean' in memory_type and value > 500:
                    analysis['high_usage_operations'].append({
                        'operation': operation_name,
                        'memory_type': memory_type,
                        'usage_mb': value
                    })
    
    # Check for potential memory leaks in recent history
    if len(profiler.recent_memory) > 10:
        # Look for consistently increasing memory usage
        recent_cpu_usage = []
        for item in list(profiler.recent_memory)[-10:]:
            if 'memory' in item and 'cpu_rss' in item['memory']:
                recent_cpu_usage.append(item['memory']['cpu_rss'])
        
        if len(recent_cpu_usage) >= 5:
            # Simple trend analysis
            x = list(range(len(recent_cpu_usage)))
            y = recent_cpu_usage
            
            # Calculate slope (simple linear regression)
            n = len(x)
            slope = (n * sum(x[i]*y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
            
            if slope > 5:  # Memory increasing by more than 5MB per measurement
                analysis['memory_leaks'].append({
                    'type': 'cpu_memory_trend',
                    'slope_mb_per_measurement': slope,
                    'description': 'CPU memory usage is consistently increasing'
                })
    
    # Generate recommendations
    if analysis['high_usage_operations']:
        analysis['recommendations'].append(
            "Consider optimizing high memory usage operations or implementing memory checkpointing"
        )
    
    if analysis['memory_leaks']:
        analysis['recommendations'].append(
            "Potential memory leaks detected. Review object lifecycle and garbage collection"
        )
    
    return analysis


# Convenience functions for common profiling tasks
def time_operation(func: Callable, *args, **kwargs) -> tuple:
    """
    Time a single operation and return (result, duration)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def compare_operations(operations: Dict[str, Callable], *args, **kwargs) -> Dict[str, float]:
    """
    Compare execution times of multiple operations
    
    Args:
        operations: Dict mapping operation names to callable functions
        *args, **kwargs: Arguments to pass to each operation
    
    Returns:
        Dict mapping operation names to execution times
    """
    results = {}
    
    for name, operation in operations.items():
        _, duration = time_operation(operation, *args, **kwargs)
        results[name] = duration
    
    return results


# Export key classes and functions
__all__ = [
    'PerformanceProfiler',
    'MemoryTracker', 
    'BenchmarkSuite',
    'get_profiler',
    'profile_function',
    'profile_block',
    'analyze_memory_patterns',
    'time_operation',
    'compare_operations'
] 