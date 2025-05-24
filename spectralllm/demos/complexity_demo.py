"""
Complexity Benchmarking for SpectralLLM
=======================================

Simplified benchmarking to demonstrate computational complexity advantages.
"""

import time
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def complexity_benchmark(seq_lengths: List[int] = None, 
                        embed_dim: int = 128, 
                        num_heads: int = 8,
                        device: Optional[torch.device] = None) -> Dict:
    """
    Benchmark complexity comparison between standard and spectral attention.
    
    Args:
        seq_lengths: List of sequence lengths to test
        embed_dim: Embedding dimension  
        num_heads: Number of attention heads
        device: Device to run on
        
    Returns:
        Dictionary with timing results
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔬 Complexity Benchmark on {device}")
    print(f"Testing sequence lengths: {seq_lengths}")
    
    from ..core.attention import FrequencyDomainAttention
    
    # Create attention mechanisms
    std_attn = nn.MultiheadAttention(embed_dim, num_heads).to(device)
    freq_attn = FrequencyDomainAttention(embed_dim, num_heads).to(device)
    
    results = {
        'seq_lengths': seq_lengths,
        'standard_times': [],
        'frequency_times': [],
        'speedup_ratios': []
    }
    
    batch_size = 4
    num_runs = 5
    
    for seq_len in seq_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Create test data
        x_std = torch.randn(seq_len, batch_size, embed_dim, device=device)
        x_freq = x_std.transpose(0, 1)  # [batch, seq, embed]
        
        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = std_attn(x_std, x_std, x_std)
                _ = freq_attn(x_freq)
        
        # Benchmark standard attention
        std_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                _ = std_attn(x_std, x_std, x_std)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            std_times.append(time.time() - start)
        
        avg_std_time = sum(std_times) / len(std_times)
        results['standard_times'].append(avg_std_time)
        print(f"  Standard attention: {avg_std_time:.4f}s")
        
        # Benchmark frequency attention
        freq_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                _ = freq_attn(x_freq)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            freq_times.append(time.time() - start)
        
        avg_freq_time = sum(freq_times) / len(freq_times)
        results['frequency_times'].append(avg_freq_time)
        print(f"  Frequency attention: {avg_freq_time:.4f}s")
        
        speedup = avg_std_time / avg_freq_time
        results['speedup_ratios'].append(speedup)
        print(f"  Speedup: {speedup:.2f}x")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Average speedup: {sum(results['speedup_ratios'])/len(results['speedup_ratios']):.2f}x")
    print(f"  Best speedup: {max(results['speedup_ratios']):.2f}x (at seq_len={seq_lengths[results['speedup_ratios'].index(max(results['speedup_ratios']))]})")
    
    return results


def embedding_efficiency_test(vocab_sizes: List[int] = None,
                            embed_dim: int = 128,
                            harmonic_bases: int = 16) -> Dict:
    """
    Test parameter efficiency of spectral vs standard embeddings.
    
    Args:
        vocab_sizes: List of vocabulary sizes to test
        embed_dim: Embedding dimension
        harmonic_bases: Number of harmonic bases for spectral embedding
        
    Returns:
        Dictionary with parameter counts and efficiency ratios
    """
    if vocab_sizes is None:
        vocab_sizes = [1000, 5000, 10000, 50000]
    
    print(f"📈 Embedding Efficiency Test")
    print(f"Embed dim: {embed_dim}, Harmonic bases: {harmonic_bases}")
    
    from ..core.embeddings import SpectralEmbedding
    
    results = {
        'vocab_sizes': vocab_sizes,
        'standard_params': [],
        'spectral_params': [],
        'efficiency_ratios': []
    }
    
    for vocab_size in vocab_sizes:
        print(f"\n📚 Vocabulary size: {vocab_size:,}")
        
        # Standard embedding parameters
        std_params = vocab_size * embed_dim
        results['standard_params'].append(std_params)
        print(f"  Standard: {std_params:,} parameters")
        
        # Spectral embedding parameters
        spectral_embed = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases)
        spec_params = sum(p.numel() for p in spectral_embed.parameters())
        results['spectral_params'].append(spec_params)
        print(f"  Spectral: {spec_params:,} parameters")
        
        # Efficiency ratio
        ratio = std_params / spec_params
        results['efficiency_ratios'].append(ratio)
        print(f"  Efficiency: {ratio:.2f}x parameter reduction")
    
    # Summary
    print(f"\n📊 Summary:")
    avg_efficiency = sum(results['efficiency_ratios']) / len(results['efficiency_ratios'])
    print(f"  Average efficiency: {avg_efficiency:.2f}x parameter reduction")
    print(f"  Best efficiency: {max(results['efficiency_ratios']):.2f}x")
    
    return results


def plot_results(complexity_results: Dict = None, 
                efficiency_results: Dict = None,
                save_path: str = None):
    """
    Plot benchmark results if matplotlib is available.
    
    Args:
        complexity_results: Results from complexity_benchmark()
        efficiency_results: Results from embedding_efficiency_test()
        save_path: Path to save plots
    """
    if not PLOTTING_AVAILABLE:
        print("📊 Matplotlib not available - cannot create plots")
        return
    
    fig, axes = plt.subplots(1, 2 if efficiency_results else 1, figsize=(12, 5))
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    
    # Plot complexity results
    if complexity_results:
        ax = axes[0]
        seq_lengths = complexity_results['seq_lengths']
        std_times = complexity_results['standard_times']
        freq_times = complexity_results['frequency_times']
        
        ax.plot(seq_lengths, std_times, 'bo-', label='Standard O(n²)')
        ax.plot(seq_lengths, freq_times, 'ro-', label='Spectral O(n log n)')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Attention Complexity Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Plot efficiency results
    if efficiency_results and len(axes) > 1:
        ax = axes[1]
        vocab_sizes = efficiency_results['vocab_sizes']
        ratios = efficiency_results['efficiency_ratios']
        
        ax.plot(vocab_sizes, ratios, 'go-', label='Parameter Efficiency')
        ax.set_xlabel('Vocabulary Size')
        ax.set_ylabel('Efficiency Ratio (x)')
        ax.set_title('Embedding Parameter Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()


def quick_benchmark():
    """Run a quick benchmark demonstration."""
    print("🚀 Quick SpectralLLM Benchmark")
    print("=" * 40)
    
    # Quick complexity test
    complexity_results = complexity_benchmark(
        seq_lengths=[128, 256, 512],
        embed_dim=64,
        num_heads=4
    )
    
    # Quick efficiency test  
    efficiency_results = embedding_efficiency_test(
        vocab_sizes=[1000, 10000],
        embed_dim=64,
        harmonic_bases=16
    )
    
    print("\n🎯 Quick Benchmark Complete!")
    return complexity_results, efficiency_results 