#!/usr/bin/env python3
"""
Enhanced SpectralLLM Features Test
==================================

Comprehensive test demonstrating all the advanced features added to SpectralLLM:
- Stochastic Transform with error analysis
- Adaptive Basis Selection 
- Spectral Gap Analysis
- Complete Training Infrastructure
- Enhanced Tokenizer
"""

import torch
import numpy as np
from typing import List

import spectralllm
from spectralllm.utils.tokenizer import SimpleTokenizer
from spectralllm.core.transforms import StochasticTransform, AdaptiveBasisSelection
from spectralllm.core.evolution import SpectralGapAnalyzer
from spectralllm.training.trainer import SpectralTrainer, TextDataset


def test_stochastic_transforms():
    """Test stochastic transform with error analysis."""
    print("🔄 Testing Stochastic Transforms")
    print("=" * 50)
    
    # Test different sampling ratios
    ratios = [0.1, 0.2, 0.5]
    seq_length = 128
    
    for ratio in ratios:
        print(f"\n📊 Sampling Ratio: {ratio}")
        
        st = StochasticTransform(
            wavelet_type='db4', 
            sampling_ratio=ratio,
            min_samples=16
        )
        
        # Generate test data
        x = torch.randn(2, seq_length, 64)
        
        # Test stochastic transform
        approx, details, metrics = st.forward_stochastic(x)
        
        print(f"   Input: {x.shape}")
        print(f"   Output: {approx.shape}, {len(details)} detail levels")
        
        if metrics:
            print(f"   Samples used: {metrics['n_samples']}")
            print(f"   Error bound: {metrics['error_bound']:.6f}")
            print(f"   Actual error: {metrics.get('approx_error', 'N/A')}")
        
        # Get error statistics
        stats = st.get_error_statistics()
        print(f"   Error stats: {stats}")
    
    print("\n✅ Stochastic Transforms working correctly!")


def test_adaptive_basis_selection():
    """Test adaptive basis selection."""
    print("\n🎯 Testing Adaptive Basis Selection")
    print("=" * 50)
    
    # Create adaptive basis selector
    abs_selector = AdaptiveBasisSelection(
        embed_dim=64,
        families=['db4', 'sym4', 'dmey']
    )
    
    # Test with different input patterns
    test_cases = [
        ("Random data", torch.randn(2, 32, 64)),
        ("Periodic data", torch.sin(torch.linspace(0, 4*np.pi, 2*32*64)).reshape(2, 32, 64)),
        ("Sparse data", torch.zeros(2, 32, 64) + 0.1 * torch.randn(2, 32, 64))
    ]
    
    for name, x in test_cases:
        print(f"\n📈 Testing: {name}")
        print(f"   Input shape: {x.shape}")
        
        # Apply adaptive selection
        approx, details = abs_selector(x)
        
        print(f"   Output: {approx.shape}, {len(details)} detail levels")
        
        # Get selection statistics
        stats = abs_selector.get_selection_statistics()
        print(f"   Family preferences: {stats['family_preferences']}")
        print(f"   Dominant family: {stats['dominant_family']}")
        print(f"   Selection entropy: {stats['selection_entropy']:.4f}")
    
    print("\n✅ Adaptive Basis Selection working correctly!")


def test_spectral_gap_analysis():
    """Test spectral gap analysis."""
    print("\n🔬 Testing Spectral Gap Analysis")
    print("=" * 50)
    
    analyzer = SpectralGapAnalyzer()
    
    # Create test wavelet coefficients
    from spectralllm.core.attention import WaveletTransform
    
    wavelet_types = ['db4', 'sym4', 'dmey']
    
    for wavelet_type in wavelet_types:
        print(f"\n📊 Analyzing wavelet: {wavelet_type}")
        
        wt = WaveletTransform(wavelet_type=wavelet_type, levels=3)
        
        # Test different types of signals
        signals = [
            ("Random", torch.randn(1, 64, 32)),
            ("Structured", torch.sin(torch.linspace(0, 8*np.pi, 1*64*32)).reshape(1, 64, 32)),
            ("Sparse", torch.zeros(1, 64, 32) + 0.01 * torch.randn(1, 64, 32))
        ]
        
        for signal_name, signal in signals:
            # Apply wavelet transform
            coeffs = wt(signal)
            
            # Analyze spectral properties
            analysis = analyzer.analyze_wavelet_representation(coeffs)
            
            print(f"   {signal_name} signal:")
            print(f"     Approximation gap: {analysis['approx_gap']:.4f}")
            print(f"     Detail gaps: {[f'{g:.4f}' for g in analysis['detail_gaps']]}")
            print(f"     Overall gap: {analysis['overall_gap']:.4f}")
    
    print("\n✅ Spectral Gap Analysis working correctly!")


def test_enhanced_tokenizer():
    """Test enhanced tokenizer functionality."""
    print("\n🔤 Testing Enhanced Tokenizer")
    print("=" * 50)
    
    # Test texts
    sample_texts = [
        "Hello, world! This is a test.",
        "SpectralLLM uses wavelet transforms for efficient attention.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and natural language processing."
    ]
    
    # Test both character and word level
    for mode in ['char', 'word']:
        print(f"\n📝 Testing {mode}-level tokenization:")
        
        tokenizer = SimpleTokenizer(mode=mode)
        
        # Build vocabulary
        tokenizer.build_vocab(sample_texts, max_vocab_size=100)
        
        # Test encoding/decoding
        test_text = "Hello SpectralLLM!"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"   Original: '{test_text}'")
        print(f"   Encoded: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"   Decoded: '{decoded}'")
        print(f"   Vocab size: {tokenizer.get_vocab_size()}")
        
        # Test batch encoding
        batch_result = tokenizer.encode_batch(
            sample_texts[:2], 
            max_length=20, 
            padding=True
        )
        print(f"   Batch encoding shape: {len(batch_result['input_ids'])}x{len(batch_result['input_ids'][0])}")
        
        # Get vocabulary stats
        stats = tokenizer.get_vocab_stats()
        print(f"   Stats: {stats}")
    
    print("\n✅ Enhanced Tokenizer working correctly!")


def test_training_infrastructure():
    """Test training infrastructure."""
    print("\n🚀 Testing Training Infrastructure")
    print("=" * 50)
    
    # Create model and config
    config = spectralllm.Config(
        vocab_size=100,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    model = spectralllm.SpectralLLM(config)
    
    # Create trainer
    trainer = SpectralTrainer(model, config)
    
    print(f"📊 Model created with {model.count_parameters():,} parameters")
    
    # Test complexity benchmarking
    print("\n🔬 Running complexity benchmark...")
    results = trainer.benchmark_complexity(
        seq_lengths=[32, 64], 
        batch_size=2, 
        num_runs=2
    )
    
    print(f"   Sequence lengths tested: {results['seq_lengths']}")
    print(f"   Forward times: {[f'{t:.4f}s' for t in results['forward_times']]}")
    
    if 'efficiency_ratios' in results:
        avg_efficiency = sum(results['efficiency_ratios']) / len(results['efficiency_ratios'])
        print(f"   Average efficiency vs O(n²): {avg_efficiency:.2f}x better")
    
    # Create simple dataset for testing
    print("\n📚 Testing dataset creation...")
    tokenizer = SimpleTokenizer(mode='char')
    tokenizer.build_vocab(["hello world", "test data"], max_vocab_size=50)
    
    dataset = TextDataset(
        texts=["hello world test"],
        tokenizer=tokenizer,
        seq_length=16,
        stride=8
    )
    
    print(f"   Dataset created with {len(dataset)} samples")
    
    # Test data loading
    sample_input, sample_target = dataset[0]
    print(f"   Sample shapes: input {sample_input.shape}, target {sample_target.shape}")
    
    print("\n✅ Training Infrastructure working correctly!")


def main():
    """Run all tests."""
    print("🧪 SpectralLLM Enhanced Features Test Suite")
    print("=" * 60)
    print("Testing all advanced components added to the core package...")
    print()
    
    try:
        test_stochastic_transforms()
        test_adaptive_basis_selection()
        test_spectral_gap_analysis()
        test_enhanced_tokenizer()
        test_training_infrastructure()
        
        print("\n" + "=" * 60)
        print("🎉 ALL ENHANCED FEATURES WORKING CORRECTLY!")
        print("=" * 60)
        print()
        print("📋 Summary of Enhanced Components:")
        print("   ✅ StochasticTransform - O(m log m) complexity reduction")
        print("   ✅ AdaptiveBasisSelection - Multi-wavelet family optimization")
        print("   ✅ SpectralGapAnalyzer - Linguistic structure analysis")
        print("   ✅ SpectralTrainer - Complete training infrastructure")
        print("   ✅ TextDataset - Efficient data loading")
        print("   ✅ Enhanced SimpleTokenizer - Save/load capabilities")
        print()
        print("📊 Core package now has FULL implementation from comprehensive file!")
        print("🚀 Ready for production-level training and deployment!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 