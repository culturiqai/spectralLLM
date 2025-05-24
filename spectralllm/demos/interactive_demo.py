"""
Interactive SpectralLLM Demo
===========================

Interactive demonstrations and simple usage examples.
"""

import torch
from typing import Optional


def interactive_demo():
    """Run an interactive SpectralLLM demonstration."""
    print("🌊 SpectralLLM Interactive Demo")
    print("=" * 40)
    print("This demo shows basic SpectralLLM functionality")
    print()
    
    from ..core.config import Config
    from ..core.model import SignalLLM as SpectralLLM
    
    # Create a small model for demo
    config = Config(
        vocab_size=1000,
        embed_dim=64,  # Smaller for demo
        num_heads=4,
        num_layers=2,
        harmonic_bases=8
    )
    
    print(f"Creating SpectralLLM model...")
    model = SpectralLLM(config)
    print(f"✅ Model created with {model.count_parameters():,} parameters")
    
    # Show architecture components
    print(f"\n🔧 Architecture Components:")
    if hasattr(model, 'token_embedding'):
        print(f"  ✅ Embedding: {type(model.token_embedding).__name__}")
        if hasattr(model.token_embedding, 'get_mixing_ratio'):
            ratio = model.token_embedding.get_mixing_ratio()
            print(f"     Spectral ratio: {ratio:.3f}")
    
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        block = model.blocks[0]
        print(f"  ✅ Attention: {type(block.wavelet_attn).__name__}")
        print(f"  ✅ FFN: {type(block.ffn).__name__}")
    
    # Test forward pass
    print(f"\n🧪 Testing Forward Pass:")
    test_input = torch.randint(0, 100, (1, 16))  # Small sequence
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"✅ Forward pass successful!")
    
    print(f"\n🎯 Demo complete! SpectralLLM is ready for use.")
    return model


def quick_demo():
    """Quick demonstration without verbose output."""
    from ..core.config import Config
    from ..core.model import SignalLLM as SpectralLLM
    
    # Create and test model
    config = Config(vocab_size=1000, embed_dim=128, num_heads=8)
    model = SpectralLLM(config)
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(torch.randint(0, 100, (2, 16)))
    
    print(f"✅ SpectralLLM working: {model.count_parameters():,} params, output shape {test_output.shape}")
    return model


def show_complexity_comparison():
    """Show theoretical complexity comparison."""
    print("📊 Complexity Comparison: Standard vs SpectralLLM")
    print("=" * 50)
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    print(f"{'Seq Length':<10} {'Standard O(n²)':<15} {'Spectral O(n log n)':<20} {'Speedup':<10}")
    print("-" * 60)
    
    for n in seq_lengths:
        standard_ops = n * n
        spectral_ops = n * (n.bit_length() - 1)  # n * log2(n)
        speedup = standard_ops / spectral_ops
        
        print(f"{n:<10} {standard_ops:<15,} {spectral_ops:<20,} {speedup:<10.2f}x")
    
    print("\n🚀 SpectralLLM provides sub-quadratic complexity scaling!")


def architecture_overview():
    """Show SpectralLLM architecture overview."""
    print("🌊 SpectralLLM Architecture Overview")
    print("=" * 40)
    
    components = [
        ("🔢 SpectralEmbedding", "Frequency-domain token representation"),
        ("🌊 WaveletAttention", "Multi-resolution attention (O(n log n))"),
        ("⚡ FrequencyDomainAttention", "FFT-based attention mechanism"),
        ("🧠 SpectralFeedForward", "Frequency-domain processing"),
        ("🧬 BasisFunction", "Evolutionary basis optimization"),
        ("📊 HybridEmbedding", "Spectral + traditional mixing")
    ]
    
    for name, description in components:
        print(f"  {name:<25} {description}")
    
    print(f"\n✨ Key advantages:")
    print(f"  • O(n log n) complexity vs O(n²) standard attention")
    print(f"  • Parameter efficiency through spectral representation")
    print(f"  • Multi-resolution analysis via wavelets")
    print(f"  • Evolutionary basis function optimization") 