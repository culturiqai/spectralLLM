#!/usr/bin/env python3
"""
Peak Spectral Analysis - Evaluating "Up to 98%" Claims
Examining different ways to honestly frame spectral computation percentages
"""

import torch
import numpy as np

def analyze_peak_spectral_claims():
    """Examine different ways to present spectral computation percentages"""
    
    print("🔬 PEAK SPECTRAL COMPUTATION ANALYSIS")
    print("=" * 60)
    
    # Load model for analysis
    try:
        model_state = torch.load('../outputs/signalllm_8m_20250523/latest_checkpoint.pt', map_location='cpu')['model']
        print("✅ Loaded trained model checkpoint")
    except:
        print("❌ Could not load checkpoint, using simulated data")
        return
    
    print("\n📊 COMPONENT-WISE SPECTRAL ANALYSIS:")
    
    # Analyze each component's peak spectral computation
    components = analyze_component_spectral_peaks(model_state)
    
    print("\n🎯 DIFFERENT WAYS TO FRAME 'UP TO X% SPECTRAL':")
    analyze_framing_approaches(components)
    
    print("\n💭 HONEST vs MISLEADING FRAMING:")
    discuss_framing_ethics(components)

def analyze_component_spectral_peaks(model_state):
    """Analyze peak spectral computation for each component"""
    
    components = {}
    
    # 1. Embedding Analysis
    print("\n  🌊 EMBEDDING LAYER ANALYSIS:")
    embedding_spectral = analyze_embedding_spectral_peak(model_state)
    components['embeddings'] = embedding_spectral
    
    # 2. Attention Block Analysis  
    print("\n  🌊 ATTENTION BLOCK ANALYSIS:")
    attention_spectral = analyze_attention_spectral_peak(model_state)
    components['attention'] = attention_spectral
    
    # 3. FFN Block Analysis
    print("\n  🌊 FFN BLOCK ANALYSIS:")
    ffn_spectral = analyze_ffn_spectral_peak(model_state)
    components['ffn'] = ffn_spectral
    
    # 4. Overall Analysis
    print("\n  📈 OVERALL MODEL ANALYSIS:")
    overall_spectral = calculate_weighted_spectral(components)
    components['overall'] = overall_spectral
    
    return components

def analyze_embedding_spectral_peak(model_state):
    """Detailed embedding spectral analysis"""
    
    # Get embedding components
    freq_amps = model_state['token_embedding.spectral_embedding.frequency_amplitudes']
    vocab_size, harmonic_bases = freq_amps.shape
    embed_dim = 128  # From model architecture
    
    # Simulate computation for typical batch
    batch_size, seq_len = 4, 32
    
    # Traditional path: Just memory lookup (minimal FLOPs)
    traditional_flops = batch_size * seq_len * 1  # Just indexing
    
    # Spectral path: Harmonic synthesis
    spectral_flops = batch_size * seq_len * embed_dim * harmonic_bases * 3  # cos + multiply + sum
    
    # Mixing: weighted combination
    mixing_flops = batch_size * seq_len * embed_dim * 2
    
    total_flops = traditional_flops + spectral_flops + mixing_flops
    effective_spectral = spectral_flops + mixing_flops * 0.7  # Assume 70% spectral mixing
    
    spectral_ratio = effective_spectral / total_flops
    
    print(f"    Traditional FLOPs: {traditional_flops:,}")
    print(f"    Spectral FLOPs: {spectral_flops:,}")
    print(f"    Mixing FLOPs: {mixing_flops:,}")
    print(f"    Total FLOPs: {total_flops:,}")
    print(f"    ⚡ Spectral Ratio: {spectral_ratio:.1%}")
    
    return {
        'ratio': spectral_ratio,
        'traditional_flops': traditional_flops,
        'spectral_flops': effective_spectral,
        'total_flops': total_flops
    }

def analyze_attention_spectral_peak(model_state):
    """Analyze attention block spectral computation"""
    
    # Count attention parameters
    attn_params = sum(
        tensor.numel() for key, tensor in model_state.items()
        if 'attn' in key and 'blocks' in key
    )
    
    # Estimate attention computation
    # Attention involves Q/K/V projections + attention computation + output projection
    # In SpectralLLM, the FFT operations in attention dominate
    
    seq_len = 32
    embed_dim = 128
    
    # Traditional attention FLOPs
    traditional_attn_flops = seq_len * embed_dim * embed_dim * 4  # Q,K,V,O projections
    
    # Spectral attention FLOPs (FFT + frequency domain operations)
    fft_flops = seq_len * embed_dim * np.log2(embed_dim) * 2  # FFT + IFFT
    freq_attn_flops = seq_len * seq_len * embed_dim  # Frequency domain attention
    
    total_attn_flops = traditional_attn_flops + fft_flops + freq_attn_flops
    spectral_attn_flops = fft_flops + freq_attn_flops
    
    spectral_ratio = spectral_attn_flops / total_attn_flops
    
    print(f"    Traditional attention FLOPs: {traditional_attn_flops:,}")
    print(f"    FFT FLOPs: {fft_flops:,}")
    print(f"    Frequency attention FLOPs: {freq_attn_flops:,}")
    print(f"    ⚡ Attention Spectral Ratio: {spectral_ratio:.1%}")
    
    return {
        'ratio': spectral_ratio,
        'traditional_flops': traditional_attn_flops,
        'spectral_flops': spectral_attn_flops,
        'total_flops': total_attn_flops
    }

def analyze_ffn_spectral_peak(model_state):
    """Analyze FFN spectral computation"""
    
    # FFN has both traditional and spectral paths
    embed_dim = 128
    ffn_dim = embed_dim * 4  # Typical transformer scaling
    
    # Traditional FFN path
    traditional_ffn_flops = embed_dim * ffn_dim * 2  # fc1 + fc2
    
    # Spectral FFN path (frequency mixing)
    freq_ffn_flops = embed_dim * np.log2(embed_dim) * 2  # FFT + IFFT
    freq_linear_flops = embed_dim * embed_dim  # Frequency domain linear
    
    total_ffn_flops = traditional_ffn_flops + freq_ffn_flops + freq_linear_flops
    spectral_ffn_flops = freq_ffn_flops + freq_linear_flops
    
    spectral_ratio = spectral_ffn_flops / total_ffn_flops
    
    print(f"    Traditional FFN FLOPs: {traditional_ffn_flops:,}")
    print(f"    Spectral FFN FLOPs: {spectral_ffn_flops:,}")
    print(f"    ⚡ FFN Spectral Ratio: {spectral_ratio:.1%}")
    
    return {
        'ratio': spectral_ratio,
        'traditional_flops': traditional_ffn_flops,
        'spectral_flops': spectral_ffn_flops,
        'total_flops': total_ffn_flops
    }

def calculate_weighted_spectral(components):
    """Calculate overall weighted spectral ratio"""
    
    # Weight components by their typical computational load
    weights = {
        'embeddings': 0.15,   # Embeddings are computed once per token
        'attention': 0.60,    # Attention dominates transformer computation
        'ffn': 0.25          # FFN is significant but less than attention
    }
    
    total_flops = sum(
        weights[comp] * components[comp]['total_flops'] 
        for comp in weights.keys()
    )
    
    total_spectral_flops = sum(
        weights[comp] * components[comp]['spectral_flops']
        for comp in weights.keys()
    )
    
    overall_ratio = total_spectral_flops / total_flops
    
    print(f"    Weighted total FLOPs: {total_flops:,.0f}")
    print(f"    Weighted spectral FLOPs: {total_spectral_flops:,.0f}")
    print(f"    ⚡ Overall Spectral Ratio: {overall_ratio:.1%}")
    
    return {
        'ratio': overall_ratio,
        'total_flops': total_flops,
        'spectral_flops': total_spectral_flops
    }

def analyze_framing_approaches(components):
    """Analyze different ways to frame the spectral percentage"""
    
    print("  1️⃣ PEAK-FOCUSED FRAMING:")
    max_ratio = max(comp['ratio'] for comp in components.values())
    print(f"     'Up to {max_ratio:.1%} spectral computing'")
    print(f"     ✅ Technically accurate")
    print(f"     ⚠️  Could be misleading without context")
    
    print("\n  2️⃣ COMPONENT-SPECIFIC FRAMING:")
    for name, comp in components.items():
        print(f"     '{name.title()}: {comp['ratio']:.1%} spectral'")
    print(f"     ✅ Provides full breakdown")
    print(f"     ✅ Allows informed evaluation")
    
    print("\n  3️⃣ RANGE FRAMING:")
    min_ratio = min(comp['ratio'] for comp in components.values() if comp['ratio'] > 0)
    max_ratio = max(comp['ratio'] for comp in components.values())
    print(f"     '{min_ratio:.1%} to {max_ratio:.1%} spectral across components'")
    print(f"     ✅ Shows full spectrum")
    print(f"     ✅ Honest about variation")
    
    print("\n  4️⃣ WEIGHTED AVERAGE FRAMING:")
    overall_ratio = components['overall']['ratio']
    print(f"     '{overall_ratio:.1%} spectral overall (up to {max_ratio:.1%} peak)'")
    print(f"     ✅ Balanced and honest")
    print(f"     ✅ Highlights innovation while providing context")

def discuss_framing_ethics(components):
    """Discuss ethical considerations in framing claims"""
    
    max_ratio = max(comp['ratio'] for comp in components.values())
    overall_ratio = components['overall']['ratio']
    
    print("  🎭 MARKETING vs HONESTY ANALYSIS:")
    
    print(f"\n  📢 MARKETING TEMPTATION:")
    print(f"     Claim: 'Up to {max_ratio:.1%} spectral computing'")
    print(f"     Appeal: Sounds impressive, technically accurate")
    print(f"     Risk: Misleads about overall architecture")
    
    print(f"\n  ✅ HONEST ALTERNATIVE:")
    print(f"     Claim: '{overall_ratio:.1%} spectral overall, up to {max_ratio:.1%} in embeddings'")
    print(f"     Appeal: Shows genuine innovation with context")
    print(f"     Benefit: Builds trust, enables informed decisions")
    
    print(f"\n  🎯 RECOMMENDATION:")
    if max_ratio >= 0.9:
        print(f"     'Up to {max_ratio:.1%}' is acceptable IF:")
        print(f"     - Component is specified ('in embeddings')")
        print(f"     - Overall ratio is also provided")
        print(f"     - Context explains the variation")
    else:
        print(f"     Focus on overall ratio with peak as supporting detail")
    
    print(f"\n  🔬 SCIENTIFIC INTEGRITY:")
    print(f"     Best practice: Report full breakdown")
    print(f"     Allow readers to make informed judgments")
    print(f"     Highlight innovations without overselling")

def main():
    """Run peak spectral analysis"""
    analyze_peak_spectral_claims()

if __name__ == "__main__":
    main() 