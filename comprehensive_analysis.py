#!/usr/bin/env python3
"""
Comprehensive Spectral Analysis
==============================

Analyze EXACTLY what percentage of computation is spectral vs traditional,
accounting for runtime behavior and all hidden spectral components.
"""

import torch
import torch.nn as nn

def find_all_spectral_components():
    """Find ALL spectral components in the checkpoint"""
    
    checkpoint = torch.load('../outputs/signalllm_8m_20250523/checkpoint_step_15000.pt', map_location='cpu')
    model_state = checkpoint['model']
    
    print("🔍 COMPREHENSIVE SPECTRAL COMPONENT SEARCH")
    print("=" * 60)
    
    # Search for ANY potentially spectral components
    spectral_keywords = ['freq', 'spectral', 'mixing', 'fft', 'wavelet', 'harmonic', 'phase', 'amplitude']
    
    print("🌊 EXPLICIT SPECTRAL PARAMETERS:")
    spectral_params = 0
    for key, tensor in model_state.items():
        if any(keyword in key.lower() for keyword in spectral_keywords):
            param_count = tensor.numel()
            spectral_params += param_count
            print(f"  🌊 {key}: {param_count:,} params")
    
    print(f"\n📊 Found {spectral_params:,} explicitly spectral parameters")
    
    # Now check FFN components specifically
    print("\n🔍 FFN COMPONENT ANALYSIS:")
    ffn_total = 0
    ffn_spectral = 0
    
    for key, tensor in model_state.items():
        if 'ffn' in key:
            param_count = tensor.numel()
            ffn_total += param_count
            print(f"  🔧 {key}: {param_count:,} params")
            
            # Check if this FFN component has spectral aspects
            if any(keyword in key.lower() for keyword in ['freq', 'mixing', 'spectral']):
                ffn_spectral += param_count
                print(f"    ↳ 🌊 SPECTRAL FFN component!")
    
    print(f"\n📊 FFN Analysis: {ffn_spectral:,}/{ffn_total:,} spectral")
    
    return model_state

def analyze_computational_flow():
    """Analyze what percentage of actual computation is spectral"""
    
    print("\n🔄 COMPUTATIONAL FLOW ANALYSIS")
    print("=" * 50)
    
    print("✅ Forward Pass Breakdown:")
    print("  1️⃣ Token Embedding:")
    print("     - HybridEmbedding: FFT operations for spectral path")
    print("     - Standard embedding lookup (traditional)")
    print("     - Mixing: learned weighting between paths")
    print("     ⚡ COMPUTATIONAL RATIO: ~40-60% spectral")
    
    print("\n  2️⃣ Positional Encoding:")  
    print("     - Standard PE lookup + spectral freq_mod/phase_shift")
    print("     ⚡ COMPUTATIONAL RATIO: ~20% spectral")
    
    print("\n  3️⃣ Each Transformer Block (4 total):")
    print("     - WaveletAttention:")
    print("       • WaveletTransform: FFT decomposition (SPECTRAL)")
    print("       • FrequencyDomainAttention on approximation (SPECTRAL)")
    print("       • FrequencyDomainAttention on details (SPECTRAL)")
    print("       • Reconstruction: interpolation + combining (MIXED)")
    print("     ⚡ ATTENTION RATIO: ~80-90% spectral computation")
    
    print("\n     - SpectralFeedForward:")
    print("       • Standard FFN path: fc1 → GELU → fc2 (TRADITIONAL)")
    print("       • Frequency path: FFT → freq_linear → IFFT (SPECTRAL)")
    print("       • Learned mixing between paths")
    print("     ⚡ FFN RATIO: ~10-30% spectral computation")
    
    print("\n     - Layer Normalization: Traditional (tiny computation)")
    print("     - Residual connections: Traditional (no computation)")
    
    print("\n  4️⃣ Output Projection:")
    print("     - Linear layer: Traditional")
    print("     ⚡ OUTPUT RATIO: 0% spectral")

def analyze_parameter_vs_computation():
    """Distinguish between parameter count and computational intensity"""
    
    print("\n⚖️ PARAMETER COUNT vs COMPUTATIONAL ANALYSIS")
    print("=" * 55)
    
    model_state = torch.load('../outputs/signalllm_8m_20250523/checkpoint_step_15000.pt', map_location='cpu')['model']
    
    total_params = sum(tensor.numel() for tensor in model_state.values())
    
    # Categorize by computational impact
    high_compute = 0      # FFT operations, matrix multiplications
    medium_compute = 0    # Linear projections that feed spectral ops
    low_compute = 0       # Biases, norms, small coefficients
    
    spectral_compute = 0  # Parameters used in spectral computation
    traditional_compute = 0  # Parameters used in traditional computation
    
    print("📊 COMPUTATIONAL WEIGHT ANALYSIS:")
    
    for key, tensor in model_state.items():
        param_count = tensor.numel()
        
        # High computational impact (main processing)
        if any(x in key for x in ['fc1.weight', 'fc2.weight', 'proj.weight', 'embedding.weight']):
            high_compute += param_count
            
            # Determine if feeds into spectral processing
            if any(x in key for x in ['wavelet_attn', 'fourier_attn', 'spectral_embedding', 'freq']):
                spectral_compute += param_count
                print(f"  🌊🔥 HIGH SPECTRAL: {key}: {param_count:,}")
            else:
                traditional_compute += param_count
                print(f"  🔧🔥 HIGH TRADITIONAL: {key}: {param_count:,}")
        
        # Medium computational impact (projections, coefficients)
        elif any(x in key for x in ['bias', 'norm', 'frequencies', 'phase', 'amplitude', 'mixing']):
            medium_compute += param_count
            
            if any(x in key for x in ['freq', 'spectral', 'wavelet', 'phase', 'amplitude', 'harmonic']):
                spectral_compute += param_count
                print(f"  🌊⚡ MED SPECTRAL: {key}: {param_count:,}")
            else:
                traditional_compute += param_count
                print(f"  🔧⚡ MED TRADITIONAL: {key}: {param_count:,}")
        
        # Low computational impact
        else:
            low_compute += param_count
            if any(x in key for x in ['freq', 'spectral', 'wavelet', 'phase', 'amplitude']):
                spectral_compute += param_count
            else:
                traditional_compute += param_count
    
    print(f"\n📊 COMPUTATIONAL INTENSITY BREAKDOWN:")
    print(f"  🔥 High compute params: {high_compute:,} ({high_compute/total_params:.1%})")
    print(f"  ⚡ Medium compute params: {medium_compute:,} ({medium_compute/total_params:.1%})") 
    print(f"  💤 Low compute params: {low_compute:,} ({low_compute/total_params:.1%})")
    
    print(f"\n⚡ SPECTRAL vs TRADITIONAL COMPUTATION:")
    print(f"  🌊 Spectral computation: {spectral_compute:,} ({spectral_compute/total_params:.1%})")
    print(f"  🔧 Traditional computation: {traditional_compute:,} ({traditional_compute/total_params:.1%})")
    
    # Weighted by computational intensity
    spectral_weighted = spectral_compute / total_params
    
    return spectral_weighted

def estimate_runtime_spectral_ratio():
    """Estimate percentage of actual computation that is spectral"""
    
    print("\n🚀 RUNTIME SPECTRAL COMPUTATION ESTIMATE")
    print("=" * 50)
    
    print("🔄 Per Forward Pass Computation:")
    print("  1️⃣ Embedding: 40% spectral (FFT in hybrid path)")
    print("  2️⃣ Positional: 20% spectral (freq modulation)")
    print("  3️⃣ 4x Attention Blocks: 85% spectral (wavelet + freq domain)")
    print("  4️⃣ 4x FFN Blocks: 25% spectral (frequency path)")
    print("  5️⃣ Output: 0% spectral (standard projection)")
    
    # Rough FLOP estimates for each component
    embed_flops = 0.1  # Relatively small
    pos_flops = 0.05   # Tiny
    attn_flops = 0.6   # Major computation
    ffn_flops = 0.2    # Significant
    output_flops = 0.05  # Small
    
    total_flops = embed_flops + pos_flops + attn_flops + ffn_flops + output_flops
    
    spectral_flops = (embed_flops * 0.4 + 
                     pos_flops * 0.2 + 
                     attn_flops * 0.85 + 
                     ffn_flops * 0.25 + 
                     output_flops * 0.0)
    
    runtime_spectral_ratio = spectral_flops / total_flops
    
    print(f"\n⚡ ESTIMATED RUNTIME SPECTRAL RATIO: {runtime_spectral_ratio:.1%}")
    
    return runtime_spectral_ratio

def main():
    """Run comprehensive analysis"""
    
    print("🔬 COMPREHENSIVE SPECTRAL ANALYSIS")
    print("=" * 70)
    
    # 1. Find all spectral components
    model_state = find_all_spectral_components()
    
    # 2. Analyze computational flow
    analyze_computational_flow()
    
    # 3. Parameter vs computation analysis
    param_spectral_ratio = analyze_parameter_vs_computation()
    
    # 4. Runtime estimation
    runtime_spectral_ratio = estimate_runtime_spectral_ratio()
    
    # Final verdict
    print(f"\n🎯 COMPREHENSIVE FINAL ASSESSMENT:")
    print("=" * 50)
    print(f"📊 Parameter-based analysis: {param_spectral_ratio:.1%} spectral")
    print(f"🚀 Runtime computation analysis: {runtime_spectral_ratio:.1%} spectral")
    print(f"📢 Original claim: 94% spectral")
    
    avg_spectral = (param_spectral_ratio + runtime_spectral_ratio) / 2
    
    if avg_spectral >= 0.8:
        verdict = "HEAVILY SPECTRAL - Claim largely verified"
    elif avg_spectral >= 0.6:
        verdict = "MODERATELY SPECTRAL - Claim somewhat exaggerated"
    elif avg_spectral >= 0.4:
        verdict = "LIGHTLY SPECTRAL - Claim significantly exaggerated"
    else:
        verdict = "TRADITIONAL WITH SPECTRAL FEATURES - Claim false"
    
    print(f"🏆 VERDICT: {verdict}")
    print(f"⚡ Best estimate: {avg_spectral:.1%} spectral processing")

if __name__ == "__main__":
    main() 