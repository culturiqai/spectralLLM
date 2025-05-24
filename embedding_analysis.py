#!/usr/bin/env python3
"""
Deep SpectralEmbedding Analysis
=============================

Analyze EXACTLY what happens computationally in the embedding layer:
- How much computation is spectral vs traditional?  
- What do the frequency_amplitudes/phases actually do?
- How does the hybrid mixing work mathematically?
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_embedding_parameters():
    """Analyze the embedding parameters from checkpoint"""
    
    print("🔍 DEEP SPECTRALEMBEDDING PARAMETER ANALYSIS")
    print("=" * 60)
    
    checkpoint = torch.load('../outputs/signalllm_8m_20250523/checkpoint_step_15000.pt', map_location='cpu')
    model_state = checkpoint['model']
    
    # Extract embedding components
    freq_amps = model_state['token_embedding.spectral_embedding.frequency_amplitudes']
    freq_phases = model_state['token_embedding.spectral_embedding.frequency_phases'] 
    frequencies = model_state['token_embedding.spectral_embedding.frequencies']
    standard_emb = model_state['token_embedding.standard_embedding.weight']
    mixing_param = model_state['token_embedding.mixing_param']
    
    print(f"📊 EMBEDDING PARAMETER BREAKDOWN:")
    print(f"  🌊 Frequency amplitudes: {freq_amps.shape} = {freq_amps.numel():,} params")
    print(f"  🌊 Frequency phases: {freq_phases.shape} = {freq_phases.numel():,} params")
    print(f"  🌊 Base frequencies: {frequencies.shape} = {frequencies.numel():,} params")
    print(f"  🔧 Standard embedding: {standard_emb.shape} = {standard_emb.numel():,} params")
    print(f"  🌊🔧 Mixing parameter: {mixing_param.shape} = {mixing_param.numel():,} params")
    
    total_emb_params = freq_amps.numel() + freq_phases.numel() + frequencies.numel() + standard_emb.numel() + mixing_param.numel()
    spectral_emb_params = freq_amps.numel() + freq_phases.numel() + frequencies.numel() + mixing_param.numel()
    
    print(f"\n📈 EMBEDDING PARAMETER RATIO:")
    print(f"  Total embedding params: {total_emb_params:,}")
    print(f"  Spectral embedding params: {spectral_emb_params:,}")
    print(f"  Parameter ratio: {spectral_emb_params/total_emb_params:.1%} spectral")
    
    # Analyze the actual learned values
    print(f"\n🔬 LEARNED VALUES ANALYSIS:")
    mixing_weight = torch.sigmoid(mixing_param).item()
    print(f"  🌊🔧 Mixing weight: {mixing_weight:.1%} spectral vs {1-mixing_weight:.1%} traditional")
    
    print(f"  🌊 Frequency amplitudes stats:")
    print(f"    Range: [{freq_amps.min().item():.3f}, {freq_amps.max().item():.3f}]")
    print(f"    Mean±Std: {freq_amps.mean().item():.3f}±{freq_amps.std().item():.3f}")
    
    print(f"  🌊 Frequency phases stats:")
    print(f"    Range: [{freq_phases.min().item():.3f}, {freq_phases.max().item():.3f}]")
    print(f"    Mean±Std: {freq_phases.mean().item():.3f}±{freq_phases.std().item():.3f}")
    
    print(f"  🌊 Base frequencies: {frequencies.tolist()[:8]}... (showing first 8)")
    
    return {
        'freq_amps': freq_amps,
        'freq_phases': freq_phases, 
        'frequencies': frequencies,
        'standard_emb': standard_emb,
        'mixing_weight': mixing_weight
    }

def analyze_embedding_computation(params):
    """Analyze what actually happens during embedding lookup"""
    
    print(f"\n🔄 EMBEDDING COMPUTATION ANALYSIS")
    print("=" * 50)
    
    vocab_size, harmonic_bases = params['freq_amps'].shape
    embed_dim = params['standard_emb'].shape[1]
    
    print(f"📐 DIMENSIONS:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Harmonic bases: {harmonic_bases}")
    
    print(f"\n⚡ FORWARD PASS COMPUTATION BREAKDOWN:")
    
    # Simulate a forward pass
    batch_size, seq_len = 4, 32  # Example dimensions
    token_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
    
    print(f"  Input: token_ids {token_ids.shape}")
    
    # 1. Standard embedding lookup
    print(f"\n  1️⃣ STANDARD PATH:")
    print(f"    Operation: standard_emb[token_ids]")
    print(f"    Computation: {batch_size}×{seq_len} lookups in {vocab_size}×{embed_dim} table")
    print(f"    FLOPs: ~{batch_size * seq_len} lookups (no multiply-adds)")
    print(f"    🔧 TRADITIONAL: Direct table lookup")
    
    # 2. Spectral embedding generation  
    print(f"\n  2️⃣ SPECTRAL PATH:")
    print(f"    Step A: freq_amps[token_ids] → {batch_size}×{seq_len}×{harmonic_bases}")
    print(f"    Step B: freq_phases[token_ids] → {batch_size}×{seq_len}×{harmonic_bases}")
    print(f"    Step C: frequencies → {harmonic_bases} base frequencies")
    print(f"    Step D: Generate harmonic series per token")
    
    # Simulate spectral generation
    token_amps = params['freq_amps'][token_ids]  # [batch, seq, bases]
    token_phases = params['freq_phases'][token_ids]  # [batch, seq, bases]
    
    print(f"    Step E: For each embedding dimension d:")
    print(f"      spectral_emb[d] = Σ(amp_b * cos(2π * freq_b * d + phase_b))")
    print(f"      where b ∈ [0, {harmonic_bases})")
    
    # Count spectral operations
    harmonic_ops = batch_size * seq_len * embed_dim * harmonic_bases
    cos_ops = batch_size * seq_len * embed_dim * harmonic_bases
    sum_ops = batch_size * seq_len * embed_dim * (harmonic_bases - 1)
    
    total_spectral_ops = harmonic_ops + cos_ops + sum_ops
    
    print(f"    🌊 SPECTRAL FLOPs:")
    print(f"      Harmonic generation: {harmonic_ops:,}")
    print(f"      Cosine evaluations: {cos_ops:,}")  
    print(f"      Summations: {sum_ops:,}")
    print(f"      Total: {total_spectral_ops:,} FLOPs")
    
    # 3. Mixing computation
    print(f"\n  3️⃣ MIXING PATH:")
    mixing_ops = batch_size * seq_len * embed_dim * 2  # Two multiply-adds
    print(f"    Operation: mix_weight * spectral + (1-mix_weight) * standard")
    print(f"    🌊🔧 HYBRID FLOPs: {mixing_ops:,}")
    
    # Total computation analysis
    total_ops = total_spectral_ops + mixing_ops
    spectral_ops = total_spectral_ops + mixing_ops * params['mixing_weight']
    
    print(f"\n📊 COMPUTATIONAL BREAKDOWN:")
    print(f"  Total embedding FLOPs: {total_ops:,}")
    print(f"  Spectral FLOPs: {spectral_ops:,}")
    print(f"  ⚡ RUNTIME SPECTRAL RATIO: {spectral_ops/total_ops:.1%}")
    
    return {
        'total_ops': total_ops,
        'spectral_ops': spectral_ops,
        'spectral_ratio': spectral_ops/total_ops
    }

def compare_embedding_approaches():
    """Compare computational cost vs traditional embeddings"""
    
    print(f"\n⚖️ SPECTRAL vs TRADITIONAL EMBEDDING COMPARISON")
    print("=" * 60)
    
    # Traditional embedding
    print(f"🔧 TRADITIONAL EMBEDDING:")
    print(f"  Operation: Simple table lookup")
    print(f"  Memory: vocab_size × embed_dim parameters")
    print(f"  Computation: O(1) per token (just indexing)")
    print(f"  FLOPs per token: ~0 (pure memory access)")
    
    print(f"\n🌊 SPECTRAL EMBEDDING:")
    print(f"  Operation: Harmonic frequency synthesis")
    print(f"  Memory: vocab_size × harmonic_bases × 2 (amp + phase)")
    print(f"  Computation: O(harmonic_bases × embed_dim) per token")
    print(f"  FLOPs per token: ~{32 * 128} (assuming 32 bases, 128 dim)")
    
    print(f"\n🌊🔧 HYBRID EMBEDDING (SpectralLLM):")
    print(f"  Memory: Both traditional + spectral parameters")
    print(f"  Computation: Both paths + mixing")
    print(f"  Flexibility: Learned weighting between approaches")
    print(f"  Overhead: ~10-100x more computation than traditional")

def analyze_spectral_expressiveness():
    """Analyze what spectral embeddings can express that traditional cannot"""
    
    print(f"\n🎨 SPECTRAL EMBEDDING EXPRESSIVENESS")
    print("=" * 50)
    
    print(f"🌊 SPECTRAL ADVANTAGES:")
    print(f"  ✅ Smooth interpolation: Similar tokens have similar frequency patterns")
    print(f"  ✅ Compositional: Meaning emerges from harmonic combinations")
    print(f"  ✅ Frequency-domain relationships: Captures periodic patterns")
    print(f"  ✅ Learnable basis: Frequencies adapt to data")
    
    print(f"\n🔧 TRADITIONAL ADVANTAGES:")
    print(f"  ✅ Arbitrary mappings: Any token → any vector")
    print(f"  ✅ Computational efficiency: O(1) lookup")
    print(f"  ✅ Independence: No constraints between token embeddings")
    print(f"  ✅ Proven effectiveness: Works well in practice")
    
    print(f"\n🌊🔧 HYBRID BENEFITS:")
    print(f"  ⚡ Best of both: Spectral structure + traditional flexibility")
    print(f"  ⚡ Adaptive: Model learns optimal mixing ratio")
    print(f"  ⚡ Graceful degradation: Falls back to traditional if spectral unhelpful")

def main():
    """Run deep embedding analysis"""
    
    print("🔬 DEEP SPECTRALEMBEDDING ANALYSIS")
    print("=" * 70)
    
    # 1. Analyze parameters
    params = analyze_embedding_parameters()
    
    # 2. Analyze computation
    computation = analyze_embedding_computation(params)
    
    # 3. Compare approaches
    compare_embedding_approaches()
    
    # 4. Analyze expressiveness
    analyze_spectral_expressiveness()
    
    # Final discussion points
    print(f"\n💭 DISCUSSION POINTS:")
    print("=" * 30)
    print(f"🤔 Questions for discussion:")
    print(f"  1. Is {computation['spectral_ratio']:.1%} runtime spectral processing significant?")
    print(f"  2. Does the computational overhead justify the spectral benefits?")
    print(f"  3. How does spectral interpolation compare to learned position embeddings?")
    print(f"  4. Could the spectral approach capture linguistic patterns traditional embeddings miss?")
    print(f"  5. Is the hybrid approach the best of both worlds or unnecessary complexity?")

if __name__ == "__main__":
    main() 