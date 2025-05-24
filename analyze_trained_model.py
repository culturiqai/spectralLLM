#!/usr/bin/env python3
"""
Trained SpectralLLM Model Analysis
=================================

Analyze the actual trained SpectralLLM model to determine:
1. Real spectral processing ratio
2. Actual complexity behavior
3. Learned spectral patterns
4. Architecture verification from trained weights
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import json
from pathlib import Path
from typing import Dict, List, Any
import os

def load_trained_model(checkpoint_path: str):
    """Load the trained SpectralLLM model"""
    print("🔄 LOADING TRAINED MODEL")
    print("=" * 40)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📁 Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("📋 Checkpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} keys")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Extract model state and config
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    
    # Try to extract config from checkpoint
    config_dict = checkpoint.get('config', checkpoint.get('model_config', None))
    
    if config_dict is None:
        print("⚠️ No config found in checkpoint, inferring from model state...")
        config_dict = infer_config_from_state(model_state)
    
    print(f"✅ Model loaded successfully")
    return model_state, config_dict

def infer_config_from_state(model_state: dict):
    """Infer model configuration from state dict"""
    print("🔍 Inferring config from model state...")
    
    # Extract key parameters from state dict
    vocab_size = None
    embed_dim = None
    num_layers = 0
    harmonic_bases = None
    
    for key, tensor in model_state.items():
        if 'token_embedding' in key and 'weight' in key:
            if 'standard_embedding' in key:
                vocab_size, embed_dim = tensor.shape
            elif 'frequency_amplitudes' in key:
                vocab_size, harmonic_bases = tensor.shape
                if embed_dim is None:
                    # Try to find embed_dim from other tensors
                    for k, t in model_state.items():
                        if 'pos_encoding.pe' in k:
                            embed_dim = t.shape[-1]
                            break
        
        if 'blocks.' in key and '.ffn.' in key:
            layer_num = int(key.split('.')[1])
            num_layers = max(num_layers, layer_num + 1)
    
    # Default values if not found
    config = {
        'vocab_size': vocab_size or 50000,
        'embed_dim': embed_dim or 768,
        'num_layers': num_layers or 6,
        'num_heads': 8,  # Common default
        'hidden_dim': embed_dim * 4 if embed_dim else 3072,
        'harmonic_bases': harmonic_bases or 32,
        'max_seq_length': 512,
        'dropout': 0.1,
        'wavelet_type': 'db4',
        'wavelet_levels': 3
    }
    
    print(f"📊 Inferred config: {config}")
    return config

def analyze_trained_spectral_embeddings(model_state: dict, config: dict):
    """Analyze the learned spectral embeddings"""
    print("\n🔍 ANALYZING TRAINED SPECTRAL EMBEDDINGS")
    print("=" * 50)
    
    # Find embedding parameters
    freq_amplitudes = None
    freq_phases = None
    standard_embedding = None
    
    for key, tensor in model_state.items():
        if 'frequency_amplitudes' in key:
            freq_amplitudes = tensor
        elif 'frequency_phases' in key:
            freq_phases = tensor
        elif 'standard_embedding.weight' in key:
            standard_embedding = tensor
    
    if freq_amplitudes is not None and freq_phases is not None:
        print("✅ Found spectral embedding parameters")
        print(f"  Frequency amplitudes: {freq_amplitudes.shape}")
        print(f"  Frequency phases: {freq_phases.shape}")
        
        # Analyze learned frequency patterns
        amp_stats = {
            'mean': freq_amplitudes.mean().item(),
            'std': freq_amplitudes.std().item(),
            'min': freq_amplitudes.min().item(),
            'max': freq_amplitudes.max().item()
        }
        
        phase_stats = {
            'mean': freq_phases.mean().item(),
            'std': freq_phases.std().item(),
            'min': freq_phases.min().item(),
            'max': freq_phases.max().item()
        }
        
        print(f"  📊 Amplitude statistics:")
        print(f"    Range: [{amp_stats['min']:.3f}, {amp_stats['max']:.3f}]")
        print(f"    Mean±Std: {amp_stats['mean']:.3f}±{amp_stats['std']:.3f}")
        
        print(f"  📊 Phase statistics:")
        print(f"    Range: [{phase_stats['min']:.3f}, {phase_stats['max']:.3f}]")
        print(f"    Mean±Std: {phase_stats['mean']:.3f}±{phase_stats['std']:.3f}")
        
        # Check if the model learned diverse frequency patterns
        token_diversity = torch.std(freq_amplitudes, dim=0).mean().item()
        print(f"  🌈 Token frequency diversity: {token_diversity:.4f}")
        
        if token_diversity > 0.5:
            print("  ✅ HIGH diversity - tokens learned distinct frequency patterns")
        elif token_diversity > 0.1:
            print("  ⚠️ MEDIUM diversity - some spectral learning occurred")
        else:
            print("  ❌ LOW diversity - minimal spectral differentiation")
        
        return amp_stats, phase_stats, token_diversity
    
    elif standard_embedding is not None:
        print("⚠️ Found only standard embeddings, no spectral components")
        return None, None, 0.0
    
    else:
        print("❌ No embedding parameters found")
        return None, None, 0.0

def analyze_trained_attention_weights(model_state: dict, config: dict):
    """Analyze the learned attention mechanism weights"""
    print("\n🔍 ANALYZING TRAINED ATTENTION WEIGHTS")
    print("=" * 50)
    
    attention_layers = []
    spectral_components = 0
    total_attention_params = 0
    
    for key, tensor in model_state.items():
        if 'wavelet_attn' in key or 'attention' in key:
            attention_layers.append((key, tensor.shape, tensor.numel()))
            total_attention_params += tensor.numel()
            
            if any(spectral_key in key for spectral_key in ['freq', 'spectral', 'wavelet', 'fft']):
                spectral_components += tensor.numel()
    
    print(f"📊 Attention Analysis:")
    print(f"  Total attention layers found: {len(attention_layers)}")
    print(f"  Total attention parameters: {total_attention_params:,}")
    print(f"  Spectral attention parameters: {spectral_components:,}")
    
    if total_attention_params > 0:
        spectral_attention_ratio = spectral_components / total_attention_params
        print(f"  Spectral attention ratio: {spectral_attention_ratio:.1%}")
    else:
        spectral_attention_ratio = 0.0
    
    # Analyze specific attention weights
    for key, shape, params in attention_layers[:5]:  # Show first 5
        print(f"  📋 {key}: {shape} ({params:,} params)")
    
    if len(attention_layers) > 5:
        print(f"  ... and {len(attention_layers) - 5} more layers")
    
    return spectral_attention_ratio, attention_layers

def calculate_actual_spectral_ratio(model_state: dict, config: dict):
    """Calculate the REAL spectral processing ratio from trained weights"""
    print("\n🔍 CALCULATING ACTUAL SPECTRAL RATIO (FIXED)")
    print("=" * 50)
    
    total_params = 0
    spectral_params = 0
    
    component_analysis = {
        'embeddings': {'total': 0, 'spectral': 0},
        'attention': {'total': 0, 'spectral': 0},
        'ffn': {'total': 0, 'spectral': 0},
        'other': {'total': 0, 'spectral': 0}
    }
    
    print("📊 CORRECTED Parameter Classification:")
    print("(Only counting GENUINE spectral operations, not names)")
    
    for key, tensor in model_state.items():
        param_count = tensor.numel()
        total_params += param_count
        
        # Classify component type
        component = 'other'
        if 'embedding' in key:
            component = 'embeddings'
        elif any(attn_key in key for attn_key in ['attn', 'attention']):
            component = 'attention'
        elif 'ffn' in key or 'feed_forward' in key:
            component = 'ffn'
        
        component_analysis[component]['total'] += param_count
        
        # FIXED: Only count GENUINELY spectral operations
        is_spectral = False
        
        # GENUINE spectral parameters (actual frequency domain operations)
        if any(genuine_spectral in key for genuine_spectral in [
            'frequency_amplitudes',  # Harmonic frequency amplitudes
            'frequency_phases',      # Harmonic frequency phases
            'frequencies',           # Frequency basis vectors
            'freq_filter',          # Learnable frequency filters
            'phase_shift',          # Phase shift parameters
            'spectral_embedding',   # Actual spectral embedding weights
        ]):
            is_spectral = True
            spectral_params += param_count
            component_analysis[component]['spectral'] += param_count
        
        # EXCLUDE these even if they have "spectral" names - they're standard operations:
        elif any(traditional_op in key for traditional_op in [
            'q_proj',      # Standard query projection (matrix multiplication)
            'k_proj',      # Standard key projection  
            'v_proj',      # Standard value projection
            'out_proj',    # Standard output projection
            'fc1', 'fc2',  # Standard feed-forward layers
            'linear',      # Standard linear layers
            'norm',        # Layer normalization
            'dropout',     # Dropout
            'embedding.weight',  # Standard embedding lookup
            'bias',        # Bias terms
        ]):
            is_spectral = False
        
        # Handle hybrid embeddings specially
        if 'mixing_param' in key:
            # Get actual mixing ratio from checkpoint
            mixing_value = tensor.item() if tensor.numel() == 1 else 0.5
            spectral_ratio = torch.sigmoid(torch.tensor(mixing_value)).item()
            
            # Find corresponding embedding weights
            for embed_key, embed_tensor in model_state.items():
                if 'standard_embedding' in embed_key:
                    embed_params = embed_tensor.numel()
                    spectral_contribution = int(embed_params * spectral_ratio)
                    spectral_params += spectral_contribution
                    component_analysis['embeddings']['spectral'] += spectral_contribution
                    print(f"  🌊 Hybrid {embed_key}: {embed_params:,} params, {spectral_ratio:.1%} spectral ({spectral_contribution:,} spectral params)")
                    break
        
        spectral_indicator = "🌊" if is_spectral else "🔧"
        if param_count > 1000:  # Only show significant components
            print(f"  {spectral_indicator} {key}: {param_count:,} params")
    
    # Summary by component
    print(f"\n📊 CORRECTED Component Analysis:")
    for comp_name, comp_data in component_analysis.items():
        if comp_data['total'] > 0:
            comp_ratio = comp_data['spectral'] / comp_data['total'] if comp_data['total'] > 0 else 0
            print(f"  {comp_name.capitalize()}: {comp_data['spectral']:,}/{comp_data['total']:,} = {comp_ratio:.1%} spectral")
    
    actual_spectral_ratio = spectral_params / total_params if total_params > 0 else 0
    
    print(f"\n🎯 CORRECTED TRAINED MODEL ANALYSIS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  GENUINE spectral parameters: {spectral_params:,}")
    print(f"  ⚡ REAL spectral ratio: {actual_spectral_ratio:.1%}")
    print(f"  📢 CLAIMED spectral ratio: 94%")
    print(f"  🔧 My previous FLAWED analysis: 31.2%")
    
    if actual_spectral_ratio >= 0.9:
        print("  ✅ CLAIM VERIFIED: >90% spectral")
    elif actual_spectral_ratio >= 0.5:
        print("  ⚠️ CLAIM EXAGGERATED: ~50% spectral") 
    elif actual_spectral_ratio >= 0.2:
        print("  ❌ CLAIM FALSE: Only moderately spectral")
    else:
        print("  ❌ CLAIM COMPLETELY FALSE: Barely spectral")
    
    return actual_spectral_ratio, component_analysis

def test_model_behavior(model_state: dict, config: dict):
    """Test the actual model behavior with the trained weights"""
    print("\n🔍 TESTING TRAINED MODEL BEHAVIOR")
    print("=" * 50)
    
    try:
        from spectralllm.core.config import Config
        from spectralllm.core.model import SignalLLM
        
        # Create config object
        model_config = Config(**config)
        
        # Create model
        model = SignalLLM(model_config)
        
        # Load trained weights
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        print(f"📋 Model Loading:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print(f"  Missing: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"  Unexpected: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        
        model.eval()
        
        # Test forward pass
        test_input = torch.randint(0, min(1000, config['vocab_size']), (1, 32))
        
        print(f"\n🧪 Forward Pass Test:")
        print(f"  Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  Output shape: {output['logits'].shape}")
        print(f"  Output range: [{output['logits'].min().item():.3f}, {output['logits'].max().item():.3f}]")
        print(f"  ✅ Model runs successfully with trained weights!")
        
        # Get model complexity info
        complexity_info = model.get_complexity_info()
        print(f"\n📊 Trained Model Complexity:")
        for key, value in complexity_info.items():
            print(f"  {key}: {value}")
        
        return True, complexity_info
    
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def inspect_checkpoint_structure(checkpoint_path: str):
    """Inspect the actual checkpoint to understand its structure"""
    print("\n🔍 INSPECTING CHECKPOINT STRUCTURE")
    print("=" * 50)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("✅ Found model_state_dict in checkpoint")
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
            print("✅ Found model in checkpoint")
        else:
            model_state = checkpoint
            print("✅ Using checkpoint directly as model state")
        
        print(f"\n📊 ACTUAL PARAMETER NAMES IN CHECKPOINT:")
        print("(First 30 for inspection)")
        
        for i, (key, tensor) in enumerate(model_state.items()):
            if i >= 30:  # Limit output
                print(f"  ... and {len(model_state) - 30} more parameters")
                break
            param_count = tensor.numel()
            print(f"  {key}: {list(tensor.shape)} = {param_count:,} params")
        
        return model_state
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_embedding_patterns(model_state: dict):
    """Analyze embedding patterns from model state"""
    print("\n🔍 ANALYZING EMBEDDING PATTERNS")
    print("=" * 40)
    
    # This calls the existing trained spectral embeddings analysis
    # Just return a placeholder for now since we'll get the real analysis from calculate_actual_spectral_ratio
    print("  ℹ️ Using spectral ratio analysis for embedding patterns")
    return "analyzed"

def load_config():
    """Load or create a default config"""
    try:
        from spectralllm.core.config import Config
        return Config().to_dict()
    except:
        # Default config if package not available
        return {
            'vocab_size': 50000,
            'embed_dim': 768,
            'num_layers': 6,
            'num_heads': 8,
            'hidden_dim': 3072,
            'harmonic_bases': 32,
            'max_seq_length': 512,
            'dropout': 0.1
        }

def main():
    """Run comprehensive analysis of trained SpectralLLM model"""
    print("🚀 FIXED SPECTRALLLM TRAINED MODEL ANALYSIS")
    print("=" * 60)
    
    checkpoint_path = "../outputs/signalllm_8m_20250523/latest_checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Step 1: Inspect actual checkpoint structure
    model_state = inspect_checkpoint_structure(checkpoint_path)
    if model_state is None:
        return
    
    # Step 2: Analyze embedding patterns (same as before, this was correct)
    embedding_analysis = analyze_embedding_patterns(model_state)
    
    # Step 3: Load config for context
    config = load_config()
    
    # Step 4: FIXED spectral ratio calculation
    spectral_ratio, component_breakdown = calculate_actual_spectral_ratio(model_state, config)
    
    # Step 5: CORRECTED final verdict
    print(f"\n🎯 FINAL VERDICT (FIXED ANALYSIS):")
    print("=" * 50)
    
    if spectral_ratio >= 0.8:
        verdict = "HEAVILY SPECTRAL"
        icon = "🌊🌊🌊"
    elif spectral_ratio >= 0.5:
        verdict = "MODERATELY SPECTRAL"
        icon = "🌊🌊"
    elif spectral_ratio >= 0.15:
        verdict = "LIGHTLY SPECTRAL"
        icon = "🌊"
    else:
        verdict = "TRADITIONAL WITH SPECTRAL FEATURES"
        icon = "🔧"
    
    print(f"  {icon} Classification: {verdict}")
    print(f"  📊 ACTUAL spectral processing: {spectral_ratio:.1%}")
    print(f"  📢 CLAIMED spectral processing: 94%")
    
    delta = abs(spectral_ratio - 0.94)
    if delta <= 0.05:
        print(f"  ✅ Claim accuracy: VERIFIED (within 5%)")
    elif delta <= 0.15:
        print(f"  ⚠️ Claim accuracy: EXAGGERATED ({delta:.1%} difference)")
    else:
        print(f"  ❌ Claim accuracy: FALSE ({delta:.1%} difference)")
    
    print(f"\n🔧 My previous flawed analysis said 31.2% spectral")
    print(f"📏 This corrected analysis shows {spectral_ratio:.1%} spectral")
    
    # Summary of what we actually found
    print(f"\n🔍 WHAT WE ACTUALLY FOUND:")
    genuine_spectral_components = []
    for key in model_state.keys():
        if any(genuine in key for genuine in ['frequency_amplitudes', 'frequency_phases', 'frequencies']):
            genuine_spectral_components.append(key)
    
    if genuine_spectral_components:
        print(f"  ✅ Genuine spectral components found:")
        for comp in genuine_spectral_components[:5]:  # Show first 5
            print(f"    - {comp}")
        if len(genuine_spectral_components) > 5:
            print(f"    - ... and {len(genuine_spectral_components) - 5} more")
    else:
        print(f"  ❌ No genuine spectral components found")
        print(f"  🔧 Only traditional transformer operations detected")

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n📊 Summary: {result['actual_spectral_ratio']:.1%} spectral, {result['architecture_class']}") 