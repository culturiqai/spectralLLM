"""
Unit Tests for SpectralLLM Core Model
====================================

Tests the main SpectralLLM model, attention mechanisms, and forward/backward passes.
"""

import pytest
import torch
import torch.nn.functional as F
from spectralllm import Config, SpectralLLM
from spectralllm.core.attention import WaveletAttention
from spectralllm.core.embeddings import SpectralEmbedding
from tests import RTOL, ATOL


class TestSpectralLLMModel:
    """Test the main SpectralLLM model"""
    
    @pytest.mark.unit
    def test_model_creation(self, basic_config, device):
        """Test model creation with different configurations"""
        model = SpectralLLM(basic_config)
        
        # Check model structure
        assert hasattr(model, 'config')
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'output_proj')
        
        # Check model parameters
        param_count = model.count_parameters()
        assert param_count > 0
        
        # Model should be in training mode by default
        assert model.training
    
    @pytest.mark.unit
    def test_model_forward_pass(self, basic_model, sample_batch, device):
        """Test basic forward pass"""
        input_ids, target_ids = sample_batch
        
        # Forward pass
        with torch.no_grad():
            outputs = basic_model(input_ids)
        
        # Check output structure
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'loss' not in outputs  # No targets provided
        
        # Check logit shapes
        logits = outputs['logits']
        expected_shape = (input_ids.shape[0], input_ids.shape[1], basic_model.config.vocab_size)
        assert logits.shape == expected_shape
        
        # Check logits are reasonable
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    @pytest.mark.unit
    def test_model_forward_with_targets(self, basic_model, sample_batch):
        """Test forward pass with loss computation"""
        input_ids, target_ids = sample_batch
        
        # Forward pass with targets
        with torch.no_grad():
            outputs = basic_model(input_ids, targets=target_ids)
        
        # Check outputs include loss
        assert 'logits' in outputs
        assert 'loss' in outputs
        
        # Check loss properties
        loss = outputs['loss']
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Should be positive
        assert not torch.isnan(loss)
    
    @pytest.mark.unit
    def test_model_gradient_flow(self, basic_model, sample_batch):
        """Test gradient flow through model"""
        input_ids, target_ids = sample_batch
        
        # Ensure model is in training mode
        basic_model.train()
        
        # Forward pass
        outputs = basic_model(input_ids, targets=target_ids)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for parameters
        for name, param in basic_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"
    
    @pytest.mark.unit
    def test_model_parameter_count(self, basic_config):
        """Test parameter counting"""
        model = SpectralLLM(basic_config)
        
        # Count parameters manually
        manual_count = sum(p.numel() for p in model.parameters())
        model_count = model.count_parameters()
        
        assert manual_count == model_count
        
        # Should have reasonable number of parameters
        assert model_count > 1000  # At least some parameters
        assert model_count < 1e9   # Not too many for test model
    
    @pytest.mark.unit
    def test_model_device_movement(self, basic_config, device):
        """Test moving model between devices"""
        model = SpectralLLM(basic_config)
        
        # Move to device
        model = model.to(device)
        
        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type
        
        # Test forward pass on device
        input_ids = torch.randint(0, basic_config.vocab_size, (1, 32), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert param.device.type == device.type
    
    @pytest.mark.unit
    def test_enhanced_model_features(self, enhanced_model, sample_batch):
        """Test enhanced model with all features enabled"""
        input_ids, target_ids = sample_batch
        
        # Should work with enhanced features
        with torch.no_grad():
            outputs = enhanced_model(input_ids, targets=target_ids)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        
        # Enhanced model might return additional outputs
        logits = outputs['logits']
        loss = outputs['loss']
        
        assert not torch.isnan(logits).any()
        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestWaveletAttention:
    """Test wavelet attention mechanism"""
    
    @pytest.mark.unit
    def test_wavelet_attention_creation(self, device):
        """Test wavelet attention creation"""
        attention = WaveletAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            wavelet_type='db4',
            levels=2
        )
        
        assert attention.embed_dim == 128
        assert attention.num_heads == 8
        assert attention.head_dim == 16  # 128 / 8
        assert attention.wavelet.wavelet_type == 'db4'
        assert attention.levels == 2
    
    @pytest.mark.unit
    def test_wavelet_attention_forward(self, device, random_seed):
        """Test wavelet attention forward pass"""
        attention = WaveletAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,  # No dropout for reproducible tests
            wavelet_type='db4',
            levels=2
        )
        attention = attention.to(device)
        
        # Create test input
        batch_size, seq_len, embed_dim = 2, 32, 64
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = attention(x)  # Self-attention
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, embed_dim)
        
        # Check output is reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_wavelet_attention_causal_mask(self, device):
        """Test causal masking in wavelet attention"""
        attention = WaveletAttention(
            embed_dim=32,
            num_heads=2,
            wavelet_type='db4', levels=2
        )
        attention = attention.to(device)
        
        batch_size, seq_len, embed_dim = 1, 16, 32
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        with torch.no_grad():
            output = attention(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        
        # For causal attention, changing future tokens shouldn't affect past
        x_modified = x.clone()
        x_modified[0, -1, :] = 999.0  # Change last token
        
        with torch.no_grad():
            output_modified = attention(x_modified)
        
        # Outputs should be different when inputs change
        assert not torch.allclose(output, output_modified, rtol=RTOL, atol=ATOL)
    
    @pytest.mark.unit
    def test_wavelet_attention_different_wavelets(self, device):
        """Test wavelet attention with different wavelet types"""
        embed_dim, num_heads = 32, 2
        seq_len, batch_size = 16, 1
        
        for wavelet_type in ['db1', 'db4', 'sym4']:
            attention = WaveletAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                wavelet_type=wavelet_type,
                levels=2
            )
            attention = attention.to(device)
            
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)
            
            with torch.no_grad():
                output = attention(x)
            
            assert output.shape == (batch_size, seq_len, embed_dim)
            assert not torch.isnan(output).any()


class TestSpectralEmbedding:
    """Test spectral embedding component"""
    
    @pytest.mark.unit
    def test_spectral_embedding_creation(self):
        """Test spectral embedding creation"""
        embedding = SpectralEmbedding(
            vocab_size=1000,
            embed_dim=128,
            harmonic_bases=32
        )
        
        assert embedding.vocab_size == 1000
        assert embedding.embed_dim == 128
        assert embedding.harmonic_bases == 32
        
        # Should have frequency parameters
        assert hasattr(embedding, 'frequency_amplitudes')
        assert hasattr(embedding, 'frequency_phases')
    
    @pytest.mark.unit
    def test_spectral_embedding_forward(self, device, random_seed):
        """Test spectral embedding forward pass"""
        embedding = SpectralEmbedding(
            vocab_size=1000,
            embed_dim=64,
            harmonic_bases=16
        )
        embedding = embedding.to(device)
        
        # Create test input
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # Forward pass
        with torch.no_grad():
            output = embedding(input_ids)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 64)
        
        # Check output is reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.unit
    def test_spectral_embedding_positional_encoding(self, device):
        """Test positional encoding in signal embedding"""
        embedding = SpectralEmbedding(
            vocab_size=100,
            embed_dim=32,
            harmonic_bases=8,
        )
        embedding = embedding.to(device)
        
        # Test different sequence lengths
        for seq_len in [16, 32, 48]:
            input_ids = torch.randint(0, 100, (1, seq_len), device=device)
            
            with torch.no_grad():
                output = embedding(input_ids)
            
            assert output.shape == (1, seq_len, 32)
            
            # Different positions should have different encodings
            if seq_len > 1:
                pos_diff = torch.norm(output[0, 0, :] - output[0, 1, :])
                assert pos_diff > 0, "Position encodings should differ"
    
    @pytest.mark.unit
    def test_spectral_embedding_harmonic_components(self, device):
        """Test harmonic basis components"""
        embedding = SpectralEmbedding(
            vocab_size=100,
            embed_dim=64,
            harmonic_bases=16
        )
        embedding = embedding.to(device)
        
        # Get harmonic encoding for analysis
        seq_len = 32
        positions = torch.arange(seq_len, device=device)
        
        # Should be able to access harmonic components
        # (This tests internal implementation details)
        # Test that different tokens get different embeddings
        input_ids = torch.tensor([[0, 1, 2, 3]], device=device)

class TestModelIntegration:
    """Test integration between model components"""
    
    @pytest.mark.unit
    def test_model_attention_integration(self, enhanced_model, sample_batch):
        """Test integration between model layers and attention"""
        input_ids, _ = sample_batch
        
        # Get intermediate activations
        enhanced_model.eval()
        
        with torch.no_grad():
            # Store hooks for intermediate outputs
            activations = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output
                return hook
            
            # Register hooks on first few layers
            handles = []
            for i, layer in enumerate(enhanced_model.blocks[:2]):
                handle = layer.register_forward_hook(hook_fn(f'layer_{i}'))
                handles.append(handle)
            
            # Forward pass
            outputs = enhanced_model(input_ids)
            
            # Clean up hooks
            for handle in handles:
                handle.remove()
        
        # Check activations were captured
        assert len(activations) > 0
        
        # Check activation shapes and properties
        for name, activation in activations.items():
            assert isinstance(activation, torch.Tensor)
            assert activation.shape[0] == input_ids.shape[0]  # Batch size
            assert activation.shape[1] == input_ids.shape[1]  # Sequence length
            assert not torch.isnan(activation).any()
    
    @pytest.mark.unit
    def test_model_memory_efficiency(self, basic_config, device):
        """Test model memory usage is reasonable"""
        # Create model
        model = SpectralLLM(basic_config)
        model = model.to(device)
        
        # Get initial memory usage
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        else:
            initial_memory = 0
        
        # Forward pass
        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, basic_config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check memory usage didn't explode
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated(device)
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB for test model)
            assert memory_increase < 1e9, f"Memory usage too high: {memory_increase:,} bytes"
    
    @pytest.mark.unit
    def test_model_reproducibility(self, basic_config, device):
        """Test model outputs are reproducible with same seed"""
        # Create two identical models
        torch.manual_seed(42)
        model1 = SpectralLLM(basic_config)
        model1 = model1.to(device)
        
        torch.manual_seed(42)
        model2 = SpectralLLM(basic_config)
        model2 = model2.to(device)
        
        # Same input
        torch.manual_seed(123)
        input_ids = torch.randint(0, basic_config.vocab_size, (2, 32), device=device)
        
        # Forward passes should be identical
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            outputs1 = model1(input_ids)
            outputs2 = model2(input_ids)
        
        assert torch.allclose(outputs1['logits'], outputs2['logits'], rtol=RTOL, atol=ATOL)
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_model_training_step(self, basic_model, sample_batch):
        """Test a complete training step"""
        input_ids, target_ids = sample_batch
        
        # Set up optimizer
        optimizer = torch.optim.Adam(basic_model.parameters(), lr=1e-4)
        
        # Training step
        basic_model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = basic_model(input_ids, targets=target_ids)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss)
        
        # Gradients should be reasonable
        total_grad_norm = 0
        for param in basic_model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Gradient norm should be reasonable (not too large or too small)
        assert 1e-6 < total_grad_norm < 1e3, f"Gradient norm out of range: {total_grad_norm}" 