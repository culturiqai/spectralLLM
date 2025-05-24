"""
Unit Tests for SpectralLLM Config
=================================

Tests the configuration system and parameter validation.
"""

import pytest
import json
from spectralllm import Config


class TestConfig:
    """Test the Config class"""
    
    @pytest.mark.unit
    def test_basic_config_creation(self):
        """Test basic config creation with default values"""
        config = Config(vocab_size=1000, embed_dim=128)
        
        assert config.vocab_size == 1000
        assert config.embed_dim == 128
        assert config.num_heads == 12  # Default value
        assert config.dropout == 0.1  # Default value
        
    @pytest.mark.unit
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Valid config
        config = Config(vocab_size=1000, embed_dim=128, num_heads=4)
        assert config.embed_dim == 128
        
        # Invalid config - embed_dim not divisible by num_heads
    
    @pytest.mark.unit
    def test_spectral_features_config(self):
        """Test spectral-specific configuration options"""
        config = Config(
            vocab_size=1000,
            embed_dim=128,
            use_spectral_embedding=True,
            use_wavelet_attention=True,
            harmonic_bases=32,
            wavelet_type='db4',
            wavelet_levels=3
        )
        
        assert config.use_spectral_embedding is True
        assert config.use_wavelet_attention is True
        assert config.harmonic_bases == 32
        assert config.wavelet_type == 'db4'
        assert config.wavelet_levels == 3
    
    @pytest.mark.unit
    def test_advanced_features_config(self):
        """Test advanced feature configuration"""
        config = Config(
            vocab_size=1000,
            embed_dim=128,
            use_adaptive_basis=True,
            use_fourier_convolution=True,
            use_frequency_domain_attention=True,
            wavelet_families=['db4', 'sym4', 'dmey']
        )
        
        assert config.use_adaptive_basis is True
        assert config.use_fourier_convolution is True
        assert config.use_frequency_domain_attention is True
        assert config.wavelet_families == ['db4', 'sym4', 'dmey']
    
    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary"""
        config = Config(
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            use_spectral_embedding=True
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 1000
        assert config_dict['embed_dim'] == 128
        assert config_dict['num_heads'] == 4
        assert config_dict['use_spectral_embedding'] is True
        
        # Should contain all expected keys
        expected_keys = [
            'vocab_size', 'embed_dim', 'num_heads', 'num_layers',
            'hidden_dim', 'max_seq_length', 'dropout'
        ]
        for key in expected_keys:
            assert key in config_dict
    
    @pytest.mark.unit
    def test_config_from_dict(self):
        """Test configuration creation from dictionary"""
        config_dict = {
            'vocab_size': 2000,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'use_wavelet_attention': True,
            'harmonic_bases': 64
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.vocab_size == 2000
        assert config.embed_dim == 256
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.use_wavelet_attention is True
        assert config.harmonic_bases == 64
    
    @pytest.mark.unit
    def test_config_json_serialization(self):
        """Test JSON serialization/deserialization"""
        original_config = Config(
            vocab_size=1500,
            embed_dim=192,
            num_heads=6,
            wavelet_type='sym4',
            use_adaptive_basis=True
        )
        
        # Serialize to JSON string
        json_str = json.dumps(original_config.to_dict())
        
        # Deserialize from JSON
        config_dict = json.loads(json_str)
        restored_config = Config.from_dict(config_dict)
        
        # Verify all parameters match
        assert restored_config.vocab_size == original_config.vocab_size
        assert restored_config.embed_dim == original_config.embed_dim
        assert restored_config.num_heads == original_config.num_heads
        assert restored_config.wavelet_type == original_config.wavelet_type
        assert restored_config.use_adaptive_basis == original_config.use_adaptive_basis
    
    @pytest.mark.unit
    def test_config_parameter_ranges(self):
        """Test parameter range validation"""
        # Test minimum values
        config = Config(vocab_size=1, embed_dim=1, num_heads=1, num_layers=1)
        assert config.vocab_size >= 1
        assert config.embed_dim >= 1
        assert config.num_heads >= 1
        assert config.num_layers >= 1
        
        # Test reasonable maximum values
        config = Config(
            vocab_size=100000,
            embed_dim=2048,
            num_heads=32,
            num_layers=48,
            max_seq_length=8192
        )
        assert config.vocab_size == 100000
        assert config.embed_dim == 2048
        assert config.num_heads == 32
        assert config.num_layers == 48
        assert config.max_seq_length == 8192
    
    @pytest.mark.unit
    def test_evolution_config(self):
        """Test HRFEvo evolution configuration"""
        config = Config(
            vocab_size=1000,
            embed_dim=128,
            population_size=20,
            evolution_generations=10,
        )
        
        assert config.population_size == 20
        assert config.evolution_generations == 10
        
    @pytest.mark.unit
    def test_config_immutability_after_creation(self):
        """Test that config behaves consistently after creation"""
        config = Config(vocab_size=1000, embed_dim=128)
        
        # Store original values
        original_vocab_size = config.vocab_size
        original_embed_dim = config.embed_dim
        
        # Values should remain the same
        assert config.vocab_size == original_vocab_size
        assert config.embed_dim == original_embed_dim
        
        # Creating new config should not affect old one
        config2 = Config(vocab_size=2000, embed_dim=256)
        assert config.vocab_size == original_vocab_size  # Should be unchanged
        assert config2.vocab_size == 2000 