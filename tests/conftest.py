"""
PyTest Configuration and Shared Fixtures
========================================

Provides shared fixtures and configuration for all SpectralLLM tests.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

import spectralllm
from spectralllm import Config, SpectralLLM, SpectralTrainer
from spectralllm.core.transforms import EnhancedWaveletTransform
from spectralllm.validation import PerplexityValidator, DatasetValidator, BaselineEvaluator

from . import (
    DEVICE, TEST_VOCAB_SIZE, TEST_EMBED_DIM, TEST_SEQ_LENGTH, 
    TEST_BATCH_SIZE, TEST_NUM_HEADS, TEST_NUM_LAYERS
)


@pytest.fixture(scope="session")
def device():
    """Test device (CPU/CUDA/MPS)"""
    return DEVICE


@pytest.fixture
def basic_config():
    """Basic SpectralLLM configuration for testing"""
    return Config(
        vocab_size=TEST_VOCAB_SIZE,
        embed_dim=TEST_EMBED_DIM,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        hidden_dim=TEST_EMBED_DIM * 4,
        max_seq_length=TEST_SEQ_LENGTH,
        dropout=0.1,
        harmonic_bases=16,
        wavelet_type='db4',
        wavelet_levels=2,
        use_spectral_embedding=True,
        use_wavelet_attention=True
    )


@pytest.fixture
def enhanced_config():
    """Enhanced configuration with all features enabled"""
    return Config(
        vocab_size=TEST_VOCAB_SIZE,
        embed_dim=TEST_EMBED_DIM,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        hidden_dim=TEST_EMBED_DIM * 4,
        max_seq_length=TEST_SEQ_LENGTH,
        dropout=0.1,
        harmonic_bases=16,
        wavelet_type='db4',
        wavelet_levels=2,
        use_spectral_embedding=True,
        use_wavelet_attention=True,
        use_fourier_convolution=True,
        use_frequency_domain_attention=True,
        wavelet_families=['db4', 'sym4']
    )


@pytest.fixture
def basic_model(basic_config, device):
    """Basic SpectralLLM model for testing"""
    model = SpectralLLM(basic_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def enhanced_model(enhanced_config, device):
    """Enhanced SpectralLLM model with all features"""
    model = SpectralLLM(enhanced_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_batch(device):
    """Sample input batch for testing"""
    batch_size = TEST_BATCH_SIZE
    seq_length = TEST_SEQ_LENGTH
    vocab_size = TEST_VOCAB_SIZE
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    return input_ids, target_ids


@pytest.fixture
def sample_texts():
    """Sample text data for dataset testing"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms analyze large datasets efficiently.",
        "Wavelet transforms provide multi-resolution signal analysis.",
        "Natural language processing enables computer understanding.",
        "Deep learning models learn complex patterns from data.",
        "Spectral methods offer efficient computation for sequences.",
        "Fourier analysis decomposes signals into frequency components.",
        "Attention mechanisms focus on relevant input parts.",
        "Transformer architectures revolutionized language modeling.",
        "Mathematics forms the foundation of artificial intelligence."
    ]


@pytest.fixture
def tokenizer():
    """Simple tokenizer for testing"""
    return spectralllm.SimpleTokenizer(mode='char')


@pytest.fixture
def perplexity_validator():
    """Perplexity validator instance"""
    return PerplexityValidator()


@pytest.fixture
def dataset_validator():
    """Dataset validator instance"""
    return DatasetValidator()


@pytest.fixture
def baseline_evaluator(device):
    """Baseline evaluator instance"""
    return BaselineEvaluator(device=str(device))


@pytest.fixture
def wavelet_transform():
    """Enhanced wavelet transform for testing"""
    return EnhancedWaveletTransform(
        wavelet_type='db4',
        wavelet_levels=2,
        use_learned=False
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)
    yield 42
    # Reset random state after test


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and complexity tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take more than 10 seconds"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU/MPS"
    )
    config.addinivalue_line(
        "markers", "validation: Validation framework tests"
    )


# Skip conditions
skip_no_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="GPU/MPS not available"
)

skip_no_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
) 