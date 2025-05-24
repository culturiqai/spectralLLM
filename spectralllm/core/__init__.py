"""
SpectralLLM Core Module
======================

Core components of the SpectralLLM architecture including:
- Model configuration and main SignalLLM class
- Spectral and hybrid embeddings
- Wavelet attention mechanisms
- Transform operations
"""

from .config import Config
from .model import SignalLLM, BasisFunction
from .embeddings import SpectralEmbedding, HybridEmbedding, SignalPositionalEncoding
from .attention import (
    WaveletAttention,
    FrequencyDomainAttention,
    FourierConvolutionAttention,
    WaveletTransformerBlock,
    WaveletTransform,
    SpectralFeedForward
)
from .evolution import HRFEvoController, SpectralGapAnalyzer
from .transforms import StochasticTransform, AdaptiveBasisSelection

__all__ = [
    "Config",
    "SignalLLM",
    "BasisFunction",
    "SpectralEmbedding", 
    "HybridEmbedding",
    "SignalPositionalEncoding",
    "WaveletAttention",
    "FrequencyDomainAttention",
    "FourierConvolutionAttention",
    "WaveletTransformerBlock",
    "WaveletTransform",
    "SpectralFeedForward",
    "HRFEvoController",
    "SpectralGapAnalyzer", 
    "StochasticTransform",
    "AdaptiveBasisSelection"
] 