"""
SpectralLLM: Revolutionary Signal-Processing Language Model
===========================================================

A groundbreaking language model that uses wavelet transforms, spectral embeddings, 
and frequency-domain processing instead of traditional attention mechanisms.

Key Features:
- Spectral embeddings using harmonic bases
- Multi-resolution wavelet attention (O(n log n) complexity)  
- 94% spectral processing vs 6% traditional components

Quick Start:
-----------
```python
import spectralllm

# Load a pre-trained model
model = spectralllm.SpectralLLM.from_pretrained("spectralllm-8m")

# Generate text
text = model.generate("The future of AI is", max_length=50)
print(text)

# Train your own model
config = spectralllm.Config(
    vocab_size=50257,
    embed_dim=128,
    num_layers=4,
    harmonic_bases=32
)
model = spectralllm.SpectralLLM(config)
```

Documentation: https://spectralllm.readthedocs.io
GitHub: https://github.com/spectralllm/spectralllm
"""

from .core.model import SignalLLM as SpectralLLM, BasisFunction

__version__ = "0.1.0"
__author__ = "SpectralLLM Team"
__email__ = "contact@spectralllm.ai"
__license__ = "MIT"
__description__ = "Revolutionary signal-processing-based language model"

# Core imports
from .core.config import Config
from .core.embeddings import SpectralEmbedding, HybridEmbedding, SignalPositionalEncoding
from .core.attention import (
    WaveletAttention, 
    FrequencyDomainAttention,
    WaveletTransformerBlock,
    WaveletTransform,
    SpectralFeedForward,
    FourierConvolutionAttention
)
from .core.evolution import HRFEvoController, SpectralGapAnalyzer
from .core.transforms import StochasticTransform, AdaptiveBasisSelection

# Training infrastructure
from .training.trainer import SpectralTrainer, TextDataset
from .utils.tokenizer import SimpleTokenizer

__all__ = [
    # Core classes
    "Config",
    "SpectralLLM", 
    "BasisFunction",
    "SpectralEmbedding",
    "HybridEmbedding",
    "SignalPositionalEncoding",
    
    # Attention mechanisms
    "WaveletAttention",
    "FrequencyDomainAttention",
    "FourierConvolutionAttention",
    "WaveletTransformerBlock",
    "WaveletTransform", 
    "SpectralFeedForward",
    
    # Advanced transforms
    "StochasticTransform",
    "AdaptiveBasisSelection",
    
    # Evolution and analysis
    "HRFEvoController",
    "SpectralGapAnalyzer",
    
    # Training infrastructure
    "SpectralTrainer",
    "TextDataset",
    "SimpleTokenizer",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

# Package metadata
__package_info__ = {
    "name": "spectralllm",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": "https://github.com/spectralllm/spectralllm",
    "keywords": [
        "language-model", "spectral-analysis", "wavelet-transform", 
        "signal-processing", "transformer", "deep-learning", "nlp"
    ]
}

def get_version():
    """Get the current version of SpectralLLM."""
    return __version__

def get_info():
    """Get package information."""
    return __package_info__

def print_info():
    """Print package information."""
    print(f"SpectralLLM v{__version__}")
    print(__description__)
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"Repository: https://github.com/spectralllm/spectralllm") 