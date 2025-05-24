"""
SpectralLLM Demo Module
======================

Interactive demonstrations of SpectralLLM capabilities including:
- Text generation with pre-trained models
- Architecture visualization
- Performance comparisons with standard transformers
- Real-time spectral analysis
"""

from .interactive import InteractiveDemo
from .cli import main

__all__ = ["InteractiveDemo", "main"] 