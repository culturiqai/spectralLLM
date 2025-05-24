#!/usr/bin/env python3
"""
SpectralLLM Demo Entry Point
===========================

This module enables running SpectralLLM demos via:
    python -m spectralllm.demo

Available demo modes:
- interactive: Interactive text generation and analysis
- benchmark: Performance comparison with standard models  
- visualize: Real-time spectral analysis visualization
- train: Quick training demonstration
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main()) 