"""
SpectralLLM Test Suite
====================

Comprehensive test suite for SpectralLLM package including:
- Unit tests for individual components
- Integration tests for end-to-end workflows  
- Performance tests for complexity validation
- Hardware compatibility tests
"""

import pytest
import torch
import sys
import os

# Add the spectralllm package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() 
                     else 'cuda' if torch.cuda.is_available() 
                     else 'cpu')

# Test data constants
TEST_VOCAB_SIZE = 1000
TEST_EMBED_DIM = 128
TEST_SEQ_LENGTH = 64
TEST_BATCH_SIZE = 2
TEST_NUM_HEADS = 4
TEST_NUM_LAYERS = 2

# Tolerance for numerical tests
RTOL = 1e-4
ATOL = 1e-6 