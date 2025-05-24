"""
SpectralLLM Configuration
========================

Configuration class for SpectralLLM models with spectral-specific parameters.
"""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Config:
    """Configuration for SpectralLLM models."""
    
    # Model architecture
    vocab_size: int = 50257
    embed_dim: int = 768
    hidden_dim: int = 3072
    num_heads: int = 12
    num_layers: int = 12
    max_seq_length: int = 1024
    dropout: float = 0.1
    
    # Spectral-specific parameters
    harmonic_bases: int = 64
    use_spectral_embedding: bool = True
    spectral_embedding_ratio: float = 0.5
    
    # Wavelet parameters
    wavelet_type: str = 'db4'
    wavelet_levels: int = 3
    wavelet_families: List[str] = None
    use_adaptive_basis: bool = True
    
    # Attention mechanisms
    use_wavelet_attention: bool = True
    use_fourier_convolution: bool = True
    use_frequency_domain_attention: bool = True
    
    # Evolution parameters
    use_hrfevo: bool = True
    evolution_generations: int = 10
    population_size: int = 20
    
    # Training parameters
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.wavelet_families is None:
            self.wavelet_families = ['db4', 'sym4']
        
        # Validate spectral parameters
        if not 0 < self.spectral_embedding_ratio <= 1:
            raise ValueError("spectral_embedding_ratio must be between 0 and 1")
        
        if self.harmonic_bases <= 0:
            raise ValueError("harmonic_bases must be positive")
    
    def get_spectral_embed_dim(self) -> int:
        """Get the dimension for spectral embeddings."""
        return int(self.embed_dim * self.spectral_embedding_ratio)
    
    def get_traditional_embed_dim(self) -> int:
        """Get the dimension for traditional embeddings."""
        return self.embed_dim - self.get_spectral_embed_dim()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout,
            'harmonic_bases': self.harmonic_bases,
            'use_spectral_embedding': self.use_spectral_embedding,
            'spectral_embedding_ratio': self.spectral_embedding_ratio,
            'wavelet_type': self.wavelet_type,
            'wavelet_levels': self.wavelet_levels,
            'wavelet_families': self.wavelet_families,
            'use_adaptive_basis': self.use_adaptive_basis,
            'use_wavelet_attention': self.use_wavelet_attention,
            'use_fourier_convolution': self.use_fourier_convolution,
            'use_frequency_domain_attention': self.use_frequency_domain_attention,
            'use_hrfevo': self.use_hrfevo,
            'evolution_generations': self.evolution_generations,
            'population_size': self.population_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict) 