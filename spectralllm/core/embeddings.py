"""
Spectral and Hybrid Embedding Modules

This module implements spectral token embeddings that represent tokens as 
superpositions of frequency components, enabling more efficient signal processing.
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


class SpectralEmbedding(nn.Module):
    """
    Embedding layer that represents tokens as superpositions of frequency components.
    Instead of learning a direct embedding vector, this learns amplitudes and phases for
    different frequency components to create more efficient embeddings.
    """
    def __init__(self, vocab_size: int, embed_dim: int, harmonic_bases: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.harmonic_bases = harmonic_bases
        
        # Each token gets frequency amplitudes and phases
        self.frequency_amplitudes = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        self.frequency_phases = nn.Parameter(torch.randn(vocab_size, harmonic_bases))
        
        # Generate frequency bases (fixed)
        self.register_buffer('frequencies', 
                           torch.linspace(0.1, math.pi, harmonic_bases))
        
        # Statistics tracking for analysis
        self.register_buffer('embedding_power', torch.zeros(embed_dim))
        self.register_buffer('embedding_usage', torch.zeros(vocab_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spectral embeddings for input tokens.
        
        Args:
            x: Token indices [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length = x.shape
        
        # Get amplitudes and phases for each token
        amplitudes = self.frequency_amplitudes[x]  # [batch_size, seq_length, harmonic_bases]
        phases = self.frequency_phases[x]  # [batch_size, seq_length, harmonic_bases]
        
        # Generate time points for embedding dimension
        t = torch.linspace(0, 1, self.embed_dim, device=x.device)
        t = t.view(1, 1, 1, self.embed_dim)  # [1, 1, 1, embed_dim]
        
        # Each frequency contributes across the embedding dimension
        frequencies = self.frequencies.view(1, 1, self.harmonic_bases, 1)  # [1, 1, harmonic_bases, 1]
        amplitudes = amplitudes.unsqueeze(-1)  # [batch, seq_len, harmonic_bases, 1]
        phases = phases.unsqueeze(-1)  # [batch, seq_len, harmonic_bases, 1]
        
        # Generate embeddings as superposition of harmonics
        signal = amplitudes * torch.sin(2 * math.pi * frequencies * t + phases)
        embeddings = signal.sum(dim=2)  # [batch, seq_len, embed_dim]
        
        # Update statistics for analysis (in training mode)
        if self.training:
            with torch.no_grad():
                # Track frequency power distribution
                power = torch.mean(torch.abs(embeddings) ** 2, dim=(0, 1))
                self.embedding_power = 0.99 * self.embedding_power + 0.01 * power
                
                # Track token usage
                tokens_used, _ = torch.unique(x, return_counts=True)
                self.embedding_usage[tokens_used] += 1
        
        return embeddings
    
    def get_spectral_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about the spectral embeddings"""
        # Compute power spectrum of the embeddings
        with torch.no_grad():
            # Average amplitudes across vocabulary
            avg_amplitudes = torch.mean(self.frequency_amplitudes, dim=0)
            # Average phases across vocabulary
            avg_phases = torch.mean(torch.abs(self.frequency_phases), dim=0)
            
            # Most frequently used tokens
            top_tokens = torch.argsort(self.embedding_usage, descending=True)[:10]
            
            stats = {
                'power_spectrum': self.embedding_power.cpu(),
                'frequency_amplitudes': avg_amplitudes.cpu(),
                'frequency_phases': avg_phases.cpu(),
                'token_usage': self.embedding_usage.cpu(),
                'top_tokens': top_tokens.cpu()
            }
        
        return stats


class HybridEmbedding(nn.Module):
    """
    Combines traditional token embeddings with spectral embeddings.
    This allows for a smooth transition from standard to spectral representations.
    """
    def __init__(self, vocab_size: int, embed_dim: int, 
                 harmonic_bases: int = 16, spectral_ratio: float = 0.5):
        super().__init__()
        self.spectral_ratio = spectral_ratio
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Standard embedding
        self.standard_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Spectral embedding
        self.spectral_embedding = SpectralEmbedding(vocab_size, embed_dim, harmonic_bases)
        
        # Learnable mixing parameter (starts with specified ratio)
        self.mixing_param = nn.Parameter(torch.tensor([spectral_ratio]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate hybrid embeddings for input tokens.
        
        Args:
            x: Token indices [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, embed_dim]
        """
        # Get both embedding types
        standard_embeds = self.standard_embedding(x)
        spectral_embeds = self.spectral_embedding(x)
        
        # Mix the embeddings with learnable parameter (constrained to [0,1])
        alpha = torch.sigmoid(self.mixing_param)
        embeddings = (1 - alpha) * standard_embeds + alpha * spectral_embeds
        
        return embeddings
    
    def get_mixing_ratio(self) -> float:
        """Get the current mixing ratio between standard and spectral embeddings"""
        with torch.no_grad():
            return torch.sigmoid(self.mixing_param).item()


class SignalPositionalEncoding(nn.Module):
    """
    Positional encoding using basis of sinusoidal signals at different frequencies.
    Extends the standard sinusoidal encoding with learnable parameters.
    """
    def __init__(self, max_seq_length: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create standard sinusoidal encoding as starting point
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Make parameters learnable with initial sinusoidal values
        self.pe = nn.Parameter(pe.unsqueeze(0))
        
        # Frequency modulation parameters
        self.freq_mod = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_length, embed_dim]
            
        Returns:
            Embeddings with positional information
        """
        seq_length = x.size(1)
        
        # Apply frequency modulation and phase shift
        pos_encoding = self.pe[:, :seq_length] * self.freq_mod + self.phase_shift
        
        return x + pos_encoding 