"""
Core SignalLLM Model Implementation

This module contains the main SignalLLM model class with spectral embeddings,
wavelet-based attention, and frequency domain processing.
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .embeddings import SpectralEmbedding, HybridEmbedding, SignalPositionalEncoding
from .attention import WaveletTransformerBlock


class BasisFunction:
    """
    Represents a basis function for signal processing.
    """
    def __init__(self, type_name: str = 'db4', params: Optional[Dict] = None):
        self.type_name = type_name
        self.params = params or {}
        self.fitness = 0.0
        
        # Default parameters for different basis types
        if not self.params:
            if type_name.startswith('db'):
                self.params = {'order': int(type_name[2:]) if len(type_name) > 2 else 4}
            elif type_name.startswith('sym'):
                self.params = {'order': int(type_name[3:]) if len(type_name) > 3 else 4}
            elif type_name == 'dmey':
                self.params = {'order': 62}
            else:
                self.params = {'order': 4}
    
    def __repr__(self) -> str:
        return f"BasisFunction(type={self.type_name}, params={self.params}, fitness={self.fitness:.4f})"
    
    def mutate(self, mutation_rate: float = 0.1) -> 'BasisFunction':
        """Create a mutated copy of this basis function"""
        new_basis = self.copy()
        
        # Mutate type occasionally
        if torch.rand(1).item() < mutation_rate * 0.1:
            families = ['db4', 'db8', 'sym4', 'sym8', 'dmey']
            new_basis.type_name = torch.tensor(families)[torch.randint(0, len(families), (1,))].item()
        
        # Mutate parameters
        for key, value in new_basis.params.items():
            if isinstance(value, (int, float)) and torch.rand(1).item() < mutation_rate:
                if isinstance(value, int):
                    new_basis.params[key] = max(1, value + torch.randint(-2, 3, (1,)).item())
                else:
                    new_basis.params[key] = value * (1 + 0.1 * torch.randn(1).item())
        
        return new_basis
    
    def crossover(self, other: 'BasisFunction') -> 'BasisFunction':
        """Create offspring by crossing over with another basis function"""
        new_basis = BasisFunction()
        
        # Choose type from either parent
        new_basis.type_name = self.type_name if torch.rand(1).item() < 0.5 else other.type_name
        
        # Mix parameters
        all_keys = set(self.params.keys()) | set(other.params.keys())
        new_basis.params = {}
        
        for key in all_keys:
            if key in self.params and key in other.params:
                # Average numerical values
                if isinstance(self.params[key], (int, float)) and isinstance(other.params[key], (int, float)):
                    if isinstance(self.params[key], int) and isinstance(other.params[key], int):
                        new_basis.params[key] = int((self.params[key] + other.params[key]) // 2)
                    else:
                        new_basis.params[key] = (self.params[key] + other.params[key]) / 2
                else:
                    # Choose from either parent
                    new_basis.params[key] = self.params[key] if torch.rand(1).item() < 0.5 else other.params[key]
            elif key in self.params:
                new_basis.params[key] = self.params[key]
            else:
                new_basis.params[key] = other.params[key]
        
        return new_basis
    
    def copy(self) -> 'BasisFunction':
        """Create a deep copy of this basis function"""
        new_basis = BasisFunction(self.type_name)
        new_basis.params = {}
        
        for key, value in self.params.items():
            if isinstance(value, (list, tuple)):
                new_basis.params[key] = list(value)
            elif isinstance(value, torch.Tensor):
                new_basis.params[key] = value.clone()
            else:
                new_basis.params[key] = value
        
        new_basis.fitness = self.fitness
        return new_basis
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'type_name': self.type_name,
            'params': self.params,
            'fitness': self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BasisFunction':
        """Create from dictionary"""
        basis = cls(data['type_name'], data['params'])
        basis.fitness = data.get('fitness', 0.0)
        return basis


class SignalLLM(nn.Module):
    """
    Complete signal-based language model architecture.
    Uses spectral embeddings, wavelet attention, and frequency-domain processing.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings - can be spectral or hybrid
        if config.harmonic_bases > 0:
            self.use_hybrid = True
            self.token_embedding = HybridEmbedding(
                config.vocab_size, config.embed_dim, config.harmonic_bases)
        else:
            self.use_hybrid = False
            self.token_embedding = SpectralEmbedding(
                config.vocab_size, config.embed_dim, 16)
        
        # Positional encoding
        self.pos_encoding = SignalPositionalEncoding(config.max_seq_length, config.embed_dim)
        
        # Transformer blocks using wavelet attention
        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(
                config.embed_dim, config.num_heads, config.hidden_dim,
                config.wavelet_type, config.wavelet_levels, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Basis function currently in use
        self.current_basis = BasisFunction(config.wavelet_type)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights"""
        # Initialize projection weights with small values
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def update_basis_function(self, basis: BasisFunction) -> None:
        """
        Update the model to use a new basis function.
        
        Args:
            basis: New basis function to use
        """
        self.current_basis = basis.copy()
        
        # Update wavelet parameters in all transformer blocks
        for block in self.blocks:
            # Update wavelet transform in attention mechanism
            if hasattr(block, 'wavelet_attn') and hasattr(block.wavelet_attn, 'wavelet'):
                block.wavelet_attn.wavelet.wavelet_type = basis.type_name
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the signal-based language model.
        
        Args:
            x: Input token indices [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
            targets: Target token indices for loss computation [batch_size, seq_length]
            
        Returns:
            Dictionary containing:
            - logits: [batch_size, seq_length, vocab_size]
            - loss: scalar loss (if targets provided)
        """
        # Get embeddings
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)
        
        # Prepare output dictionary
        outputs = {'logits': logits}
        
        # Compute loss if targets provided
        if targets is not None:
            # Flatten for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Compute cross entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            outputs['loss'] = loss
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_complexity_info(self) -> Dict:
        """
        Get information about computational complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Collect operation count ratios from all attention layers
        attn_op_ratios = []
        for block in self.blocks:
            if hasattr(block, 'wavelet_attn') and hasattr(block.wavelet_attn, 'approx_attention'):
                if hasattr(block.wavelet_attn.approx_attention, 'get_complexity_ratio'):
                    ratio = block.wavelet_attn.approx_attention.get_complexity_ratio()
                    attn_op_ratios.append(ratio)
        
        # Get embedding info
        if self.use_hybrid:
            spectral_ratio = self.token_embedding.get_mixing_ratio()
        else:
            spectral_ratio = 1.0
        
        return {
            'attention_op_ratios': attn_op_ratios,
            'spectral_embedding_ratio': spectral_ratio,
            'parameter_count': self.count_parameters()
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[torch.device] = None):
        """Load a pre-trained model from checkpoint"""
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(model_path, map_location=device)
        config = Config.from_dict(checkpoint['config'])
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        
        return model
    
    def generate(self, prompt: str, tokenizer, max_length: int = 100, 
                temperature: float = 1.0, device: Optional[torch.device] = None) -> str:
        """Generate text from a prompt"""
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self(input_ids)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token or max length
                if input_ids.shape[1] >= self.config.max_seq_length:
                    break
        
        # Decode generated sequence
        generated_tokens = input_ids[0].cpu().tolist()
        return tokenizer.decode(generated_tokens) 