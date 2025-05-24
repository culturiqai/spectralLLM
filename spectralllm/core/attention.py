"""
Wavelet-based Attention and Transformer Blocks

This module implements attention mechanisms that operate in the wavelet domain
for efficient multi-resolution analysis of language sequences.
"""

import math
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for PyWavelets availability
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False


class WaveletTransform(nn.Module):
    """
    Implements wavelet transform for multi-resolution analysis.
    Falls back to FFT-based approximation if PyWavelets is not available.
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 mode: str = 'reflect'):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        # Use FFT by default for more robust training
        self.using_pywt = False  # Changed from WAVELET_AVAILABLE
        
        # Always initialize FFT frequencies for fallback
        self.register_buffer('frequencies', torch.linspace(0.1, math.pi, 64))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform wavelet decomposition.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Tuple of (approximation, [detail_coefficients])
        """
        if self.using_pywt:
            return self.pywt_forward(x)
        else:
            return self.fft_forward(x)
    
    def pywt_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """PyWavelets-based forward transform"""
        batch_size, seq_length, embed_dim = x.shape
        
        # Process each embedding dimension separately
        approx_list = []
        details_list = [[] for _ in range(self.levels)]
        
        for dim in range(embed_dim):
            for batch in range(batch_size):
                signal = x[batch, :, dim].detach().cpu().numpy()
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet_type, 
                                    level=self.levels, mode=self.mode)
                
                if dim == 0 and batch == 0:
                    # Initialize tensors based on coefficient sizes
                    approx_shape = (batch_size, len(coeffs[0]), embed_dim)
                    approx = torch.zeros(approx_shape, device=x.device)
                    details = [torch.zeros((batch_size, len(coeffs[i+1]), embed_dim), 
                                         device=x.device) for i in range(self.levels)]
                
                # Store coefficients
                approx[batch, :, dim] = torch.tensor(coeffs[0], device=x.device)
                for level in range(self.levels):
                    details[level][batch, :, dim] = torch.tensor(coeffs[level+1], device=x.device)
        
        return approx, details
    
    def fft_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """FFT-based approximation of wavelet transform"""
        batch_size, seq_length, embed_dim = x.shape
        
        # Take FFT
        X = torch.fft.fft(x, dim=1)
        
        # Split into frequency bands (approximating wavelet levels)
        freqs = torch.fft.fftfreq(seq_length, device=x.device)
        
        # Low frequencies (approximation)
        low_freq_mask = torch.abs(freqs) < 0.1
        approx_freq = X.clone()
        approx_freq[:, ~low_freq_mask, :] = 0
        approx = torch.fft.ifft(approx_freq, dim=1).real
        
        # High frequency details at different scales
        details = []
        for level in range(self.levels):
            scale = 2 ** level
            freq_low = 0.1 / scale
            freq_high = 0.1 / (scale / 2) if level > 0 else 1.0
            
            band_mask = (torch.abs(freqs) >= freq_low) & (torch.abs(freqs) < freq_high)
            detail_freq = X.clone()
            detail_freq[:, ~band_mask, :] = 0
            detail = torch.fft.ifft(detail_freq, dim=1).real
            details.append(detail)
        
        return approx, details


class FrequencyDomainAttention(nn.Module):
    """
    Attention mechanism that operates in the frequency domain.
    Processes queries, keys, and values using FFT for O(n log n) complexity.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Frequency domain attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Convert to frequency domain using FFT
        q_freq = torch.fft.fft(q, dim=-2)
        k_freq = torch.fft.fft(k, dim=-2)
        v_freq = torch.fft.fft(v, dim=-2)
        
        # Frequency domain "attention" using element-wise operations
        # This approximates attention in the frequency domain
        attn_freq = q_freq * torch.conj(k_freq)
        attn_weights = torch.abs(attn_freq)
        
        # Apply softmax in frequency domain (approximation)
        attn_weights = F.softmax(attn_weights, dim=-2)
        
        # Apply attention to values in frequency domain
        out_freq = attn_weights * v_freq
        
        # Convert back to time domain
        out = torch.fft.ifft(out_freq, dim=-2).real
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.out_proj(out)
        
        return self.dropout_layer(out)


class FourierConvolutionAttention(nn.Module):
    """
    Fourier Convolution-based Attention mechanism.
    
    This attention mechanism operates by performing convolution in the frequency domain,
    providing O(n log n) complexity while maintaining the mathematical properties
    of convolution operations.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable frequency filters
        self.freq_filter = nn.Parameter(torch.randn(num_heads, embed_dim // num_heads) * 0.1)
        self.phase_shift = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def _ensure_compatible_sizes(self, tensor1, tensor2, dim=1):
        """Ensure two tensors have compatible sizes for operations."""
        size1 = tensor1.size(dim)
        size2 = tensor2.size(dim)
        
        if size1 == size2:
            return tensor1, tensor2
        
        # Pad the smaller tensor
        max_size = max(size1, size2)
        
        def pad_tensor(tensor, target_size, pad_dim):
            padding = [0] * (2 * tensor.dim())
            padding[2 * (tensor.dim() - 1 - pad_dim)] = target_size - tensor.size(pad_dim)
            return F.pad(tensor, padding)
        
        if size1 < max_size:
            tensor1 = pad_tensor(tensor1, max_size, dim)
        if size2 < max_size:
            tensor2 = pad_tensor(tensor2, max_size, dim)
            
        return tensor1, tensor2
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Fourier convolution attention.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, embed]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, heads, seq, head_dim]
        
        # Convert to frequency domain
        q_fft = torch.fft.rfft(q, dim=2)  # [batch, heads, freq_bins, head_dim]
        k_fft = torch.fft.rfft(k, dim=2)
        v_fft = torch.fft.rfft(v, dim=2)
        
        # Ensure compatible sizes
        q_fft, k_fft = self._ensure_compatible_sizes(q_fft, k_fft, dim=2)
        q_fft, v_fft = self._ensure_compatible_sizes(q_fft, v_fft, dim=2)
        
        # Apply learnable frequency filters
        freq_bins = q_fft.size(2)
        filter_expanded = self.freq_filter.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
        phase_expanded = self.phase_shift.unsqueeze(0).unsqueeze(2)
        
        # Modulate queries and keys with frequency filters
        freq_modulation = torch.exp(1j * phase_expanded) * filter_expanded
        if freq_modulation.size(2) != freq_bins:
            # Simply repeat the filter to match frequency bins
            repeat_factor = max(1, freq_bins // freq_modulation.size(2) + 1)
            freq_modulation = freq_modulation.repeat(1, 1, repeat_factor, 1)[:, :, :freq_bins, :]
        
        q_filtered = q_fft * freq_modulation
        k_filtered = k_fft * freq_modulation
        
        # Convolution in frequency domain (element-wise multiplication)
        # This implements circular convolution between queries and keys
        conv_qk = q_filtered * torch.conj(k_filtered)  # [batch, heads, freq_bins, head_dim]
        
        # Compute attention weights from convolution result
        attn_weights_fft = torch.abs(conv_qk)  # Take magnitude
        
        # Apply softmax in frequency domain (approximate)
        # Convert back to time domain for softmax
        attn_weights_time = torch.fft.irfft(attn_weights_fft, n=seq_length, dim=2)
        
        # Apply causal mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(3)  # [batch, 1, seq, 1]
            attn_weights_time = attn_weights_time.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax attention
        attn_weights = F.softmax(attn_weights_time, dim=2)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Convert attention weights back to frequency domain
        attn_weights_fft = torch.fft.rfft(attn_weights, dim=2)
        
        # Apply attention to values in frequency domain
        # This performs convolution between attention weights and values
        output_fft = attn_weights_fft * v_fft
        
        # Convert back to time domain
        output = torch.fft.irfft(output_fft, n=seq_length, dim=2)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(output)
        
        return self.layer_norm(x + self.dropout_layer(output))
    
    def get_complexity_ratio(self) -> float:
        """Get the complexity ratio compared to standard attention."""
        return math.log2(max(32, self.embed_dim)) / self.embed_dim
    
    def get_associativity_metrics(self) -> Dict[str, float]:
        """Get metrics related to the associativity of the convolution operation."""
        return {
            'frequency_sparsity': torch.mean(torch.abs(self.freq_filter)).item(),
            'phase_variance': torch.var(self.phase_shift).item(),
            'filter_norm': torch.norm(self.freq_filter).item()
        }


class WaveletAttention(nn.Module):
    """
    Multi-head attention using wavelet decomposition for multi-resolution analysis.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                wavelet_type: str = 'db4', levels: int = 3,
                dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.levels = levels
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
        
        # Wavelet transform
        self.wavelet = WaveletTransform(wavelet_type, levels)
        
        # Attention for approximation coefficients
        self.approx_attention = FrequencyDomainAttention(embed_dim, num_heads, dropout)
        
        # Attention for detail coefficients at each level
        self.detail_attentions = nn.ModuleList([
            FrequencyDomainAttention(embed_dim, num_heads, dropout)
            for _ in range(levels)
        ])
        
        # Reconstruction projection
        self.reconstruction_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Wavelet attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        original_shape = x.shape
        
        # Wavelet decomposition
        approx, details = self.wavelet(x)
        
        # Apply attention to approximation
        approx_out = self.approx_attention(approx, mask)
        
        # Apply attention to detail coefficients
        detail_outs = []
        for i, detail in enumerate(details):
            if detail.numel() > 0:  # Check if detail coefficients exist
                detail_out = self.detail_attentions[i](detail, mask)
                detail_outs.append(detail_out)
            else:
                detail_outs.append(detail)
        
        # Reconstruct to original sequence length
        # For simplicity, we'll interpolate the approximation to match input size
        if approx_out.shape[1] != original_shape[1]:
            approx_upsampled = F.interpolate(
                approx_out.transpose(1, 2), 
                size=original_shape[1], 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            approx_upsampled = approx_out
        
        # Start with approximation as base
        output = approx_upsampled
        
        # Add detail contributions (simplified reconstruction)
        for detail_out in detail_outs:
            if detail_out.numel() > 0:
                # Upsample detail to match sequence length
                if detail_out.shape[1] != original_shape[1]:
                    detail_upsampled = F.interpolate(
                        detail_out.transpose(1, 2), 
                        size=original_shape[1], 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    detail_upsampled = detail_out
                
                output = output + 0.1 * detail_upsampled  # Weighted combination
        
        # Ensure output matches input shape exactly
        output = output.view(original_shape)
        
        return self.reconstruction_proj(output)


class SpectralFeedForward(nn.Module):
    """
    Feed-forward network that operates partially in the frequency domain.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Standard FFN components
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Frequency domain processing
        self.freq_linear = nn.Linear(embed_dim, embed_dim)
        self.freq_mixing = nn.Parameter(torch.tensor(0.1))  # Small contribution initially
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral feed-forward processing.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        # Standard FFN path
        x_ffn = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        
        # Frequency domain path
        x_freq = torch.fft.fft(x, dim=1)
        x_freq_processed = self.freq_linear(x_freq.real) + 1j * self.freq_linear(x_freq.imag)
        x_freq_out = torch.fft.ifft(x_freq_processed, dim=1).real
        
        # Combine paths
        mixing_weight = torch.sigmoid(self.freq_mixing)
        return (1 - mixing_weight) * x_ffn + mixing_weight * x_freq_out


class WaveletTransformerBlock(nn.Module):
    """
    A transformer block that uses wavelet-based attention for multi-resolution processing.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, 
                wavelet_type: str = 'db4', levels: int = 3,
                dropout: float = 0.1):
        super().__init__()
        
        self.wavelet_attn = WaveletAttention(
            embed_dim, num_heads, wavelet_type, levels, dropout
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = SpectralFeedForward(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through wavelet transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_dim]
        """
        # Wavelet attention with residual connection
        attn_output = self.wavelet_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Spectral FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x 