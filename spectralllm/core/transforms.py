"""
Advanced Wavelet Transforms and Basis Selection
==============================================

This module implements advanced transformation techniques including stochastic
approximation and adaptive basis selection for optimal performance.
"""

import math
import random
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check for PyWavelets availability
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

from .attention import WaveletTransform


class EnhancedWaveletTransform(nn.Module):
    """
    Enhanced wavelet transform with better error handling and device management.
    Integrates improvements from MPS optimizations for more robust training.
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 mode: str = 'reflect', use_learned: bool = True):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        self.use_learned = use_learned
        
        # Initialize filter parameters
        self.initialize_filters()
        
        # Set flag for using optimized implementation
        self.using_optimized = True
        
        # Performance tracking
        self.register_buffer('error_counts', torch.zeros(1))
        self.register_buffer('fallback_counts', torch.zeros(1))
    
    def initialize_filters(self):
        """Initialize wavelet filter coefficients with fallback support"""
        # Pre-defined coefficients for common wavelets
        wavelet_coeffs = {
            'db1': {'dec_lo': [0.7071067811865476, 0.7071067811865476],
                    'dec_hi': [0.7071067811865476, -0.7071067811865476]},
            'db2': {'dec_lo': [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341],
                    'dec_hi': [-0.4829629131445341, 0.8365163037378079, -0.2241438680420134, -0.1294095225512604]},
            'db4': {'dec_lo': [0.0107772507, 0.0328830116, 0.0308413818, -0.1870348117, 
                             -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133],
                    'dec_hi': [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694, 
                              0.1870348117, 0.0308413818, -0.0328830116, 0.0107772507]},
        }
        
        # Use db4 as default if wavelet type not found
        coeffs = wavelet_coeffs.get(self.wavelet_type, wavelet_coeffs['db4'])
        
        # Create tensor filter banks with proper shape for group convolution
        if self.use_learned:
            # Learnable filters initialized with standard coefficients
            self.dec_lo = nn.Parameter(torch.tensor(coeffs['dec_lo'], dtype=torch.float32).view(1, 1, -1))
            self.dec_hi = nn.Parameter(torch.tensor(coeffs['dec_hi'], dtype=torch.float32).view(1, 1, -1))
            # Add reconstruction filters (time-reversed decomposition filters)
            self.rec_lo = nn.Parameter(torch.tensor(coeffs['dec_lo'][::-1], dtype=torch.float32).view(1, 1, -1))
            self.rec_hi = nn.Parameter(torch.tensor(coeffs['dec_hi'][::-1], dtype=torch.float32).view(1, 1, -1))
        else:
            # Fixed filters
            self.register_buffer('dec_lo', torch.tensor(coeffs['dec_lo'], dtype=torch.float32).view(1, 1, -1))
            self.register_buffer('dec_hi', torch.tensor(coeffs['dec_hi'], dtype=torch.float32).view(1, 1, -1))
            # Add reconstruction filters (time-reversed decomposition filters)
            self.register_buffer('rec_lo', torch.tensor(coeffs['dec_lo'][::-1], dtype=torch.float32).view(1, 1, -1))
            self.register_buffer('rec_hi', torch.tensor(coeffs['dec_hi'][::-1], dtype=torch.float32).view(1, 1, -1))
        
        # Pre-calculated filter lengths for performance
        self.filter_len = len(coeffs['dec_lo'])
        
        # FFT filters for frequency domain operations
        if self.use_learned:
            self.low_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.7)
            self.high_pass = nn.Parameter(torch.ones(1, 1, 32) * 0.3)
        else:
            self.register_buffer('low_pass', torch.ones(1, 1, 32) * 0.7)
            self.register_buffer('high_pass', torch.ones(1, 1, 32) * 0.3)
    
    def _ensure_coefficient_shapes(self, approx: torch.Tensor, details: List[torch.Tensor], 
                                 expected_sizes: List[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Ensure coefficients have consistent shapes to prevent mismatches
        
        Args:
            approx: Approximation coefficients [batch_size, seq_length, embed_dim]
            details: List of detail coefficients
            expected_sizes: Expected sequence lengths for each level
            
        Returns:
            Corrected approx and details with consistent shapes
        """
        # If no expected sizes provided, maintain current sizes
        if expected_sizes is None:
            if not details:
                return approx, details
                
            # Use the size of the first detail coefficient as reference
            expected_sizes = [d.size(1) for d in details]
            expected_sizes.append(approx.size(1))
        
        # Ensure details have expected sizes
        corrected_details = []
        for idx, detail in enumerate(details):
            if idx >= len(expected_sizes):
                # More details than expected, just append as is
                corrected_details.append(detail)
                continue
                
            expected_size = expected_sizes[idx]
            actual_size = detail.size(1)
            
            if actual_size != expected_size:
                # Resize using interpolation for better numerical stability
                resized_detail = torch.nn.functional.interpolate(
                    detail.transpose(1, 2),  # [batch, embed, seq]
                    size=expected_size,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # back to [batch, seq, embed]
                corrected_details.append(resized_detail)
            else:
                corrected_details.append(detail)
        
        # Ensure approx has expected size
        if expected_sizes and approx.size(1) != expected_sizes[-1]:
            approx = torch.nn.functional.interpolate(
                approx.transpose(1, 2),  # [batch, embed, seq]
                size=expected_sizes[-1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # back to [batch, seq, embed]
        
        return approx, corrected_details
    
    def _pad_signal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad signal for convolution with proper padding based on filter length
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Padded tensor [batch_size, seq_length+pad, embed_dim]
        """
        # Calculate padding based on filter length to ensure proper coefficient sizes
        pad_len = self.filter_len - 1
        
        # Apply symmetric padding (better for wavelets) or zero padding
        if self.mode == 'reflect':
            # Use symmetric padding which better preserves frequency characteristics
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='reflect')
        elif self.mode == 'symmetric':
            # Symmetric padding is similar to reflect but handles boundaries differently
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='replicate')
        else:
            # Zero padding
            padded = torch.nn.functional.pad(x, (0, 0, pad_len//2, pad_len//2), mode='constant')
        
        return padded
    
    def _conv1d_enhanced(self, x: torch.Tensor, filter_bank: torch.Tensor) -> torch.Tensor:
        """
        Enhanced 1D convolution with better device management and error handling
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            filter_bank: Filter coefficients
            
        Returns:
            Convolution result [batch_size, seq_length, embed_dim]
        """
        try:
            # Ensure inputs are on the correct device
            if filter_bank.device != x.device:
                filter_bank = filter_bank.to(x.device)
            
            # Reshape for conv1d (batch, channels, length)
            batch_size, seq_length, embed_dim = x.shape
            x_reshaped = x.transpose(1, 2)  # [batch, embed, seq]
            
            # Ensure filter_bank has correct dimensions
            if filter_bank.dim() == 1:
                # If 1D tensor, add two dimensions
                filter_bank = filter_bank.unsqueeze(0).unsqueeze(0)
            elif filter_bank.dim() == 2:
                # If 2D tensor, add one dimension
                filter_bank = filter_bank.unsqueeze(0)
            
            # Get filter length for manual padding calculation
            filter_length = filter_bank.size(-1)
            
            # Manual padding calculation for better device compatibility
            total_pad = filter_length - 1
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            
            # Apply manual padding before convolution
            x_padded = torch.nn.functional.pad(x_reshaped, (pad_left, pad_right), mode='reflect')
            
            # Safely expand to [embed_dim, 1, filter_length]
            filter_expanded = filter_bank.expand(embed_dim, 1, -1)
            
            # Apply convolution with no padding (since we manually padded)
            output = torch.nn.functional.conv1d(
                x_padded,
                filter_expanded,
                groups=embed_dim,
                padding=0
            )
            
            # Reshape back to original format
            output = output.transpose(1, 2)  # [batch, seq, embed]
            
            return output
        
        except Exception as e:
            # Log error and fallback
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                print(f"Enhanced convolution failed: {e}. Falling back to standard method.")
            self.error_counts[0] += 1
            
            # Fallback to standard convolution
            return self._fallback_conv1d(x, filter_bank)
    
    def _fallback_conv1d(self, x: torch.Tensor, filter_bank: torch.Tensor) -> torch.Tensor:
        """Fallback convolution method using standard operations"""
        # Simple fallback using matrix multiplication
        batch_size, seq_length, embed_dim = x.shape
        
        # Convert filter to appropriate shape
        if filter_bank.dim() > 1:
            filter_1d = filter_bank.view(-1)
        else:
            filter_1d = filter_bank
        
        # Apply simple filtering (this is a basic fallback)
        output = x.clone()
        filter_size = len(filter_1d)
        
        for i in range(filter_size, seq_length - filter_size):
            for j, coeff in enumerate(filter_1d):
                if j < seq_length - i:
                    output[:, i, :] += coeff * x[:, i + j, :]
        
        return output
    
    def forward_enhanced(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Enhanced forward wavelet transform with robust error handling
        
        Args:
            x: Input signal [batch_size, seq_length, embed_dim]
            
        Returns:
            approximation, detail_coefficients
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Store original sequence length for reconstruction
        self.original_seq_length = seq_length
        
        # Check if signal length is sufficient for the requested level
        min_length = 2 ** self.levels
        if seq_length < min_length:
            actual_levels = max(1, int(math.log2(seq_length)))
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                print(f"Warning: Sequence length {seq_length} is too short for {self.levels} levels. Using {actual_levels} levels instead.")
            levels = actual_levels
        else:
            levels = self.levels
        
        # For very short sequences, use FFT-based method
        if seq_length < 16:
            return self.fft_forward(x)
        
        try:
            # Validate filter dimensions early
            if self.dec_lo.dim() == 1:
                self.dec_lo = self.dec_lo.unsqueeze(0).unsqueeze(0)
                self.dec_hi = self.dec_hi.unsqueeze(0).unsqueeze(0)
            elif self.dec_lo.dim() == 2:
                self.dec_lo = self.dec_lo.unsqueeze(0)
                self.dec_hi = self.dec_hi.unsqueeze(0)
            
            # Calculate expected output sizes for each level
            expected_sizes = []
            current_size = seq_length
            for _ in range(levels):
                next_size = (current_size + self.filter_len - 1) // 2
                expected_sizes.append(next_size)
                current_size = next_size
            
            # Create lists for coefficients
            detail_coeffs = []
            approx = x
            
            # Apply wavelet transform for each level
            for level in range(levels):
                # Pad signal for convolution
                padded = self._pad_signal(approx)
                
                # Apply low-pass and high-pass filters
                approx_coeffs = self._conv1d_enhanced(padded, self.dec_lo)
                detail = self._conv1d_enhanced(padded, self.dec_hi)
                
                # Downsample
                approx_coeffs = approx_coeffs[:, ::2, :]
                detail = detail[:, ::2, :]
                
                # Adjust sizes to match expected dimensions
                expected_size = expected_sizes[level]
                if approx_coeffs.size(1) != expected_size:
                    if approx_coeffs.size(1) > expected_size:
                        approx_coeffs = approx_coeffs[:, :expected_size, :]
                    else:
                        pad_size = expected_size - approx_coeffs.size(1)
                        approx_coeffs = torch.nn.functional.pad(approx_coeffs, (0, 0, 0, pad_size), mode='constant')
                
                if detail.size(1) != expected_size:
                    if detail.size(1) > expected_size:
                        detail = detail[:, :expected_size, :]
                    else:
                        pad_size = expected_size - detail.size(1)
                        detail = torch.nn.functional.pad(detail, (0, 0, 0, pad_size), mode='constant')
                
                # Save detail coefficients
                detail_coeffs.append(detail)
                
                # Use approximation for next level
                approx = approx_coeffs
                
                # Check if we can continue
                if approx.size(1) < 2:
                    break
            
            # Ensure consistent coefficient shapes before returning
            approx, detail_coeffs = self._ensure_coefficient_shapes(approx, detail_coeffs, expected_sizes[-1:])
            
            # Store expected sizes for reconstruction
            self.expected_sizes = expected_sizes
            
            return approx, detail_coeffs
        
        except Exception as e:
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                print(f"Enhanced wavelet transform failed, falling back to FFT: {str(e)}")
            self.fallback_counts[0] += 1
            return self.fft_forward(x)
    
    def fft_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fallback FFT-based forward transform implementation"""
        batch_size, seq_length, embed_dim = x.shape
        
        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=1)
        fft_size = x_fft.size(1)
        
        # Create results based on levels
        detail_tensors = []
        approx = x.clone()
        
        # Apply filters for each level
        for level in range(self.levels):
            # Calculate frequency bands for this level
            band_size = fft_size // (2 ** (level + 1))
            if band_size < 1:
                break
                
            # Apply low-pass and high-pass filters
            x_fft_lowpass = x_fft.clone()
            x_fft_highpass = x_fft.clone()
            
            # Shape filters for proper broadcasting
            filter_idx = level % self.low_pass.size(2)
            low_filter = torch.sigmoid(self.low_pass[:, :, filter_idx])
            high_filter = torch.sigmoid(self.high_pass[:, :, filter_idx])
            
            # Apply filters
            x_fft_lowpass[:, band_size:] = 0
            x_fft_highpass[:, :band_size] = 0
            
            # Apply learnable filter shapes
            x_fft_lowpass = x_fft_lowpass * low_filter.unsqueeze(-1)
            x_fft_highpass = x_fft_highpass * high_filter.unsqueeze(-1)
            
            # Inverse FFT to get coefficients
            approx = torch.fft.irfft(x_fft_lowpass, n=seq_length, dim=1)
            detail = torch.fft.irfft(x_fft_highpass, n=seq_length, dim=1)
            
            # Save detail coefficients
            detail_tensors.append(detail)
            
            # Update FFT for next level
            x_fft = x_fft_lowpass
        
        return approx, detail_tensors
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with enhanced error handling"""
        try:
            # Disable debug output if environment variable is set
            if os.environ.get('SPECTRALLLM_DISABLE_DEBUG', '0') == '1':
                return self.forward_enhanced(x)
                
            # Try enhanced version first
            return self.forward_enhanced(x)
        except Exception as e:
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                print(f"Enhanced wavelet transform failed, falling back to FFT: {str(e)}")
            # Fallback to FFT method
            return self.fft_forward(x)
    
    def inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """Apply inverse wavelet transform with enhanced error handling"""
        try:
            # Try to use the original wavelet transform's inverse method
            base_wavelet = WaveletTransform(self.wavelet_type, self.levels, self.mode)
            return base_wavelet.inverse(approx, details)
        except Exception as e:
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                print(f"Inverse wavelet transform failed, using FFT fallback: {str(e)}")
            # Fallback to FFT-based inverse
            return self.fft_inverse(approx, details)
    
    def fft_inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """Fallback FFT-based inverse transform implementation"""
        batch_size, seq_length, embed_dim = approx.shape
        
        # Convert approximation to frequency domain
        x_fft = torch.fft.rfft(approx, dim=1)
        
        # Add detail coefficients in frequency domain
        for detail in details:
            # Ensure detail has the same sequence length as approx
            if detail.size(1) != seq_length:
                detail = torch.nn.functional.interpolate(
                    detail.transpose(1, 2),  # [batch, embed, seq]
                    size=seq_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # back to [batch, seq, embed]
            
            detail_fft = torch.fft.rfft(detail, dim=1)
            
            # Ensure FFT outputs have the same size
            min_freq_size = min(x_fft.size(1), detail_fft.size(1))
            x_fft_trim = x_fft[:, :min_freq_size, :]
            detail_fft_trim = detail_fft[:, :min_freq_size, :]
            
            # Add the frequency components
            x_fft = x_fft_trim + detail_fft_trim
        
        # Inverse FFT to get reconstructed signal
        output = torch.fft.irfft(x_fft, n=seq_length, dim=1)
        
        return output
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'error_count': self.error_counts.item(),
            'fallback_count': self.fallback_counts.item(),
            'total_calls': self.error_counts.item() + self.fallback_counts.item(),
            'success_rate': 1.0 - (self.fallback_counts.item() / max(1, self.error_counts.item() + self.fallback_counts.item()))
        }


class StochasticTransform(nn.Module):
    """
    Implements stochastic approximation for wavelet transforms as suggested by Dr. Tao,
    reducing complexity from O(n log n) to O(m log m) where m << n.
    """
    def __init__(self, wavelet_type: str = 'db4', levels: int = 3, 
                 sampling_ratio: float = 0.1, min_samples: int = 32):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        
        # Calculate expected error bound based on sampling theory: O(1/√m)
        self.register_buffer('error_bound', None)
        self.register_buffer('actual_errors', None)
    
    def compute_error_bound(self, n_samples: int) -> float:
        """
        Compute theoretical error bound based on stochastic approximation theory.
        
        Args:
            n_samples: Number of samples used
            
        Returns:
            Theoretical error bound O(1/√m)
        """
        return 1.0 / np.sqrt(n_samples)
    
    def forward_stochastic(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """
        Apply stochastic approximation of wavelet transform.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Approximation coefficients, detail coefficients, and error metrics
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Determine number of samples
        n_samples = max(self.min_samples, int(self.sampling_ratio * seq_length))
        n_samples = min(n_samples, seq_length)  # Can't have more samples than sequence length
        
        # For comparison, also compute full transform on a subset of the batch
        # to estimate actual error
        comparison_batch_idx = 0
        full_approx, full_details = None, None
        
        if self.training:
            # Use standard wavelet transform for the first item in batch
            wavelet = WaveletTransform(self.wavelet_type, self.levels)
            full_approx, full_details = wavelet(x[comparison_batch_idx:comparison_batch_idx+1])
        
        # Randomly sample points from the sequence
        indices = torch.randperm(seq_length, device=x.device)[:n_samples]
        indices, _ = torch.sort(indices)  # Sort for coherent transform
        
        # Extract sampled points
        x_sampled = x[:, indices, :]
        
        # Apply transform to sampled points
        wavelet = WaveletTransform(self.wavelet_type, self.levels)
        sampled_approx, sampled_details = wavelet(x_sampled)
        
        # Scale up to original size (simple interpolation)
        approx_ratio = seq_length / sampled_approx.size(1)
        
        # Use interpolation to upsample approximation coefficients
        approx_upsampled = F.interpolate(
            sampled_approx.permute(0, 2, 1),  # [batch, embed, seq]
            size=seq_length,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # back to [batch, seq, embed]
        
        # Upsample detail coefficients
        details_upsampled = []
        for detail in sampled_details:
            detail_upsampled = F.interpolate(
                detail.permute(0, 2, 1),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            details_upsampled.append(detail_upsampled)
        
        # Calculate error metrics if full transform was computed
        error_metrics = {}
        if full_approx is not None:
            # Calculate approximation error
            approx_error = torch.norm(approx_upsampled[comparison_batch_idx] - full_approx[0]) / torch.norm(full_approx[0])
            error_metrics['approx_error'] = approx_error.item()
            
            # Calculate theoretical error bound
            error_bound = self.compute_error_bound(n_samples)
            error_metrics['error_bound'] = error_bound
            error_metrics['n_samples'] = n_samples
            error_metrics['sampling_ratio'] = self.sampling_ratio
            
            # Update error tracking buffers
            if self.error_bound is None:
                self.error_bound = torch.tensor([error_bound], device=x.device, dtype=torch.float32)
                self.actual_errors = torch.tensor([approx_error.item()], device=x.device)
            else:
                self.error_bound = torch.cat([self.error_bound, torch.tensor([error_bound], device=x.device, dtype=torch.float32)])
                self.actual_errors = torch.cat([self.actual_errors, torch.tensor([approx_error.item()], device=x.device)])
                
                # Keep only recent values (last 100)
                if len(self.error_bound) > 100:
                    self.error_bound = self.error_bound[-100:]
                    self.actual_errors = self.actual_errors[-100:]
        
        return approx_upsampled, details_upsampled, error_metrics
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Standard forward pass for compatibility."""
        approx, details, _ = self.forward_stochastic(x)
        return approx, details
    
    def inverse_stochastic(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply inverse transform from stochastically computed coefficients.
        
        Args:
            approx: Approximation coefficients
            details: Detail coefficients
            
        Returns:
            Reconstructed signal
        """
        # Use standard inverse wavelet transform
        wavelet = WaveletTransform(self.wavelet_type, self.levels)
        return wavelet.inverse(approx, details)
    
    def get_error_statistics(self) -> Dict:
        """
        Get statistics about the approximation error.
        
        Returns:
            Dictionary with error statistics
        """
        if self.error_bound is None or len(self.error_bound) == 0:
            return {
                'mean_error_bound': 0.0,
                'mean_actual_error': 0.0,
                'bound_to_actual_ratio': 1.0
            }
        
        mean_bound = self.error_bound.mean().item()
        mean_actual = self.actual_errors.mean().item()
        ratio = mean_bound / (mean_actual + 1e-8)
        
        return {
            'mean_error_bound': mean_bound,
            'mean_actual_error': mean_actual,
            'bound_to_actual_ratio': ratio
        }


class AdaptiveBasisSelection(nn.Module):
    """
    Implements adaptive selection between multiple wavelet basis families
    based on Dr. Tao's recommendation for optimal time-frequency trade-offs.
    """
    def __init__(self, embed_dim: int, families: List[str] = None):
        super().__init__()
        # Default wavelet families if not provided
        self.families = families or ['db4', 'sym4', 'dmey']
        self.embed_dim = embed_dim
        
        # Create transform for each wavelet family
        self.transforms = nn.ModuleDict({
            family: WaveletTransform(family, levels=3) 
            for family in self.families
        })
        
        # Context-based selection mechanism
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(self.families)),
            nn.Softmax(dim=-1)
        )
        
        # Learnable weights for combining outputs
        self.combination_weights = nn.Parameter(torch.ones(len(self.families)) / len(self.families))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multiple wavelet transforms and adaptively combine them.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_dim]
            
        Returns:
            Combined approximation and detail coefficients
        """
        # Compute context features for selection
        # Use mean pooling over sequence dimension
        context = x.mean(dim=1)
        selection_weights = self.selector(context).unsqueeze(1)  # [batch, 1, n_families]
        
        # Apply each wavelet transform
        all_approx = []
        all_details = []
        
        for i, (family, transform) in enumerate(self.transforms.items()):
            approx, details = transform(x)
            all_approx.append(approx)
            all_details.append(details)
        
        # Combine approximation coefficients
        stacked_approx = torch.stack(all_approx, dim=-1)  # [batch, seq, embed, n_families]
        weights_expanded = selection_weights.unsqueeze(2)  # [batch, 1, 1, n_families]
        combined_approx = torch.sum(stacked_approx * weights_expanded, dim=-1)
        
        # Combine detail coefficients at each level
        combined_details = []
        n_levels = len(all_details[0])
        
        for level in range(n_levels):
            level_details = [details[level] for details in all_details]
            stacked_details = torch.stack(level_details, dim=-1)
            combined_detail = torch.sum(stacked_details * weights_expanded, dim=-1)
            combined_details.append(combined_detail)
        
        return combined_approx, combined_details
    
    def inverse(self, approx: torch.Tensor, details: List[torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform using the first available transform.
        
        Args:
            approx: Approximation coefficients
            details: Detail coefficients
            
        Returns:
            Reconstructed signal
        """
        # Use the first transform for inverse (could be made adaptive too)
        first_transform = next(iter(self.transforms.values()))
        return first_transform.inverse(approx, details)
    
    def get_selection_statistics(self) -> Dict:
        """Get statistics about basis selection patterns."""
        with torch.no_grad():
            weights = torch.softmax(self.combination_weights, dim=0)
            return {
                'family_preferences': {family: weights[i].item() 
                                     for i, family in enumerate(self.families)},
                'selection_entropy': -torch.sum(weights * torch.log(weights + 1e-8)).item(),
                'dominant_family': self.families[torch.argmax(weights).item()]
            } 