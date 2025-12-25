"""
Layer Normalization implementation.

Implements layer normalization from scratch using PyTorch tensors.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Normalizes inputs across the feature dimension with learnable
    scale and shift parameters.
    
    Args:
        normalized_shape: Shape of the normalized dimension
        eps: Small value to prevent division by zero
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # TODO: Initialize learnable parameters (scale and bias)
        # TODO: Store normalized_shape and eps
        
    def forward(self, x):
        """
        Apply layer normalization to input tensor.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # TODO: Compute mean across the normalized dimension
        # TODO: Compute variance across the normalized dimension
        # TODO: Normalize: (x - mean) / sqrt(variance + eps)
        # TODO: Apply scale and shift parameters
        # TODO: Return normalized tensor
        pass

