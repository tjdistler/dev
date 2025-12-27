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
        normalized_shape: Size of the normalized dimension (int).
            For GPT-1, this is the embedding dimension (e.g., 768).
        eps: Small value added to variance to prevent division by zero.
            Default: 1e-5
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # Initialize learnable parameters: scale (gamma) and bias (beta)
        # Scale starts at 1.0, bias starts at 0.0 (identity transformation initially)
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Store configuration parameters
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        """
        Apply layer normalization to input tensor.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Normalize over the last dimension (the feature/embedding dimension)
        # For input shape (batch_size, seq_len, n_embd), this normalizes over n_embd
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute variance across the normalized dimension (unbiased=False for consistency with PyTorch)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(variance + eps)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        
        # Apply scale and shift parameters: element-wise multiplication and addition
        # This results in shape (..., normalized_shape)
        return self.scale * normalized + self.bias

