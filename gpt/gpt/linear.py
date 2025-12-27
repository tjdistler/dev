"""
Custom Linear Layer.

Implements a linear transformation layer from scratch.
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear transformation layer (fully connected layer).
    
    Performs the operation: output = input @ weight.T + bias
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include a bias term (default: True)
    """
    
    def __init__(self, in_features, out_features, bias=True, initializer_range=0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter: shape (out_features, in_features)
        # The weight matrix defines the linear transformation. Each row corresponds to
        # one output feature, and each column corresponds to one input feature.
        # We use normal distribution initialization with standard deviation set to initializer_range.
        # This matches the GPT-1 initialization scheme used in the model, ensuring consistent
        # initialization across all linear layers. The small standard deviation (default 0.02)
        # prevents activations from being too large at the start of training, which helps
        # with gradient flow and training stability.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.normal_(self.weight, mean=0.0, std=initializer_range)
        
        # Initialize bias parameter if requested: shape (out_features,)
        # Bias adds a learnable offset to each output feature. This allows the model
        # to shift the output distribution, which is important for learning.
        # We initialize bias to zero, which is standard practice - the model will learn
        # appropriate bias values during training through backpropagation.
        # If bias=False, we set it to None (no bias term).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # Register None as a buffer so it's clear that bias is intentionally not used
            self.register_parameter('bias', None)
        
    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Performs the linear transformation: output = input @ weight.T + bias
        
        The operation x @ weight.T is equivalent to x @ weight^T, which computes:
        - For each sample in the batch, multiply the input vector by the weight matrix
        - The weight matrix is transposed because PyTorch stores it as (out_features, in_features)
        - This allows efficient matrix multiplication: (..., in_features) @ (in_features, out_features)
        
        Args:
            x: Input tensor of shape (..., in_features)
                The leading dimensions can be any shape (batch, sequence length, etc.)
                The last dimension must match in_features.
            
        Returns:
            Output tensor of shape (..., out_features)
                All leading dimensions are preserved, only the last dimension changes.
        """
        # Perform matrix multiplication: x @ weight.T
        # x has shape (..., in_features)
        # weight has shape (out_features, in_features)
        # weight.T has shape (in_features, out_features)
        # Result: (..., in_features) @ (in_features, out_features) -> (..., out_features)
        # The @ operator performs batched matrix multiplication, handling all leading dimensions.
        output = x @ self.weight.T
        
        # Add bias if it exists
        # Bias broadcasting: bias has shape (out_features,), which automatically broadcasts
        # to match the output shape (..., out_features) when added.
        # This adds the same bias value to all samples in the batch at each position.
        if self.bias is not None:
            output = output + self.bias
        
        return output

