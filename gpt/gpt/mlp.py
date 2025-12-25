"""
Multi-Layer Perceptron (MLP) / Feed-Forward Network.

Implements the position-wise feed-forward network used in transformer blocks.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    
    Consists of two linear transformations with a GELU activation
    in between, as used in GPT-1.
    
    Args:
        n_embd: Embedding dimension (input and output size)
        n_inner: Inner dimension (size of hidden layer)
        activation: Activation function type (default: 'gelu')
        dropout: Dropout probability
    """
    
    def __init__(self, n_embd, n_inner=None, activation='gelu', dropout=0.1):
        super().__init__()
        # TODO: Set n_inner (typically 4 * n_embd if not specified)
        # TODO: Initialize first linear layer (n_embd -> n_inner)
        # TODO: Initialize activation function
        # TODO: Initialize second linear layer (n_inner -> n_embd)
        # TODO: Initialize dropout layer
        
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # TODO: Apply first linear transformation
        # TODO: Apply activation function
        # TODO: Apply dropout
        # TODO: Apply second linear transformation
        # TODO: Return output
        pass

