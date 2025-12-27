"""
Multi-Layer Perceptron (MLP) / Feed-Forward Network.

Implements the position-wise feed-forward network used in transformer blocks.
"""

import torch
import torch.nn as nn

from .linear import Linear


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    
    Consists of two linear transformations with a GELU activation
    in between, as used in GPT-1.
    
    Architecture:
        x -> Linear(n_embd -> n_inner) -> GELU -> Dropout -> Linear(n_inner -> n_embd) -> output
    
    The MLP is applied independently to each position in the sequence (position-wise),
    meaning the same transformation is applied to each token's embedding vector.
    This allows the model to learn complex non-linear transformations of the representations.
    
    Args:
        n_embd: Embedding dimension (input and output size)
        n_inner: Inner dimension (size of hidden layer, typically 4 * n_embd)
        activation: Activation function type (default: 'gelu')
        dropout: Dropout probability (applied after activation)
        initializer_range: Standard deviation for weight initialization (default: 0.02)
    """
    
    def __init__(self, n_embd, n_inner=None, activation='gelu', dropout=0.1, initializer_range=0.02):
        super().__init__()
        # Store dimensions for use in forward pass
        self.n_embd = n_embd
        
        # Set n_inner (typically 4 * n_embd if not specified)
        # The expansion factor of 4 is a common choice in transformers, providing
        # enough capacity for the model to learn complex transformations while keeping
        # the number of parameters reasonable.
        if n_inner is None:
            n_inner = 4 * n_embd
        self.n_inner = n_inner
        
        # Initialize first linear layer: expands from embedding dimension to inner dimension
        # This layer projects each position's embedding vector into a higher-dimensional space,
        # allowing the model to learn more complex features. The expansion (typically 4x)
        # provides additional capacity for the transformation.
        # Shape transformation: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_inner)
        self.c_fc = Linear(n_embd, n_inner, bias=True, initializer_range=initializer_range)
        
        # Initialize activation function
        # GELU (Gaussian Error Linear Unit) is the activation function used in GPT-1.
        # It's a smooth, non-linear function that provides better gradients than ReLU
        # and has been shown to work well in transformer models.
        # GELU(x) = x * Φ(x), where Φ(x) is the CDF of the standard normal distribution.
        # PyTorch provides nn.GELU() which implements this efficiently.
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Initialize dropout layer: applied after activation for regularization
        # Dropout randomly sets a fraction of activations to zero during training,
        # which helps prevent overfitting by encouraging the model to not rely too
        # heavily on any single activation. During evaluation, dropout is disabled.
        self.dropout = nn.Dropout(dropout)
        
        # Initialize second linear layer: contracts from inner dimension back to embedding dimension
        # This layer projects the expanded representation back to the original embedding dimension,
        # creating a bottleneck architecture. The combination of expansion and contraction allows
        # the model to learn complex transformations while maintaining the embedding dimension.
        # Shape transformation: (batch_size, seq_len, n_inner) -> (batch_size, seq_len, n_embd)
        self.c_proj = Linear(n_inner, n_embd, bias=True, initializer_range=initializer_range)
        
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        The forward pass applies the following transformations:
        1. First linear layer: expands embedding dimension (n_embd -> n_inner)
        2. Activation function: applies GELU non-linearity
        3. Dropout: randomly zeros activations during training
        4. Second linear layer: contracts back to embedding dimension (n_inner -> n_embd)
        
        The MLP is applied position-wise, meaning the same transformation is applied
        independently to each position in the sequence. This allows each token's
        representation to be transformed based on its own features, while the
        attention mechanism handles interactions between tokens.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
                The input embeddings from the previous layer (typically after layer norm).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
                The transformed embeddings with the same shape as the input.
                This allows the MLP to be used in a residual connection.
        """
        # Step 1: Apply first linear transformation (expansion)
        # Expands each position's embedding from n_embd to n_inner dimensions.
        # This provides additional capacity for learning complex transformations.
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_inner)
        x = self.c_fc(x)
        
        # Step 2: Apply activation function (GELU)
        # The GELU activation introduces non-linearity, allowing the model to learn
        # complex, non-linear transformations. GELU is smooth and differentiable,
        # which helps with gradient flow during backpropagation.
        # Shape: (batch_size, seq_len, n_inner) -> (batch_size, seq_len, n_inner)
        x = self.activation(x)
        
        # Step 3: Apply dropout for regularization
        # During training, dropout randomly sets some activations to zero.
        # This prevents the model from overfitting by making it more robust to
        # missing or noisy activations. During evaluation (eval mode), dropout is
        # automatically disabled and all activations pass through unchanged.
        # Shape: (batch_size, seq_len, n_inner) -> (batch_size, seq_len, n_inner)
        x = self.dropout(x)
        
        # Step 4: Apply second linear transformation (contraction)
        # Projects the expanded representation back to the original embedding dimension.
        # This creates a bottleneck that forces the model to learn efficient representations.
        # Shape: (batch_size, seq_len, n_inner) -> (batch_size, seq_len, n_embd)
        x = self.c_proj(x)
        
        return x

