"""
Transformer Decoder Block.

Implements a single transformer decoder block as used in GPT-1.
"""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import MLP
from .layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    Transformer decoder block (GPT-1 post-norm architecture).
    
    Consists of:
    1. Multi-head self-attention with residual connection, followed by layer norm
    2. Position-wise MLP with residual connection, followed by layer norm
    
    Architecture (post-norm):
        x = x + attention(x)
        x = layer_norm(x)
        x = x + mlp(x)
        x = layer_norm(x)
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_inner: Inner dimension for MLP (default: 4 * n_embd)
        dropout: Dropout probability
        activation: Activation function type
    """
    
    def __init__(self, n_embd, n_head, n_inner=None, dropout=0.1, activation='gelu'):
        super().__init__()
        # Set n_inner (typically 4 * n_embd if not specified)
        if n_inner is None:
            n_inner = 4 * n_embd
        
        # Initialize multi-head self-attention
        self.attention = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        
        # Initialize layer norm (applied after attention in post-norm architecture)
        self.ln1 = LayerNorm(n_embd)
        
        # Initialize MLP
        self.mlp = MLP(n_embd, n_inner, activation=activation, dropout=dropout)
        
        # Initialize layer norm (applied after MLP in post-norm architecture)
        self.ln2 = LayerNorm(n_embd)
        
        # Initialize dropout layers
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Apply attention with residual connection: x = x + attention(x)
        x = x + self.attention(x)
        
        # Apply layer norm after attention (post-norm architecture)
        x = self.ln1(x)
        
        # Apply MLP with residual connection: x = x + mlp(x)
        x = x + self.mlp(x)
        
        # Apply layer norm after MLP (post-norm architecture)
        x = self.ln2(x)
        
        # Return output
        return x

