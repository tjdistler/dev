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
    Transformer decoder block.
    
    Consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Position-wise MLP with residual connection and layer norm
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_inner: Inner dimension for MLP (default: 4 * n_embd)
        dropout: Dropout probability
        activation: Activation function type
    """
    
    def __init__(self, n_embd, n_head, n_inner=None, dropout=0.1, activation='gelu'):
        super().__init__()
        # TODO: Initialize layer norm for attention
        # TODO: Initialize multi-head self-attention
        # TODO: Initialize layer norm for MLP
        # TODO: Initialize MLP
        # TODO: Initialize dropout layers
        
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # TODO: Apply layer norm
        # TODO: Apply attention with residual connection: x = x + attention(ln(x))
        # TODO: Apply layer norm
        # TODO: Apply MLP with residual connection: x = x + mlp(ln(x))
        # TODO: Return output
        pass

