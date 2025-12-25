"""
Multi-Head Self-Attention implementation.

Implements scaled dot-product attention and multi-head attention
from scratch using PyTorch tensors.
"""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    Implements the attention mechanism used in GPT-1 with causal masking
    to prevent attending to future tokens.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1, bias=True):
        super().__init__()
        # TODO: Validate that n_embd is divisible by n_head
        # TODO: Calculate head dimension (n_embd // n_head)
        # TODO: Store n_head and head_dim
        # TODO: Initialize query, key, value projection layers
        # TODO: Initialize output projection layer
        # TODO: Initialize dropout layer
        # TODO: Create causal mask (lower triangular matrix)
        
    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # TODO: Get batch size and sequence length
        # TODO: Project to Q, K, V
        # TODO: Reshape for multi-head attention
        # TODO: Compute scaled dot-product attention
        # TODO: Apply causal mask to prevent attending to future tokens
        # TODO: Compute attention scores (Q @ K^T / sqrt(head_dim))
        # TODO: Apply softmax to get attention weights
        # TODO: Apply dropout to attention weights
        # TODO: Compute weighted sum (attention_weights @ V)
        # TODO: Reshape and concatenate heads
        # TODO: Apply output projection
        # TODO: Return output
        pass
    
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output and attention weights
        """
        # TODO: Compute attention scores: Q @ K^T / sqrt(head_dim)
        # TODO: Apply mask if provided (set masked positions to -inf)
        # TODO: Apply softmax to get attention weights
        # TODO: Apply dropout
        # TODO: Compute weighted sum: attention_weights @ V
        # TODO: Return output and attention weights
        pass

