"""
Embedding layers for GPT-1.

Implements token embeddings and positional embeddings.
"""

import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """
    Combined token and positional embeddings for GPT-1.
    
    Args:
        vocab_size: Size of the vocabulary
        n_embd: Embedding dimension
        n_positions: Maximum sequence length
        embd_pdrop: Embedding dropout probability
    """
    
    def __init__(self, vocab_size, n_embd, n_positions, embd_pdrop=0.1):
        super().__init__()
        # TODO: Initialize token embedding layer
        # TODO: Initialize positional embedding (learnable or sinusoidal)
        # TODO: Initialize dropout layer
        
    def forward(self, input_ids):
        """
        Forward pass through embeddings.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, n_embd)
        """
        # TODO: Get sequence length from input_ids
        # TODO: Get token embeddings
        # TODO: Get positional embeddings (for positions 0 to seq_len-1)
        # TODO: Add token and positional embeddings
        # TODO: Apply dropout
        # TODO: Return combined embeddings
        pass

