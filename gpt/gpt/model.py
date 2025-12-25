"""
Main GPT-1 Model.

Implements the complete GPT-1 transformer model.
"""

import torch
import torch.nn as nn

from .config import GPTConfig
from .embeddings import GPTEmbeddings
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm


class GPT(nn.Module):
    """
    GPT-1 Language Model.
    
    Complete transformer-based language model architecture.
    
    Args:
        config: GPTConfig instance with model hyperparameters
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # TODO: Initialize embeddings (token + positional)
        # TODO: Initialize transformer blocks (n_layer blocks)
        # TODO: Initialize final layer norm
        # TODO: Initialize language modeling head (linear layer: n_embd -> vocab_size)
        # TODO: Initialize dropout for embeddings
        
        # TODO: Apply weight initialization
        
    def forward(self, input_ids, targets=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            targets: Optional target token indices for loss computation
            
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: tuple of (logits, loss)
        """
        # TODO: Get batch size and sequence length
        # TODO: Get embeddings
        # TODO: Pass through transformer blocks
        # TODO: Apply final layer norm
        # TODO: Get logits from language modeling head
        
        # TODO: If targets provided, compute cross-entropy loss
        # TODO: Return logits (and loss if targets provided)
        pass
    
    def get_num_params(self, non_embedding=False):
        """
        Get the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        # TODO: Count total parameters
        # TODO: If non_embedding=True, subtract embedding parameters
        # TODO: Return count
        pass
    
    def _init_weights(self, module):
        """
        Initialize model weights.
        
        Args:
            module: PyTorch module to initialize
        """
        # TODO: Initialize linear layers with normal distribution
        # TODO: Initialize embeddings with normal distribution
        # TODO: Initialize biases to zero
        # TODO: Use config.initializer_range for std
        pass

