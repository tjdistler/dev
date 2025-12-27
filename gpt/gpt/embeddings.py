"""
Embedding layers for GPT-1.

Implements token embeddings and positional embeddings.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        # Token embeddings map each token ID (integer) to a dense vector representation in a continuous space.
        # This is a learnable lookup table of shape (vocab_size, n_embd) where each row represents the embedding
        # vector for a specific token. During training, these embeddings are learned to capture semantic and
        # syntactic relationships between tokens - similar tokens will have similar embedding vectors. The embeddings
        # are initialized randomly and updated through backpropagation as the model learns.
        self.token_embeddings = nn.Parameter(torch.randn(vocab_size, n_embd))
        logger.debug(f"Token embeddings: {self.token_embeddings.shape}")

        # Positional embeddings encode the position of each token in the sequence, allowing the model to understand
        # word order and relative positions. This is a learnable lookup table of shape (n_positions, n_embd) where
        # each row represents the embedding vector for a specific position (0, 1, 2, ..., n_positions-1). Since
        # transformers process all tokens in parallel (unlike RNNs), they need explicit positional information to
        # understand sequence order. These embeddings are added to token embeddings before being passed to the
        # transformer blocks. GPT-1 uses learnable positional embeddings (initialized randomly) rather than fixed
        # sinusoidal patterns.
        self.position_embeddings = nn.Parameter(torch.randn(n_positions, n_embd))
        logger.debug(f"Position embeddings: {self.position_embeddings.shape}")
        # Dropout is a regularization technique that randomly sets a fraction of activations to zero during training
        # to prevent overfitting. During training, each element has probability p of being set to 0, and the
        # remaining values are scaled by 1/(1-p) to maintain the expected sum. During evaluation, dropout is
        # automatically disabled and all values pass through unchanged. This helps the model learn more robust
        # representations that don't over-rely on specific embedding dimensions.
        self.dropout = nn.Dropout(embd_pdrop)
        logger.debug(f"Dropout: {self.dropout}")

    def forward(self, input_ids):
        """
        Forward pass through embeddings.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, n_embd)
        """
        # Get token embeddings: use advanced indexing to look up embeddings for all tokens in all batches in parallel.
        # input_ids has shape (batch_size, seq_len) with token IDs, and self.token_embeddings has shape (vocab_size, n_embd).
        # Indexing self.token_embeddings[input_ids] automatically broadcasts: for each token ID in input_ids, it retrieves
        # the corresponding row from self.token_embeddings, resulting in shape (batch_size, seq_len, n_embd).
        token_embeddings = self.token_embeddings[input_ids]
        logger.debug(f"Token embeddings: {token_embeddings.shape}")

        # Get positional embeddings: use advanced indexing to look up embeddings for all positions in all batches in parallel.
        # self.position_embeddings has shape (n_positions, n_embd), and we want to get the embeddings for positions 0 to seq_len-1.
        # input_ids.shape[1] is the sequence length, so we can use slice indexing to get the embeddings for all positions up to seq_len-1.
        # This results in shape (batch_size, seq_len, n_embd).
        positional_embeddings = self.position_embeddings[:input_ids.shape[1]]
        logger.debug(f"Positional embeddings: {positional_embeddings.shape}")

        # Add token and positional embeddings: element-wise addition of token_embeddings and positional_embeddings.
        # This results in shape (batch_size, seq_len, n_embd).
        embeddings = token_embeddings + positional_embeddings
        logger.debug(f"Embeddings: {embeddings.shape}")

        # Apply dropout: randomly set a fraction of activations to zero during training to prevent overfitting.
        # This helps the model learn more robust representations that don't over-rely on specific embedding dimensions.
        embeddings = self.dropout(embeddings)
        logger.debug(f"Dropout embeddings: {embeddings.shape}")

        # Return combined embeddings
        return embeddings

