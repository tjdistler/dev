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
        # Validate that n_embd is divisible by n_head
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        # Calculate head dimension (needed for scaling and reshaping)
        head_dim = n_embd // n_head
        
        # Store only what's needed for forward pass
        self.n_head = n_head  # Needed for splitting into multiple heads
        self.head_dim = head_dim  # Needed for reshaping Q, K, V
        
        # Pre-compute scaling factor for attention scores (1 / sqrt(head_dim))
        # This is a constant used in every forward pass, so compute once in init
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Define the W_q, W_k, W_v, W_o matrices as nn.Parameter
        # In GPT-1, each head projects to head_dim (64), so we project to n_head * head_dim = n_embd
        # This is equivalent to: n_embd -> (n_head * head_dim) = n_embd -> n_embd
        # But conceptually, each head operates on head_dim dimensions
        # W_q, W_k, W_v: project n_embd -> n_embd (which will be reshaped to n_head heads of head_dim each)
        self.W_q = nn.Parameter(torch.randn(n_embd, n_embd))
        self.W_k = nn.Parameter(torch.randn(n_embd, n_embd))
        self.W_v = nn.Parameter(torch.randn(n_embd, n_embd))
        # W_o: output projection from n_embd -> n_embd
        self.W_o = nn.Parameter(torch.randn(n_embd, n_embd))
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask is sequence-length dependent (seq_len x seq_len), not embedding dimension
        # We create it dynamically in forward() based on actual sequence length because:
        # 1. Sequence length can vary between batches
        # 2. We don't know the device (CPU/GPU) until forward pass
        # Optional optimization: If n_positions is known, we could create a max-size mask
        # here and slice it in forward(), but for simplicity we create it on-demand
        
    def forward(self, x):
        """
        Forward pass through multi-head causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Step 1: Project input to Query, Key, and Value representations
        # Each projection uses a learned weight matrix to transform the input embeddings
        # Input: x shape (batch_size, seq_len, n_embd)
        # W_q, W_k, W_v shape: (n_embd, n_embd)
        # Output: Q, K, V shape (batch_size, seq_len, n_embd)
        # This creates three different representations of the same input:
        # - Q (Query): "what am I looking for?"
        # - K (Key): "what do I represent?"
        # - V (Value): "what information do I contain?"
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Step 2: Reshape Q, K, V for multi-head attention
        # Split the embedding dimension into n_head separate heads, each with head_dim dimensions
        # Reshape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_head, head_dim)
        # Then transpose to: (batch_size, n_head, seq_len, head_dim)
        # This allows each head to operate independently on its own subspace
        # Example: n_embd=768, n_head=12, head_dim=64
        #   Before: (batch, seq_len, 768)
        #   After:  (batch, 12, seq_len, 64) - 12 heads, each with 64 dimensions
        Q = Q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # Q, K, V shape: (batch_size, n_head, seq_len, head_dim)
        
        # Step 3: Compute attention scores using scaled dot-product (per head)
        # Attention scores measure how much each token should attend to every other token
        # Q @ K^T: computes dot products between all query-key pairs for each head
        #   Shape: (batch_size, n_head, seq_len, head_dim) @ (batch_size, n_head, head_dim, seq_len)
        #   Result: (batch_size, n_head, seq_len, seq_len)
        # Scale by 1/sqrt(head_dim): prevents dot products from growing too large,
        #   which would push softmax into regions with extremely small gradients
        #   This is critical for training stability. We scale by head_dim (64) not n_embd (768)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        # scores shape: (batch_size, n_head, seq_len, seq_len)
        
        # Step 4: Apply causal mask to enforce autoregressive property
        # Causal masking prevents tokens from attending to future tokens (GPT is decoder-only)
        # Creates a lower triangular matrix: 1s allow attention, 0s mask out future positions
        # Example for seq_len=4:
        #   [[1, 0, 0, 0],
        #    [1, 1, 0, 0],
        #    [1, 1, 1, 0],
        #    [1, 1, 1, 1]]
        # mask shape: (seq_len, seq_len) - broadcasted to (batch_size, n_head, seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Set masked positions (future tokens) to -inf before softmax
        # After softmax, -inf becomes 0, effectively preventing attention to future tokens
        # The mask is broadcasted across batch and head dimensions
        # scores shape remains: (batch_size, n_head, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Step 5: Apply softmax to convert scores into attention weights (probabilities)
        # Softmax normalizes scores so each row sums to 1, creating a probability distribution
        # dim=-1: normalize across the last (sequence) dimension (each token's attention over all tokens)
        # attn_weights[i, h, j, k] = probability that token j in head h attends to token k
        # Shape: (batch_size, n_head, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Step 6: Apply dropout to attention weights for regularization
        # Randomly sets some attention weights to 0 during training to prevent overfitting
        # This encourages the model to not rely too heavily on any single attention connection
        # Shape remains: (batch_size, n_head, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Step 7: Compute weighted sum of values using attention weights (per head)
        # This is the core of attention: each token's output is a weighted combination of all values
        # attn_weights @ V: for each token in each head, sum up all value vectors weighted by attention
        #   Shape: (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, head_dim)
        #   Result: (batch_size, n_head, seq_len, head_dim)
        # Each output token in each head is now a context-aware representation
        output = attn_weights @ V
        # output shape: (batch_size, n_head, seq_len, head_dim)
        
        # Step 8: Concatenate heads back together
        # Transpose and reshape to combine all heads into a single tensor
        # Transpose: (batch_size, n_head, seq_len, head_dim) -> (batch_size, seq_len, n_head, head_dim)
        # Reshape: (batch_size, seq_len, n_head, head_dim) -> (batch_size, seq_len, n_embd)
        # This concatenates the head_dim dimensions from all n_head heads back into n_embd
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        # output shape: (batch_size, seq_len, n_embd)
        
        # Step 9: Apply output projection to transform the attended representations
        # W_o projects the concatenated attention output back to the embedding dimension
        # This allows the model to learn a final transformation that combines information from all heads
        # Shape: (batch_size, seq_len, n_embd) @ (n_embd, n_embd) -> (batch_size, seq_len, n_embd)
        output = output @ self.W_o
        
        return output

