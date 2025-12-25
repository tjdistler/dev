"""
Unit tests for CausalSelfAttention.
"""

import pytest
import torch

from gpt.attention import CausalSelfAttention


class TestCausalSelfAttention:
    """Test cases for CausalSelfAttention class."""
    
    def test_attention_initialization(self):
        """Test CausalSelfAttention initialization."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Assert Q, K, V projection layers exist
        # TODO: Assert output projection layer exists
        # TODO: Assert dropout layer exists
        pass
    
    def test_attention_initialization_head_dim(self):
        """Test that head_dim is calculated correctly."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Assert head_dim = n_embd // n_head
        pass
    
    def test_attention_forward_shape(self):
        """Test that forward pass maintains input shape."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Create input tensor (batch, seq_len, n_embd)
        # TODO: Run forward pass
        # TODO: Assert output shape is (batch, seq_len, n_embd)
        pass
    
    def test_attention_causal_mask(self):
        """Test that causal mask prevents attending to future tokens."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Extract attention weights
        # TODO: Assert upper triangular part is zero (or -inf before softmax)
        pass
    
    def test_attention_scaled_dot_product(self):
        """Test that attention scores are scaled by sqrt(head_dim)."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Create input tensor
        # TODO: Manually compute attention scores
        # TODO: Assert scaling factor is sqrt(head_dim)
        pass
    
    def test_attention_softmax(self):
        """Test that attention weights sum to 1."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Create input tensor
        # TODO: Extract attention weights
        # TODO: Assert each row sums to approximately 1
        pass
    
    def test_attention_dropout_training(self):
        """Test dropout is applied in training mode."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Set to training mode
        # TODO: Run forward pass multiple times
        # TODO: Assert some attention weights are zeroed
        pass
    
    def test_attention_dropout_eval(self):
        """Test dropout is not applied in eval mode."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Set to eval mode
        # TODO: Run forward pass multiple times
        # TODO: Assert outputs are consistent
        pass
    
    def test_attention_multi_head(self):
        """Test that multiple attention heads are computed."""
        # TODO: Create CausalSelfAttention with n_head > 1
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Verify that heads are computed and concatenated
        pass
    
    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention."""
        # TODO: Create CausalSelfAttention instance
        # TODO: Create input with requires_grad=True
        # TODO: Run forward and backward pass
        # TODO: Assert gradients exist for all parameters
        pass

