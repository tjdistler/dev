"""
Unit tests for TransformerBlock.
"""

import pytest
import torch

from gpt.transformer_block import TransformerBlock


class TestTransformerBlock:
    """Test cases for TransformerBlock class."""
    
    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization."""
        # TODO: Create TransformerBlock instance
        # TODO: Assert attention layer exists
        # TODO: Assert MLP layer exists
        # TODO: Assert layer norm layers exist
        pass
    
    def test_transformer_block_forward_shape(self):
        """Test that forward pass maintains input shape."""
        # TODO: Create TransformerBlock instance
        # TODO: Create input tensor (batch, seq_len, n_embd)
        # TODO: Run forward pass
        # TODO: Assert output shape is (batch, seq_len, n_embd)
        pass
    
    def test_transformer_block_residual_connection_attention(self):
        """Test that residual connection is applied after attention."""
        # TODO: Create TransformerBlock instance
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Verify that input is added to attention output
        pass
    
    def test_transformer_block_residual_connection_mlp(self):
        """Test that residual connection is applied after MLP."""
        # TODO: Create TransformerBlock instance
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Verify that input is added to MLP output
        pass
    
    def test_transformer_block_layer_norm_order(self):
        """Test that layer norm is applied before attention and MLP."""
        # TODO: Create TransformerBlock instance
        # TODO: Create input tensor
        # TODO: Manually verify layer norm is applied before transformations
        pass
    
    def test_transformer_block_gradient_flow(self):
        """Test that gradients flow through transformer block."""
        # TODO: Create TransformerBlock instance
        # TODO: Create input with requires_grad=True
        # TODO: Run forward and backward pass
        # TODO: Assert gradients exist for all parameters
        pass
    
    def test_transformer_block_dropout(self):
        """Test that dropout is applied correctly."""
        # TODO: Create TransformerBlock instance
        # TODO: Test in training and eval modes
        # TODO: Verify dropout behavior
        pass

