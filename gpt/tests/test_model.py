"""
Unit tests for GPT model.
"""

import pytest
import torch

from gpt.model import GPT
from gpt.config import GPTConfig


class TestGPT:
    """Test cases for GPT class."""
    
    def test_gpt_initialization(self):
        """Test GPT model initialization."""
        # TODO: Create GPTConfig
        # TODO: Create GPT model
        # TODO: Assert embeddings exist
        # TODO: Assert transformer blocks exist (n_layer blocks)
        # TODO: Assert final layer norm exists
        # TODO: Assert language modeling head exists
        pass
    
    def test_gpt_forward_shape(self):
        """Test that forward pass returns correct logits shape."""
        # TODO: Create GPT model
        # TODO: Create input_ids (batch, seq_len)
        # TODO: Run forward pass
        # TODO: Assert output shape is (batch, seq_len, vocab_size)
        pass
    
    def test_gpt_forward_with_targets(self):
        """Test forward pass with targets returns loss."""
        # TODO: Create GPT model
        # TODO: Create input_ids and targets
        # TODO: Run forward pass
        # TODO: Assert loss is returned
        # TODO: Assert loss is a scalar tensor
        pass
    
    def test_gpt_forward_without_targets(self):
        """Test forward pass without targets returns only logits."""
        # TODO: Create GPT model
        # TODO: Create input_ids (no targets)
        # TODO: Run forward pass
        # TODO: Assert only logits are returned (not tuple)
        pass
    
    def test_gpt_num_params(self):
        """Test parameter counting."""
        # TODO: Create GPT model
        # TODO: Call get_num_params()
        # TODO: Assert parameter count is reasonable
        pass
    
    def test_gpt_num_params_non_embedding(self):
        """Test parameter counting excluding embeddings."""
        # TODO: Create GPT model
        # TODO: Call get_num_params(non_embedding=True)
        # TODO: Assert count is less than total params
        pass
    
    def test_gpt_weight_initialization(self):
        """Test that weights are initialized correctly."""
        # TODO: Create GPT model
        # TODO: Check weight initialization of linear layers
        # TODO: Check weight initialization of embeddings
        # TODO: Assert biases are initialized to zero
        pass
    
    def test_gpt_different_sequence_lengths(self):
        """Test model handles different sequence lengths."""
        # TODO: Create GPT model
        # TODO: Create input_ids with different seq lengths
        # TODO: Run forward pass for each
        # TODO: Assert all work correctly
        pass
    
    def test_gpt_gradient_flow(self):
        """Test that gradients flow through entire model."""
        # TODO: Create GPT model
        # TODO: Create input_ids with requires_grad=False (but model params do)
        # TODO: Run forward and backward pass
        # TODO: Assert gradients exist for model parameters
        pass
    
    def test_gpt_causal_language_modeling(self):
        """Test that model predicts next token correctly."""
        # TODO: Create GPT model
        # TODO: Create input_ids
        # TODO: Run forward pass
        # TODO: Verify logits predict next tokens (not previous)
        pass

