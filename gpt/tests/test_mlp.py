"""
Unit tests for MLP (Multi-Layer Perceptron).
"""

import pytest
import torch

from gpt.mlp import MLP


class TestMLP:
    """Test cases for MLP class."""
    
    def test_mlp_initialization(self):
        """Test MLP initialization."""
        # TODO: Create MLP instance
        # TODO: Assert linear layers exist
        # TODO: Assert activation function exists
        # TODO: Assert dropout layer exists
        pass
    
    def test_mlp_initialization_default_n_inner(self):
        """Test that n_inner defaults to 4 * n_embd."""
        # TODO: Create MLP without specifying n_inner
        # TODO: Assert n_inner is 4 * n_embd
        pass
    
    def test_mlp_forward_shape(self):
        """Test that forward pass maintains batch and seq dimensions."""
        # TODO: Create MLP instance
        # TODO: Create input tensor (batch, seq_len, n_embd)
        # TODO: Run forward pass
        # TODO: Assert output shape is (batch, seq_len, n_embd)
        pass
    
    def test_mlp_activation_function(self):
        """Test that activation function is applied."""
        # TODO: Create MLP with known activation
        # TODO: Create input tensor
        # TODO: Manually check intermediate values
        # TODO: Assert activation is applied correctly
        pass
    
    def test_mlp_dropout_training_mode(self):
        """Test dropout is applied in training mode."""
        # TODO: Create MLP instance
        # TODO: Set to training mode
        # TODO: Run forward pass multiple times
        # TODO: Assert some values are zeroed (with high probability)
        pass
    
    def test_mlp_dropout_eval_mode(self):
        """Test dropout is not applied in eval mode."""
        # TODO: Create MLP instance
        # TODO: Set to eval mode
        # TODO: Run forward pass multiple times
        # TODO: Assert outputs are consistent (no dropout)
        pass
    
    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP."""
        # TODO: Create MLP instance
        # TODO: Create input with requires_grad=True
        # TODO: Run forward and backward pass
        # TODO: Assert gradients exist for all parameters
        pass

