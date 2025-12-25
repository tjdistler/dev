"""
Unit tests for LayerNorm.
"""

import pytest
import torch

from gpt.layer_norm import LayerNorm


class TestLayerNorm:
    """Test cases for LayerNorm class."""
    
    def test_layer_norm_initialization(self):
        """Test LayerNorm initialization."""
        # TODO: Create LayerNorm instance
        # TODO: Assert scale and bias parameters exist
        # TODO: Assert correct shape of parameters
        pass
    
    def test_layer_norm_forward_shape(self):
        """Test that forward pass maintains input shape."""
        # TODO: Create LayerNorm instance
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Assert output shape matches input shape
        pass
    
    def test_layer_norm_normalization(self):
        """Test that layer norm normalizes across correct dimension."""
        # TODO: Create LayerNorm instance
        # TODO: Create input tensor
        # TODO: Run forward pass
        # TODO: Assert mean is approximately 0
        # TODO: Assert variance is approximately 1
        pass
    
    def test_layer_norm_learnable_parameters(self):
        """Test that scale and bias are learnable."""
        # TODO: Create LayerNorm instance
        # TODO: Check that scale and bias require gradients
        pass
    
    def test_layer_norm_eps_parameter(self):
        """Test that eps parameter prevents division by zero."""
        # TODO: Create LayerNorm with different eps values
        # TODO: Test with very small variance inputs
        # TODO: Assert no NaN or Inf values
        pass
    
    def test_layer_norm_gradient_flow(self):
        """Test that gradients flow through layer norm."""
        # TODO: Create LayerNorm instance
        # TODO: Create input with requires_grad=True
        # TODO: Run forward and backward pass
        # TODO: Assert gradients exist for input and parameters
        pass

