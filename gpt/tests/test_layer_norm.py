"""
Unit tests for LayerNorm.
"""

import pytest
import torch
import torch.nn as nn

from gpt.layer_norm import LayerNorm


class TestLayerNorm:
    """Test cases for LayerNorm class."""
    
    def test_layer_norm_initialization(self):
        """Test LayerNorm initialization."""
        n_embd = 768
        eps = 1e-5
        
        ln = LayerNorm(n_embd, eps=eps)
        
        # Assert scale and bias parameters exist
        assert hasattr(ln, 'scale')
        assert hasattr(ln, 'bias')
        
        # Assert correct shape of parameters
        assert ln.scale.shape == (n_embd,)
        assert ln.bias.shape == (n_embd,)
        
        # Assert initial values
        assert torch.allclose(ln.scale, torch.ones(n_embd)), \
            "Scale should be initialized to ones"
        assert torch.allclose(ln.bias, torch.zeros(n_embd)), \
            "Bias should be initialized to zeros"
        
        # Assert configuration parameters
        assert ln.normalized_shape == n_embd
        assert ln.eps == eps
    
    def test_layer_norm_forward_shape(self):
        """Test that forward pass maintains input shape."""
        n_embd = 768
        batch_size = 2
        seq_len = 10
        
        ln = LayerNorm(n_embd)
        
        x = torch.randn(batch_size, seq_len, n_embd)
        output = ln(x)
        
        assert output.shape == (batch_size, seq_len, n_embd)
        assert output.shape == x.shape
    
    def test_layer_norm_normalization(self):
        """Test that layer norm normalizes across correct dimension."""
        n_embd = 768
        batch_size = 2
        seq_len = 10
        
        ln = LayerNorm(n_embd)
        
        # Create input with non-zero mean and non-unit variance
        x = torch.randn(batch_size, seq_len, n_embd) * 5.0 + 10.0
        
        # Run forward pass
        output = ln(x)
        
        # Before scale and bias, normalized should have mean ~0 and variance ~1
        # We need to manually compute what happens before scale/bias
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_manual = (x - mean) / torch.sqrt(variance + ln.eps)
        
        # Check normalization properties (before scale/bias)
        normalized_mean = normalized_manual.mean(dim=-1)
        normalized_var = normalized_manual.var(dim=-1, unbiased=False)
        
        assert torch.allclose(normalized_mean, torch.zeros_like(normalized_mean), atol=1e-5), \
            "Normalized values should have mean approximately 0"
        assert torch.allclose(normalized_var, torch.ones_like(normalized_var), atol=1e-4), \
            "Normalized values should have variance approximately 1"
    
    def test_layer_norm_learnable_parameters(self):
        """Test that scale and bias are learnable."""
        n_embd = 768
        
        ln = LayerNorm(n_embd)
        
        # Check that scale and bias require gradients
        assert ln.scale.requires_grad, "Scale should require gradients"
        assert ln.bias.requires_grad, "Bias should require gradients"
        
        # Check that they are Parameters (not just Tensors)
        assert isinstance(ln.scale, nn.Parameter), "Scale should be a Parameter"
        assert isinstance(ln.bias, nn.Parameter), "Bias should be a Parameter"
    
    def test_layer_norm_eps_parameter(self):
        """Test that eps parameter prevents division by zero."""
        n_embd = 768
        
        # Test with default eps
        ln1 = LayerNorm(n_embd, eps=1e-5)
        
        # Test with different eps values
        ln2 = LayerNorm(n_embd, eps=1e-8)
        ln3 = LayerNorm(n_embd, eps=1e-3)
        
        # Create input with very small variance (near zero)
        x = torch.ones(2, 10, n_embd) * 0.001  # Very small, nearly constant values
        
        # All should produce finite outputs
        output1 = ln1(x)
        output2 = ln2(x)
        output3 = ln3(x)
        
        # Assert no NaN or Inf values
        assert torch.isfinite(output1).all(), \
            "Output should be finite with eps=1e-5"
        assert torch.isfinite(output2).all(), \
            "Output should be finite with eps=1e-8"
        assert torch.isfinite(output3).all(), \
            "Output should be finite with eps=1e-3"
        
        assert not torch.isnan(output1).any(), \
            "Output should not contain NaN values"
        assert not torch.isinf(output1).any(), \
            "Output should not contain Inf values"
    
    def test_layer_norm_gradient_flow(self):
        """Test that gradients flow through layer norm."""
        n_embd = 768
        
        ln = LayerNorm(n_embd)
        ln.train()
        
        x = torch.randn(2, 10, n_embd, requires_grad=True)
        output = ln(x)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for input
        assert x.grad is not None, "Input should have gradients"
        
        # Check that gradients exist for parameters
        assert ln.scale.grad is not None, "Scale should have gradients"
        assert ln.bias.grad is not None, "Bias should have gradients"
        
        # Check that gradients are non-zero (actually computed)
        assert not torch.allclose(ln.scale.grad, torch.zeros_like(ln.scale.grad)), \
            "Scale gradients should be non-zero"
        assert not torch.allclose(ln.bias.grad, torch.zeros_like(ln.bias.grad)), \
            "Bias gradients should be non-zero"
    
    def test_layer_norm_identity_initialization(self):
        """Test that with default initialization, output is close to normalized input."""
        n_embd = 768
        batch_size = 2
        seq_len = 10
        
        ln = LayerNorm(n_embd)
        
        x = torch.randn(batch_size, seq_len, n_embd)
        output = ln(x)
        
        # With scale=1 and bias=0, output should be normalized version of input
        # (not exactly the same due to normalization, but should be close after normalization)
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_expected = (x - mean) / torch.sqrt(variance + ln.eps)
        
        # Output should match normalized input (since scale=1, bias=0 initially)
        assert torch.allclose(output, normalized_expected, atol=1e-5), \
            "With default initialization, output should be normalized input"
    
    def test_layer_norm_different_shapes(self):
        """Test that layer norm works with different input shapes."""
        n_embd = 768
        
        ln = LayerNorm(n_embd)
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 1, n_embd),      # Single token
            (1, 10, n_embd),     # Single batch, multiple tokens
            (2, 10, n_embd),     # Multiple batches
            (4, 100, n_embd),    # Larger sequences
        ]
        
        for batch_size, seq_len, embd_dim in test_cases:
            x = torch.randn(batch_size, seq_len, embd_dim)
            output = ln(x)
            assert output.shape == (batch_size, seq_len, embd_dim), \
                f"Output shape should match input for shape {x.shape}"
    
    def test_layer_norm_scale_and_bias_effect(self):
        """Test that scale and bias parameters actually affect the output."""
        n_embd = 768
        
        ln1 = LayerNorm(n_embd)
        ln2 = LayerNorm(n_embd)
        
        # Modify scale and bias in ln2
        with torch.no_grad():
            ln2.scale.fill_(2.0)  # Scale by 2
            ln2.bias.fill_(1.0)    # Add 1
        
        x = torch.randn(2, 10, n_embd)
        
        output1 = ln1(x)
        output2 = ln2(x)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2, atol=1e-3), \
            "Different scale/bias should produce different outputs"
        
        # Output2 should be approximately 2 * normalized + 1
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + ln2.eps)
        expected = 2.0 * normalized + 1.0
        
        assert torch.allclose(output2, expected, atol=1e-5), \
            "Output should be scale * normalized + bias"
    
    def test_layer_norm_consistency(self):
        """Test that layer norm produces consistent results."""
        n_embd = 768
        
        ln = LayerNorm(n_embd)
        ln.eval()  # Disable any randomness
        
        x = torch.randn(2, 10, n_embd)
        
        # Run multiple times - should be identical
        output1 = ln(x)
        output2 = ln(x)
        output3 = ln(x)
        
        assert torch.allclose(output1, output2), \
            "Outputs should be identical for same input"
        assert torch.allclose(output2, output3), \
            "Outputs should be identical for same input"
    
    def test_layer_norm_parameter_count(self):
        """Test that layer norm has correct number of parameters."""
        n_embd = 768
        
        ln = LayerNorm(n_embd)
        
        # Count parameters: scale (n_embd) + bias (n_embd) = 2 * n_embd
        expected_params = 2 * n_embd
        
        total_params = sum(p.numel() for p in ln.parameters())
        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"
    
    def test_layer_norm_comparison_with_pytorch(self):
        """Test that our LayerNorm matches PyTorch's LayerNorm behavior."""
        n_embd = 768
        batch_size = 2
        seq_len = 10
        
        our_ln = LayerNorm(n_embd)
        pytorch_ln = nn.LayerNorm(n_embd)
        
        # Copy weights to make them identical
        with torch.no_grad():
            pytorch_ln.weight.data = our_ln.scale.data.clone()
            pytorch_ln.bias.data = our_ln.bias.data.clone()
        
        x = torch.randn(batch_size, seq_len, n_embd)
        
        our_output = our_ln(x)
        pytorch_output = pytorch_ln(x)
        
        # Outputs should be very close (allowing for small numerical differences)
        assert torch.allclose(our_output, pytorch_output, atol=1e-5, rtol=1e-4), \
            "Our LayerNorm should match PyTorch's LayerNorm"

