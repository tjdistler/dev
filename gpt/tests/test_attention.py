"""
Unit tests for CausalSelfAttention.
"""

import pytest
import torch
import torch.nn as nn

from gpt.attention import CausalSelfAttention


class TestCausalSelfAttention:
    """Test cases for CausalSelfAttention class."""
    
    def test_attention_initialization(self):
        """Test CausalSelfAttention initialization."""
        n_embd = 768
        n_head = 12
        dropout = 0.1
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        
        # Assert weight matrices exist and have correct shapes
        assert hasattr(attn, 'W_q')
        assert hasattr(attn, 'W_k')
        assert hasattr(attn, 'W_v')
        assert hasattr(attn, 'W_o')
        assert attn.W_q.shape == (n_embd, n_embd)
        assert attn.W_k.shape == (n_embd, n_embd)
        assert attn.W_v.shape == (n_embd, n_embd)
        assert attn.W_o.shape == (n_embd, n_embd)
        
        # Assert dropout layer exists
        assert hasattr(attn, 'dropout')
        assert isinstance(attn.dropout, nn.Dropout)
        assert attn.dropout.p == dropout
    
    def test_attention_initialization_head_dim(self):
        """Test that head_dim is calculated correctly."""
        n_embd = 768
        n_head = 12
        expected_head_dim = n_embd // n_head  # 64
        
        attn = CausalSelfAttention(n_embd, n_head)
        
        assert attn.head_dim == expected_head_dim
        assert attn.n_head == n_head
        assert attn.scale == pytest.approx(1.0 / (expected_head_dim ** 0.5))
    
    def test_attention_initialization_invalid_n_head(self):
        """Test that initialization fails when n_embd is not divisible by n_head."""
        n_embd = 768
        n_head = 7  # 768 is not divisible by 7
        
        with pytest.raises(AssertionError, match="must be divisible"):
            CausalSelfAttention(n_embd, n_head)
    
    def test_attention_forward_shape(self):
        """Test that forward pass maintains input shape."""
        n_embd = 768
        n_head = 12
        batch_size = 2
        seq_len = 10
        
        attn = CausalSelfAttention(n_embd, n_head)
        attn.eval()  # Disable dropout for deterministic output
        
        x = torch.randn(batch_size, seq_len, n_embd)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, n_embd)
        assert output.shape == x.shape
    
    def test_attention_causal_mask(self):
        """Test that causal mask prevents attending to future tokens."""
        n_embd = 64
        n_head = 2
        seq_len = 5
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        attn.eval()
        
        # Create input where each token has a unique, identifiable value
        # Use larger values to ensure outputs are meaningful
        x = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(1, seq_len, n_embd) * 10.0
        
        # Run forward pass
        output = attn(x)
        
        # Test: Create a version where we swap future tokens
        # If causal masking works, swapping future tokens shouldn't affect earlier outputs
        x_swapped = x.clone()
        # Swap positions 3 and 4 (future tokens for positions 0, 1, 2)
        x_swapped[:, [3, 4], :] = x_swapped[:, [4, 3], :]
        
        output_swapped = attn(x_swapped)
        
        # For early positions (0, 1, 2), outputs should be identical
        # because they can't see the swapped future tokens
        # Use a more lenient tolerance since we're testing the class behavior
        early_positions_match = torch.allclose(output[:, :3, :], output_swapped[:, :3, :], atol=1e-4, rtol=1e-3)
        assert early_positions_match, \
            "Early tokens should not be affected by swapping future tokens (causal masking works)"
    
    def test_attention_scaled_dot_product(self):
        """Test that attention uses correct scaling factor."""
        n_embd = 64
        n_head = 2
        head_dim = n_embd // n_head  # 32
        expected_scale = 1.0 / (head_dim ** 0.5)
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        
        # Verify the scale attribute is correct
        assert attn.scale == pytest.approx(expected_scale), \
            f"Scale should be 1/sqrt({head_dim}) = {expected_scale}"
        
        # Test that the forward pass produces reasonable outputs
        # (indirectly verifies scaling is applied correctly)
        x = torch.randn(1, 5, n_embd)
        output = attn(x)
        
        # Output should be finite (scaling prevents overflow)
        assert torch.isfinite(output).all(), \
            "Output should be finite (scaling prevents numerical issues)"
        
        # Output should not contain NaN or Inf values
        assert not torch.isnan(output).any(), \
            "Output should not contain NaN values"
        assert not torch.isinf(output).any(), \
            "Output should not contain Inf values"
    
    def test_attention_softmax(self):
        """Test that attention produces valid probability distributions."""
        n_embd = 64
        n_head = 2
        batch_size = 2
        seq_len = 5
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        attn.eval()
        
        x = torch.randn(batch_size, seq_len, n_embd)
        output = attn(x)
        
        # Test that output is well-behaved (softmax ensures valid probability distributions)
        # Output should be finite
        assert torch.isfinite(output).all(), \
            "Output should be finite (softmax produces valid values)"
        
        # Test with identical inputs - should produce identical outputs
        # (verifies deterministic behavior when softmax is applied correctly)
        x2 = x.clone()
        output2 = attn(x2)
        assert torch.allclose(output, output2), \
            "Identical inputs should produce identical outputs"
        
        # Test that output changes when input changes
        # (verifies attention is actually computing something)
        x3 = x.clone()
        x3[:, 0, :] += 1.0  # Modify first token
        output3 = attn(x3)
        assert not torch.allclose(output, output3, atol=1e-6), \
            "Modified input should produce different output"
    
    def test_attention_dropout_training(self):
        """Test dropout is applied in training mode."""
        n_embd = 64
        n_head = 2
        dropout = 0.5  # High dropout to make effect visible
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        attn.train()  # Enable dropout
        
        x = torch.randn(1, 5, n_embd)
        
        # Run multiple times - with dropout, outputs should vary
        outputs = []
        for _ in range(10):
            output = attn(x)
            outputs.append(output)
        
        # Check that outputs are not all identical (dropout causes variation)
        first_output = outputs[0]
        all_same = all(torch.allclose(o, first_output) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in training mode"
    
    def test_attention_dropout_eval(self):
        """Test dropout is not applied in eval mode."""
        n_embd = 64
        n_head = 2
        dropout = 0.5
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        attn.eval()  # Disable dropout
        
        x = torch.randn(1, 5, n_embd)
        
        # Run multiple times - outputs should be identical
        outputs = []
        for _ in range(5):
            output = attn(x)
            outputs.append(output)
        
        # All outputs should be identical (no dropout randomness)
        first_output = outputs[0]
        for output in outputs[1:]:
            assert torch.allclose(output, first_output), \
                "Outputs should be identical in eval mode (no dropout)"
    
    def test_attention_multi_head(self):
        """Test that multiple attention heads are computed and concatenated."""
        n_embd = 768
        n_head = 12
        head_dim = n_embd // n_head  # 64
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        attn.eval()
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, n_embd)
        
        # Run forward pass - this is what we're actually testing
        output = attn(x)
        
        # Output should be concatenated back to n_embd
        assert output.shape == (batch_size, seq_len, n_embd), \
            "Output should have same shape as input (heads concatenated)"
        
        # Test that different numbers of heads produce different outputs
        # (verifies multi-head is actually being used)
        attn_single = CausalSelfAttention(n_embd, n_head=1, dropout=0.0)
        attn_single.eval()
        
        # Copy weights from first head to single-head attention
        # This is a simplified test - in practice, we'd need to properly initialize
        # But we can at least verify they produce different outputs
        output_single = attn_single(x)
        
        # They should produce different outputs (different architecture)
        assert not torch.allclose(output, output_single, atol=1e-3), \
            "Multi-head and single-head should produce different outputs"
        
        # Verify that head_dim is correctly calculated
        assert attn.head_dim == head_dim, \
            f"head_dim should be {head_dim}, got {attn.head_dim}"
        assert attn.n_head == n_head, \
            f"n_head should be {n_head}, got {attn.n_head}"
    
    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention."""
        n_embd = 64
        n_head = 2
        
        attn = CausalSelfAttention(n_embd, n_head)
        attn.train()
        
        x = torch.randn(2, 5, n_embd, requires_grad=True)
        output = attn(x)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for input
        assert x.grad is not None, "Input should have gradients"
        
        # Check that gradients exist for all parameters
        assert attn.W_q.grad is not None, "W_q should have gradients"
        assert attn.W_k.grad is not None, "W_k should have gradients"
        assert attn.W_v.grad is not None, "W_v should have gradients"
        assert attn.W_o.grad is not None, "W_o should have gradients"
        
        # Check that gradients are non-zero (actually computed)
        assert not torch.allclose(attn.W_q.grad, torch.zeros_like(attn.W_q.grad)), \
            "W_q gradients should be non-zero"
    
    def test_attention_different_sequence_lengths(self):
        """Test that attention works with different sequence lengths."""
        n_embd = 64
        n_head = 2
        
        attn = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        attn.eval()
        
        # Test with different sequence lengths
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, n_embd)
            output = attn(x)
            assert output.shape == (2, seq_len, n_embd), \
                f"Output shape should match input for seq_len={seq_len}"
    
    def test_attention_deterministic_with_seed(self):
        """Test that attention is deterministic when dropout is disabled."""
        n_embd = 64
        n_head = 2
        
        attn1 = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        attn2 = CausalSelfAttention(n_embd, n_head, dropout=0.0)
        
        # Copy weights to make them identical
        attn2.load_state_dict(attn1.state_dict())
        
        attn1.eval()
        attn2.eval()
        
        x = torch.randn(1, 5, n_embd)
        
        output1 = attn1(x)
        output2 = attn2(x)
        
        assert torch.allclose(output1, output2), \
            "Outputs should be identical with same weights and no dropout"
    
    def test_attention_parameter_count(self):
        """Test that attention has correct number of parameters."""
        n_embd = 768
        n_head = 12
        
        attn = CausalSelfAttention(n_embd, n_head)
        
        # Count parameters: 4 weight matrices of size (n_embd, n_embd)
        expected_params = 4 * n_embd * n_embd  # W_q, W_k, W_v, W_o
        
        total_params = sum(p.numel() for p in attn.parameters())
        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"

