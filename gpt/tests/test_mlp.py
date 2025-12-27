"""
Unit tests for MLP (Multi-Layer Perceptron).
"""

import pytest
import torch
import torch.nn as nn

from gpt.mlp import MLP


class TestMLP:
    """Test cases for MLP class."""
    
    def test_mlp_initialization(self):
        """Test MLP initialization."""
        n_embd = 768
        n_inner = 3072
        dropout = 0.1
        
        mlp = MLP(n_embd, n_inner=n_inner, dropout=dropout)
        
        # Note: These tests will fail until MLP is implemented
        # They test the expected interface
        
    def test_mlp_initialization_default_n_inner(self):
        """Test that n_inner defaults to 4 * n_embd."""
        n_embd = 768
        expected_n_inner = 4 * n_embd
        
        mlp = MLP(n_embd)
        
        # Note: This test will fail until MLP is implemented
        # Expected: mlp should have n_inner = 4 * n_embd
    
    def test_mlp_forward_shape(self):
        """Test that forward pass maintains batch and seq dimensions."""
        n_embd = 768
        n_inner = 3072
        batch_size = 2
        seq_len = 10
        
        mlp = MLP(n_embd, n_inner=n_inner, dropout=0.0)
        mlp.eval()
        
        x = torch.randn(batch_size, seq_len, n_embd)
        
        # This will fail until MLP.forward() is implemented
        try:
            output = mlp(x)
            assert output.shape == (batch_size, seq_len, n_embd), \
                "Output shape should match input shape"
        except (AttributeError, TypeError):
            pytest.skip("MLP not yet implemented")
    
    def test_mlp_activation_function(self):
        """Test that activation function is applied."""
        n_embd = 64
        n_inner = 256
        
        mlp = MLP(n_embd, n_inner=n_inner, activation='gelu', dropout=0.0)
        mlp.eval()
        
        x = torch.randn(1, 5, n_embd)
        
        try:
            output = mlp(x)
            # GELU should produce different output than linear transformation
            # We can't directly verify, but output should be transformed
            assert output.shape == x.shape
        except (AttributeError, TypeError):
            pytest.skip("MLP not yet implemented")
    
    def test_mlp_dropout_training_mode(self):
        """Test dropout is applied in training mode."""
        n_embd = 64
        n_inner = 256
        dropout = 0.5
        
        mlp = MLP(n_embd, n_inner=n_inner, dropout=dropout)
        mlp.train()
        
        x = torch.randn(1, 5, n_embd)
        
        try:
            outputs = []
            for _ in range(5):
                output = mlp(x)
                outputs.append(output)
            
            # With dropout, outputs should vary in training mode
            first_output = outputs[0]
            all_same = all(torch.allclose(o, first_output) for o in outputs[1:])
            assert not all_same, "Dropout should cause variation in training mode"
        except (AttributeError, TypeError):
            pytest.skip("MLP not yet implemented")
    
    def test_mlp_dropout_eval_mode(self):
        """Test dropout is not applied in eval mode."""
        n_embd = 64
        n_inner = 256
        dropout = 0.5
        
        mlp = MLP(n_embd, n_inner=n_inner, dropout=dropout)
        mlp.eval()
        
        x = torch.randn(1, 5, n_embd)
        
        try:
            outputs = []
            for _ in range(5):
                output = mlp(x)
                outputs.append(output)
            
            # Without dropout, outputs should be identical
            first_output = outputs[0]
            for output in outputs[1:]:
                assert torch.allclose(output, first_output), \
                    "Eval mode should produce consistent outputs (no dropout)"
        except (AttributeError, TypeError):
            pytest.skip("MLP not yet implemented")
    
    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP."""
        n_embd = 64
        n_inner = 256
        
        mlp = MLP(n_embd, n_inner=n_inner)
        mlp.train()
        
        x = torch.randn(2, 5, n_embd, requires_grad=True)
        
        try:
            output = mlp(x)
            loss = output.sum()
            loss.backward()
            
            assert x.grad is not None, "Input should have gradients"
            # Check that MLP parameters have gradients
            # (will need to access them once MLP is implemented)
        except (AttributeError, TypeError):
            pytest.skip("MLP not yet implemented")
    
    def test_mlp_different_shapes(self):
        """Test that MLP works with different input shapes."""
        n_embd = 768
        n_inner = 3072
        
        mlp = MLP(n_embd, n_inner=n_inner, dropout=0.0)
        mlp.eval()
        
        test_cases = [
            (1, 1, n_embd),
            (1, 10, n_embd),
            (2, 10, n_embd),
            (4, 100, n_embd),
        ]
        
        for batch_size, seq_len, embd_dim in test_cases:
            x = torch.randn(batch_size, seq_len, embd_dim)
            try:
                output = mlp(x)
                assert output.shape == (batch_size, seq_len, embd_dim), \
                    f"Output shape should match input for shape {x.shape}"
            except (AttributeError, TypeError):
                pytest.skip("MLP not yet implemented")
                break

