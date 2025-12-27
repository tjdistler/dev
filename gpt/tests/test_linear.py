"""
Unit tests for Linear layer.
"""

import pytest
import torch
import torch.nn as nn

from gpt.linear import Linear


class TestLinear:
    """Test cases for Linear class."""
    
    def test_linear_initialization(self):
        """Test Linear initialization."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features)
        
        # Assert weight parameter exists
        assert hasattr(linear, 'weight'), "Linear should have weight parameter"
        assert isinstance(linear.weight, nn.Parameter), "Weight should be a Parameter"
        
        # Assert correct shape of weight
        assert linear.weight.shape == (out_features, in_features), \
            f"Weight shape should be ({out_features}, {in_features})"
        
        # Assert bias parameter exists (default is True)
        assert hasattr(linear, 'bias'), "Linear should have bias parameter when bias=True"
        assert isinstance(linear.bias, nn.Parameter), "Bias should be a Parameter"
        
        # Assert correct shape of bias
        assert linear.bias.shape == (out_features,), \
            f"Bias shape should be ({out_features},)"
    
    def test_linear_initialization_no_bias(self):
        """Test Linear initialization without bias."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features, bias=False)
        
        # Assert weight parameter exists
        assert hasattr(linear, 'weight'), "Linear should have weight parameter"
        assert linear.weight.shape == (out_features, in_features)
        
        # Assert bias is None when bias=False
        assert linear.bias is None, "Bias should be None when bias=False"
    
    def test_linear_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        in_features = 768
        out_features = 512
        batch_size = 2
        seq_len = 10
        
        linear = Linear(in_features, out_features)
        
        x = torch.randn(batch_size, seq_len, in_features)
        output = linear(x)
        
        assert output.shape == (batch_size, seq_len, out_features), \
            f"Output shape should be ({batch_size}, {seq_len}, {out_features})"
    
    def test_linear_forward_computation(self):
        """Test that forward pass performs correct computation."""
        in_features = 4
        out_features = 3
        
        linear = Linear(in_features, out_features, bias=True)
        
        # Set weights and bias to known values for testing
        with torch.no_grad():
            linear.weight.data = torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features)
            linear.bias.data = torch.arange(out_features, dtype=torch.float32)
        
        x = torch.ones(1, 2, in_features)  # Shape: (1, 2, 4)
        
        output = linear(x)
        
        # Expected: x @ weight.T + bias
        # x = [[1, 1, 1, 1], [1, 1, 1, 1]]
        # weight = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        # weight.T = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
        # x @ weight.T = [[6, 22, 38], [6, 22, 38]]
        # bias = [0, 1, 2]
        # output = [[6, 23, 40], [6, 23, 40]]
        expected = torch.tensor([[[6.0, 23.0, 40.0], [6.0, 23.0, 40.0]]])
        
        assert torch.allclose(output, expected), \
            "Forward pass should compute x @ weight.T + bias correctly"
    
    def test_linear_forward_no_bias(self):
        """Test forward pass without bias."""
        in_features = 4
        out_features = 3
        
        linear = Linear(in_features, out_features, bias=False)
        
        # Set weights to known values
        with torch.no_grad():
            linear.weight.data = torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features)
        
        x = torch.ones(1, 2, in_features)
        
        output = linear(x)
        
        # Expected: x @ weight.T (no bias)
        # x = [[1, 1, 1, 1], [1, 1, 1, 1]]
        # weight.T = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
        # output = [[6, 22, 38], [6, 22, 38]]
        expected = torch.tensor([[[6.0, 22.0, 38.0], [6.0, 22.0, 38.0]]])
        
        assert torch.allclose(output, expected), \
            "Forward pass without bias should compute x @ weight.T correctly"
    
    def test_linear_learnable_parameters(self):
        """Test that weight and bias are learnable parameters."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features)
        
        # Check that weight requires gradients
        assert linear.weight.requires_grad, "Weight should require gradients"
        
        # Check that bias requires gradients
        assert linear.bias.requires_grad, "Bias should require gradients"
        
        # Check that they are Parameters (not just Tensors)
        assert isinstance(linear.weight, nn.Parameter), "Weight should be a Parameter"
        assert isinstance(linear.bias, nn.Parameter), "Bias should be a Parameter"
    
    def test_linear_gradient_flow(self):
        """Test that gradients flow through Linear layer."""
        in_features = 64
        out_features = 32
        
        linear = Linear(in_features, out_features)
        linear.train()
        
        x = torch.randn(2, 5, in_features, requires_grad=True)
        
        output = linear(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        assert linear.weight.grad is not None, "Weight should have gradients"
        assert linear.bias.grad is not None, "Bias should have gradients"
    
    def test_linear_different_shapes(self):
        """Test that Linear works with different input shapes."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features)
        
        test_cases = [
            (1, in_features),  # 1D
            (10, in_features),  # 2D
            (2, 10, in_features),  # 3D
            (4, 10, 20, in_features),  # 4D
        ]
        
        for shape in test_cases:
            x = torch.randn(*shape)
            output = linear(x)
            
            expected_shape = (*shape[:-1], out_features)
            assert output.shape == expected_shape, \
                f"Output shape should be {expected_shape} for input shape {shape}"
    
    def test_linear_equivalence_to_nn_linear(self):
        """Test that custom Linear produces same output as nn.Linear."""
        in_features = 64
        out_features = 32
        batch_size = 2
        seq_len = 10
        
        # Create custom Linear and nn.Linear with same weights
        custom_linear = Linear(in_features, out_features)
        torch_linear = nn.Linear(in_features, out_features)
        
        # Copy weights and bias from custom to torch
        with torch.no_grad():
            torch_linear.weight.data = custom_linear.weight.data.clone()
            torch_linear.bias.data = custom_linear.bias.data.clone()
        
        x = torch.randn(batch_size, seq_len, in_features)
        
        custom_output = custom_linear(x)
        torch_output = torch_linear(x)
        
        assert torch.allclose(custom_output, torch_output, atol=1e-5), \
            "Custom Linear should produce same output as nn.Linear"
    
    def test_linear_equivalence_to_nn_linear_no_bias(self):
        """Test that custom Linear without bias matches nn.Linear without bias."""
        in_features = 64
        out_features = 32
        batch_size = 2
        seq_len = 10
        
        custom_linear = Linear(in_features, out_features, bias=False)
        torch_linear = nn.Linear(in_features, out_features, bias=False)
        
        # Copy weights
        with torch.no_grad():
            torch_linear.weight.data = custom_linear.weight.data.clone()
        
        x = torch.randn(batch_size, seq_len, in_features)
        
        custom_output = custom_linear(x)
        torch_output = torch_linear(x)
        
        assert torch.allclose(custom_output, torch_output, atol=1e-5), \
            "Custom Linear without bias should produce same output as nn.Linear without bias"
    
    def test_linear_weight_initialization(self):
        """Test that weight is properly initialized."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features)
        
        # Weight should not be all zeros (unless explicitly initialized that way)
        # Check that weight has reasonable values (not NaN, not Inf)
        assert not torch.isnan(linear.weight).any(), "Weight should not contain NaN"
        assert not torch.isinf(linear.weight).any(), "Weight should not contain Inf"
        
        # Weight should have correct dtype
        assert linear.weight.dtype == torch.float32, "Weight should be float32"
    
    def test_linear_bias_initialization(self):
        """Test that bias is properly initialized."""
        in_features = 768
        out_features = 512
        
        linear = Linear(in_features, out_features)
        
        # Bias should not be NaN or Inf
        assert not torch.isnan(linear.bias).any(), "Bias should not contain NaN"
        assert not torch.isinf(linear.bias).any(), "Bias should not contain Inf"
        
        # Bias should have correct dtype
        assert linear.bias.dtype == torch.float32, "Bias should be float32"
    
    def test_linear_device_consistency(self):
        """Test that Linear works on different devices if available."""
        in_features = 64
        out_features = 32
        
        linear = Linear(in_features, out_features)
        
        x = torch.randn(2, 5, in_features)
        output = linear(x)
        
        # Output should be on same device as input
        assert output.device == x.device, "Output should be on same device as input"
        
        # If CUDA is available, test on GPU
        if torch.cuda.is_available():
            linear_cuda = Linear(in_features, out_features).cuda()
            x_cuda = x.cuda()
            output_cuda = linear_cuda(x_cuda)
            
            assert output_cuda.device.type == 'cuda', "Output should be on CUDA device"
            assert output_cuda.shape == output.shape, "Output shapes should match"

