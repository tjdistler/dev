"""
Unit tests for GPTEmbeddings.
"""

import pytest
import torch
import torch.nn as nn

from gpt.embeddings import GPTEmbeddings


class TestGPTEmbeddings:
    """Test cases for GPTEmbeddings class."""
    
    def test_embeddings_initialization(self):
        """Test GPTEmbeddings initialization."""
        vocab_size = 1000
        n_embd = 768
        n_positions = 512
        embd_pdrop = 0.1
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=embd_pdrop)
        
        # Assert token embedding exists
        assert hasattr(emb, 'token_embeddings')
        assert isinstance(emb.token_embeddings, nn.Parameter)
        assert emb.token_embeddings.shape == (vocab_size, n_embd)
        
        # Assert positional embedding exists
        assert hasattr(emb, 'position_embeddings')
        assert isinstance(emb.position_embeddings, nn.Parameter)
        assert emb.position_embeddings.shape == (n_positions, n_embd)
        
        # Assert dropout layer exists
        assert hasattr(emb, 'dropout')
        assert isinstance(emb.dropout, nn.Dropout)
        assert emb.dropout.p == embd_pdrop
    
    def test_embeddings_forward_shape(self):
        """Test that forward pass returns correct shape."""
        vocab_size = 1000
        n_embd = 768
        n_positions = 512
        batch_size = 2
        seq_len = 10
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = emb(input_ids)
        
        assert output.shape == (batch_size, seq_len, n_embd)
    
    def test_embeddings_token_embedding(self):
        """Test that token embeddings are applied correctly."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Create input with specific token IDs
        input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        output = emb(input_ids)
        
        # Verify that token embeddings are used
        # Token 0 should use emb.token_embeddings[0]
        token_0_embedding = emb.token_embeddings[0]
        assert torch.allclose(output[0, 0, :], token_0_embedding + emb.position_embeddings[0], atol=1e-5), \
            "Token embeddings should be correctly indexed"
    
    def test_embeddings_positional_embedding(self):
        """Test that positional embeddings are added."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Create input with same token at different positions
        input_ids = torch.tensor([[5, 5, 5], [5, 5, 5]])
        output = emb(input_ids)
        
        # Same token at different positions should have different embeddings
        # (due to positional embeddings)
        assert not torch.allclose(output[0, 0, :], output[0, 1, :], atol=1e-3), \
            "Same token at different positions should have different embeddings"
    
    def test_embeddings_different_positions(self):
        """Test that different positions have different embeddings."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Create input with same tokens at different positions
        token_id = 10
        input_ids = torch.tensor([[token_id, token_id], [token_id, token_id]])
        output = emb(input_ids)
        
        # Position 0 and position 1 should have different embeddings
        pos_0_embedding = output[0, 0, :]
        pos_1_embedding = output[0, 1, :]
        
        assert not torch.allclose(pos_0_embedding, pos_1_embedding, atol=1e-3), \
            "Different positions should have different embeddings"
    
    def test_embeddings_dropout_training(self):
        """Test dropout is applied in training mode."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        dropout = 0.5
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=dropout)
        emb.train()
        
        input_ids = torch.randint(0, vocab_size, (1, 10))
        
        # Run multiple times - with dropout, outputs should vary
        outputs = []
        for _ in range(10):
            output = emb(input_ids)
            outputs.append(output)
        
        # Check that outputs are not all identical (dropout causes variation)
        first_output = outputs[0]
        all_same = all(torch.allclose(o, first_output) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in training mode"
    
    def test_embeddings_dropout_eval(self):
        """Test dropout is not applied in eval mode."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        dropout = 0.5
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=dropout)
        emb.eval()
        
        input_ids = torch.randint(0, vocab_size, (1, 10))
        
        # Run multiple times - outputs should be identical
        outputs = []
        for _ in range(5):
            output = emb(input_ids)
            outputs.append(output)
        
        # All outputs should be identical (no dropout randomness)
        first_output = outputs[0]
        for output in outputs[1:]:
            assert torch.allclose(output, first_output), \
                "Outputs should be identical in eval mode (no dropout)"
    
    def test_embeddings_vocab_size(self):
        """Test that vocab_size is respected."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Try to use token_id >= vocab_size
        # This should work (no error), but will use out-of-bounds indexing
        # In practice, token IDs should be in range [0, vocab_size)
        input_ids = torch.tensor([[0, vocab_size - 1]])  # Valid range
        
        # This should work
        output = emb(input_ids)
        assert output.shape == (1, 2, n_embd)
    
    def test_embeddings_max_position(self):
        """Test that max position is respected."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Create input up to n_positions (the max)
        # The implementation uses [:input_ids.shape[1]], so it slices positional embeddings
        seq_len = n_positions  # Use exactly n_positions
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        
        # This should work - it will use all n_positions positional embeddings
        output = emb(input_ids)
        assert output.shape == (1, seq_len, n_embd)
        
        # Test with shorter sequence
        seq_len_short = n_positions // 2
        input_ids_short = torch.randint(0, vocab_size, (1, seq_len_short))
        output_short = emb(input_ids_short)
        assert output_short.shape == (1, seq_len_short, n_embd)
    
    def test_embeddings_gradient_flow(self):
        """Test that gradients flow through embeddings."""
        vocab_size = 100
        n_embd = 64
        n_positions = 50
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions)
        emb.train()
        
        input_ids = torch.randint(0, vocab_size, (2, 10))
        output = emb(input_ids)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for embedding parameters
        assert emb.token_embeddings.grad is not None, \
            "Token embeddings should have gradients"
        assert emb.position_embeddings.grad is not None, \
            "Position embeddings should have gradients"
        
        # Check that gradients are non-zero
        assert not torch.allclose(emb.token_embeddings.grad, 
                                 torch.zeros_like(emb.token_embeddings.grad)), \
            "Token embedding gradients should be non-zero"
    
    def test_embeddings_different_sequence_lengths(self):
        """Test that embeddings work with different sequence lengths."""
        vocab_size = 100
        n_embd = 64
        n_positions = 512
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions, embd_pdrop=0.0)
        emb.eval()
        
        # Test different sequence lengths
        for seq_len in [1, 5, 10, 50, 100]:
            input_ids = torch.randint(0, vocab_size, (2, seq_len))
            output = emb(input_ids)
            assert output.shape == (2, seq_len, n_embd), \
                f"Output shape should match input for seq_len={seq_len}"
    
    def test_embeddings_parameter_count(self):
        """Test that embeddings have correct number of parameters."""
        vocab_size = 1000
        n_embd = 768
        n_positions = 512
        
        emb = GPTEmbeddings(vocab_size, n_embd, n_positions)
        
        # Count parameters: token_embeddings (vocab_size * n_embd) + 
        #                    position_embeddings (n_positions * n_embd)
        expected_params = vocab_size * n_embd + n_positions * n_embd
        
        total_params = sum(p.numel() for p in emb.parameters())
        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"

