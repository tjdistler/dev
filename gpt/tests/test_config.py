"""
Unit tests for GPTConfig.
"""

import pytest

from gpt.config import GPTConfig


class TestGPTConfig:
    """Test cases for GPTConfig class."""
    
    def test_default_config(self):
        """Test that default config is created correctly."""
        config = GPTConfig()
        
        # Assert all default values are correct
        assert config.vocab_size == 40478
        assert config.n_positions == 512
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner == 4 * config.n_embd  # Should be set in __post_init__
        assert config.embd_pdrop == 0.1
        assert config.resid_pdrop == 0.1
        assert config.attn_pdrop == 0.1
        assert config.activation == "gelu"
        assert config.initializer_range == 0.02
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        custom_config = GPTConfig(
            vocab_size=5000,
            n_positions=256,
            n_embd=512,
            n_layer=6,
            n_head=8,
            n_inner=2048,
            embd_pdrop=0.2,
            resid_pdrop=0.2,
            attn_pdrop=0.2,
            activation='relu',
            initializer_range=0.01
        )
        
        # Assert all custom values are set correctly
        assert custom_config.vocab_size == 5000
        assert custom_config.n_positions == 256
        assert custom_config.n_embd == 512
        assert custom_config.n_layer == 6
        assert custom_config.n_head == 8
        assert custom_config.n_inner == 2048
        assert custom_config.embd_pdrop == 0.2
        assert custom_config.resid_pdrop == 0.2
        assert custom_config.attn_pdrop == 0.2
        assert custom_config.activation == 'relu'
        assert custom_config.initializer_range == 0.01
    
    def test_config_validation_n_embd_divisible_by_n_head(self):
        """Test that n_embd must be divisible by n_head."""
        # Try creating config with n_embd not divisible by n_head
        with pytest.raises(AssertionError, match="must be divisible"):
            GPTConfig(n_embd=768, n_head=7)  # 768 is not divisible by 7
    
    def test_config_validation_n_embd_divisible_by_n_head_success(self):
        """Test that valid n_embd and n_head combination works."""
        # Create config with n_embd divisible by n_head
        config = GPTConfig(n_embd=768, n_head=12)  # 768 is divisible by 12
        
        # Assert no error is raised and values are set correctly
        assert config.n_embd == 768
        assert config.n_head == 12
    
    def test_config_n_inner_default(self):
        """Test that n_inner defaults to 4 * n_embd."""
        n_embd = 768
        config = GPTConfig(n_embd=n_embd, n_inner=None)
        
        assert config.n_inner == 4 * n_embd, \
            "n_inner should default to 4 * n_embd"
    
    def test_config_n_inner_custom(self):
        """Test that custom n_inner is respected."""
        n_embd = 768
        custom_n_inner = 2048
        
        config = GPTConfig(n_embd=n_embd, n_inner=custom_n_inner)
        
        assert config.n_inner == custom_n_inner, \
            "Custom n_inner should be set correctly"
    
    def test_config_from_tokenizer(self):
        """Test creating config from tokenizer."""
        # Create a mock tokenizer object
        class MockTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
        
        tokenizer = MockTokenizer(vocab_size=10000)
        
        config = GPTConfig.from_tokenizer(tokenizer)
        
        assert config.vocab_size == 10000, \
            "vocab_size should be set from tokenizer"
        
        # Other defaults should still apply
        assert config.n_embd == 768
        assert config.n_layer == 12
    
    def test_config_from_tokenizer_with_overrides(self):
        """Test creating config from tokenizer with parameter overrides."""
        class MockTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
        
        tokenizer = MockTokenizer(vocab_size=5000)
        
        # Use n_head that divides n_embd
        config = GPTConfig.from_tokenizer(
            tokenizer,
            n_embd=512,
            n_head=8,  # 512 is divisible by 8
            n_layer=6
        )
        
        assert config.vocab_size == 5000, \
            "vocab_size should be set from tokenizer"
        assert config.n_embd == 512, \
            "n_embd should be overridden"
        assert config.n_layer == 6, \
            "n_layer should be overridden"
    
    def test_config_various_valid_combinations(self):
        """Test various valid n_embd and n_head combinations."""
        valid_combinations = [
            (768, 12),   # GPT-1 default
            (512, 8),    # Smaller model
            (1024, 16),  # Larger model
            (256, 4),    # Small model
            (384, 6),    # Another valid combination
        ]
        
        for n_embd, n_head in valid_combinations:
            config = GPTConfig(n_embd=n_embd, n_head=n_head)
            assert config.n_embd == n_embd
            assert config.n_head == n_head
            assert config.n_embd % config.n_head == 0, \
                f"n_embd ({n_embd}) should be divisible by n_head ({n_head})"

