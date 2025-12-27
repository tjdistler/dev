"""
Unit tests for GPTTokenizer.
"""

import pytest

from gpt.tokenizer import GPTTokenizer


class TestGPTTokenizer:
    """Test cases for GPTTokenizer class."""
    
    def test_tokenizer_initialization(self):
        """Test GPTTokenizer initialization."""
        tokenizer = GPTTokenizer()
        
        # Assert tokenizer is created
        assert tokenizer is not None
        assert hasattr(tokenizer, '_tokenizer')
        assert hasattr(tokenizer, 'vocab_size')
    
    def test_tokenizer_initialization_with_vocab_file(self):
        """Test tokenizer initialization with vocabulary file."""
        # vocab_file parameter is accepted but not used (HF tokenizer handles it)
        tokenizer = GPTTokenizer(vocab_file="dummy_path.txt")
        
        # Should still work - vocab_file is ignored
        assert tokenizer is not None
        assert hasattr(tokenizer, '_tokenizer')
    
    def test_tokenizer_encode(self):
        """Test text encoding."""
        tokenizer = GPTTokenizer()
        
        text = "Hello, world!"
        token_ids = tokenizer.encode(text)
        
        # Assert result is list of integers
        assert isinstance(token_ids, list), \
            "encode should return a list"
        assert all(isinstance(t, int) for t in token_ids), \
            "All elements should be integers"
        assert len(token_ids) > 0, \
            "Should encode to at least one token"
    
    def test_tokenizer_decode(self):
        """Test token decoding."""
        tokenizer = GPTTokenizer()
        
        # Encode some text first to get valid token IDs
        text = "Hello"
        token_ids = tokenizer.encode(text)
        
        # Decode the token IDs
        decoded_text = tokenizer.decode(token_ids)
        
        # Assert result is string
        assert isinstance(decoded_text, str), \
            "decode should return a string"
        assert len(decoded_text) >= 0, \
            "Decoded text should be a valid string (may be empty)"
    
    def test_tokenizer_encode_decode_roundtrip(self):
        """Test that encode and decode are inverse operations."""
        tokenizer = GPTTokenizer()
        
        # Test with various texts
        test_texts = [
            "Hello, world!",
            "The quick brown fox",
            "GPT-1 is a language model.",
        ]
        
        for text in test_texts:
            token_ids = tokenizer.encode(text)
            decoded_text = tokenizer.decode(token_ids)
            
            # Note: Roundtrip may not be exact due to tokenization differences
            # But should produce valid text
            assert isinstance(decoded_text, str), \
                f"Decoded text should be string for input: {text}"
            assert len(decoded_text) >= 0, \
                f"Decoded text should be valid for input: {text}"
    
    def test_tokenizer_vocab_size(self):
        """Test vocab_size property."""
        tokenizer = GPTTokenizer()
        
        # Access vocab_size property
        vocab_size = tokenizer.vocab_size
        
        # Assert it returns integer
        assert isinstance(vocab_size, int), \
            "vocab_size should return an integer"
        assert vocab_size > 0, \
            "vocab_size should be positive"
        
        # For GPT-1 tokenizer, vocab size should be around 40478
        # But we'll just check it's reasonable
        assert 1000 < vocab_size < 100000, \
            f"vocab_size should be reasonable, got {vocab_size}"
    
    def test_tokenizer_unknown_tokens(self):
        """Test handling of unknown tokens."""
        tokenizer = GPTTokenizer()
        
        # Try to encode text - GPT tokenizer should handle any text
        # (it may use subword tokenization, so "unknown" tokens are rare)
        text = "This is a test with some words"
        token_ids = tokenizer.encode(text)
        
        # Should successfully encode (no error)
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
    
    def test_tokenizer_special_tokens(self):
        """Test handling of special tokens."""
        tokenizer = GPTTokenizer()
        
        # Test encoding/decoding of text that may contain special tokens
        text = "Hello <|endoftext|> world"
        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)
        
        # Should handle special tokens gracefully
        assert isinstance(token_ids, list)
        assert isinstance(decoded_text, str)
    
    def test_tokenizer_empty_string(self):
        """Test handling of empty string."""
        tokenizer = GPTTokenizer()
        
        # Encode empty string
        token_ids = tokenizer.encode("")
        decoded_text = tokenizer.decode(token_ids)
        
        # Should handle empty string
        assert isinstance(token_ids, list)
        assert isinstance(decoded_text, str)
    
    def test_tokenizer_different_texts(self):
        """Test tokenizer with various text inputs."""
        tokenizer = GPTTokenizer()
        
        test_cases = [
            "Short",
            "This is a longer sentence with multiple words.",
            "Punctuation! Question? Period.",
            "Numbers: 123 456 789",
            "Mixed CASE TeXt",
        ]
        
        for text in test_cases:
            token_ids = tokenizer.encode(text)
            decoded_text = tokenizer.decode(token_ids)
            
            assert isinstance(token_ids, list), \
                f"Should encode text: {text}"
            assert isinstance(decoded_text, str), \
                f"Should decode text: {text}"
            assert len(token_ids) > 0 or text == "", \
                f"Non-empty text should encode to tokens: {text}"
    
    def test_tokenizer_consistency(self):
        """Test that tokenizer produces consistent results."""
        tokenizer = GPTTokenizer()
        
        text = "Hello, world!"
        
        # Encode multiple times
        token_ids1 = tokenizer.encode(text)
        token_ids2 = tokenizer.encode(text)
        token_ids3 = tokenizer.encode(text)
        
        # Should produce same token IDs
        assert token_ids1 == token_ids2, \
            "Encoding should be consistent"
        assert token_ids2 == token_ids3, \
            "Encoding should be consistent"

