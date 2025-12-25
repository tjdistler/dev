"""
Tokenizer for GPT-1.

Handles text tokenization and detokenization.
"""


class GPTTokenizer:
    """
    Tokenizer for GPT-1 model.
    
    Handles conversion between text and token indices.
    """
    
    def __init__(self, vocab_file=None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file (optional)
        """
        # TODO: Load vocabulary if vocab_file provided
        # TODO: Create token-to-id and id-to-token mappings
        # TODO: Initialize special tokens (if any)
        pass
    
    def encode(self, text):
        """
        Encode text into token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        # TODO: Tokenize text
        # TODO: Convert tokens to indices
        # TODO: Return list of token indices
        pass
    
    def decode(self, token_ids):
        """
        Decode token indices back to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded text string
        """
        # TODO: Convert indices to tokens
        # TODO: Join tokens into text
        # TODO: Return decoded text
        pass
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        # TODO: Return size of vocabulary
        pass

