"""
Tokenizer for GPT-1.

Handles text tokenization and detokenization using Hugging Face's OpenAIGPTTokenizer.
"""

from transformers import OpenAIGPTTokenizer


class GPTTokenizer:
    """
    Tokenizer for GPT-1 model.
    
    Wrapper around Hugging Face's OpenAIGPTTokenizer for GPT-1 compatibility.
    Handles conversion between text and token indices.
    """
    
    def __init__(self, vocab_file=None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file (optional, not used with HF tokenizer)
        """
        # Load the pre-trained GPT-1 tokenizer from Hugging Face
        self._tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    
    def encode(self, text):
        """
        Encode text into token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        return self._tokenizer.encode(text)
    
    def decode(self, token_ids):
        """
        Decode token indices back to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self._tokenizer)

