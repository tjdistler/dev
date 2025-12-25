"""
Text generation script for GPT-1 model.
"""

import torch

from gpt.model import GPT
from gpt.config import GPTConfig
from tokenizer import GPTTokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=None):
    """
    Generate text using the trained GPT model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        prompt: Starting text prompt
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (None = no top-k filtering)
        
    Returns:
        Generated text string
    """
    model.eval()
    # TODO: Encode prompt
    # TODO: Convert to tensor and move to device
    # TODO: Generate tokens one at a time:
    #   - Forward pass to get logits
    #   - Apply temperature
    #   - Apply top-k filtering if specified
    #   - Sample next token
    #   - Append to sequence
    #   - Repeat until max_length or end token
    # TODO: Decode generated tokens
    # TODO: Return generated text
    pass


def main():
    """Main generation function."""
    # TODO: Set device
    # TODO: Load configuration
    # TODO: Initialize tokenizer
    # TODO: Load trained model checkpoint
    # TODO: Generate text from prompt
    # TODO: Print generated text
    pass


if __name__ == "__main__":
    main()

