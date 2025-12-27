"""
Text generation script for GPT-1 model.
"""

import argparse
import logging
import torch

from gpt.model import GPT
from gpt.config import GPTConfig
from gpt.tokenizer import GPTTokenizer

logger = logging.getLogger(__name__)


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=None):
    """
    Generate text using the trained GPT model.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        prompt: Starting text prompt
        max_length: Maximum number of new tokens to generate (excluding the prompt)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (None = no top-k filtering)
        
    Returns:
        Generated text string (includes the original prompt)
    """
    model.eval()
    with torch.no_grad():
        logger.info(f"Prompt: {prompt}")
        in_tokens = tokenizer.encode(prompt)
        logger.debug(f"In tokens: {in_tokens}")

        # Unsqueeze adds a single "batch" dimension to the front of the tensor (shape [seq_len] -> [1, seq_len]).
        # This code also moves the tensor to the device (CPU or GPU).
        in_tensor = torch.tensor(in_tokens).unsqueeze(0).to(model.device)
        logger.debug(f"In tensor: {in_tensor.shape}")

        generated_tokens = model.generate(in_tensor, max_length=max_length, temperature=temperature, top_k=top_k)
        logger.debug(f"Generated tokens: {generated_tokens}")
        generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())
    
    return generated_text


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate text using GPT-1 model')
    parser.add_argument(
        'prompt',
        type=str,
        help='Text prompt to generate from'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Enable verbose logging: -v for INFO, -vv for DEBUG (default: WARNING)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set logging level (default: WARNING)'
    )
    parser.add_argument(
        '--max-length', '-m',
        type=int,
        default=100,
        help='Maximum number of new tokens to generate, excluding the prompt (default: 100)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Sampling temperature - higher = more random (default: 1.0)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=None,
        help='Top-k sampling - limit sampling to top k tokens (default: None, no limit)'
    )
    args = parser.parse_args()
    
    # Configure logging: default to WARNING, allow INFO via -v, DEBUG via -vv, or --log-level
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    else:
        log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPTTokenizer()
    config = GPTConfig.from_tokenizer(tokenizer)
    model = GPT(config, device=device)
    logger.info(f"Model device: {model.device}, vocab size: {model.config.vocab_size}")

    generated_text = generate_text(model, tokenizer, args.prompt, max_length=args.max_length, 
                                   temperature=args.temperature, top_k=args.top_k)
    print(f">>> {generated_text}")


if __name__ == "__main__":
    main()

