"""
Model configuration for GPT-1.

Defines the hyperparameters and architecture settings for the GPT-1 model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration class for GPT-1 model."""
    
    # Model dimensions
    vocab_size: int = 40478  # Vocabulary size
    n_positions: int = 512   # Maximum sequence length
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads
    n_inner: Optional[int] = None  # Inner dimension for MLP (default: 4 * n_embd)
    
    # Dropout
    embd_pdrop: float = 0.1  # Embedding dropout probability
    resid_pdrop: float = 0.1  # Residual dropout probability
    attn_pdrop: float = 0.1   # Attention dropout probability
    
    # Activation function
    activation: str = "gelu"  # Activation function type
    
    # Initialization
    initializer_range: float = 0.02  # Range for weight initialization
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        
        # Set n_inner to default (4 * n_embd) if not specified
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
    
    @classmethod
    def from_tokenizer(cls, tokenizer, **kwargs):
        """
        Create GPTConfig with vocab_size automatically set from tokenizer.
        
        Args:
            tokenizer: GPTTokenizer instance (or any object with vocab_size property)
            **kwargs: Additional configuration parameters to override defaults
            
        Returns:
            GPTConfig instance with vocab_size set from tokenizer
        """
        vocab_size = tokenizer.vocab_size
        return cls(vocab_size=vocab_size, **kwargs)

