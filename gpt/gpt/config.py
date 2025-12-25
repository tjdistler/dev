"""
Model configuration for GPT-1.

Defines the hyperparameters and architecture settings for the GPT-1 model.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration class for GPT-1 model."""
    
    # Model dimensions
    vocab_size: int = 40478  # Vocabulary size
    n_positions: int = 512   # Maximum sequence length
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads
    
    # Dropout
    embd_pdrop: float = 0.1  # Embedding dropout
    resid_pdrop: float = 0.1  # Residual dropout
    attn_pdrop: float = 0.1   # Attention dropout
    
    # Activation function
    activation: str = "gelu"  # Activation function type
    
    # Initialization
    initializer_range: float = 0.02  # Range for weight initialization
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

