"""
Main GPT-1 Model.

Implements the complete GPT-1 transformer model.
"""

import logging
import torch
import torch.nn as nn

from .config import GPTConfig
from .embeddings import GPTEmbeddings
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm
from .linear import Linear

logger = logging.getLogger(__name__)


class GPT(nn.Module):
    """
    GPT-1 Language Model.
    
    Complete transformer-based language model architecture.
    
    Args:
        config: GPTConfig instance with model hyperparameters
        device: Device to run the model on (torch.device or str, default: 'cpu')
    """
    
    def __init__(self, config: GPTConfig, device=None):
        super().__init__()
        self.config = config
        
        # Set device (default to CPU if not specified)
        # The device determines where tensors are allocated (CPU or GPU)
        # This is used for moving tensors to the correct device during forward pass
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        # Initialize embeddings layer: combines token and positional embeddings
        # Token embeddings map each token ID to a dense vector representation, while
        # positional embeddings encode the position of each token in the sequence.
        # The embeddings layer handles both lookups and applies dropout for regularization.
        # Output shape: (batch_size, seq_len, n_embd)
        self.embeddings = GPTEmbeddings(
            config.vocab_size, 
            config.n_embd, 
            config.n_positions, 
            config.embd_pdrop
        )

        # Initialize transformer blocks: stack of n_layer identical transformer decoder blocks
        # Each block implements the post-norm architecture:
        #   1. Multi-head causal self-attention with residual connection, followed by layer norm
        #   2. Position-wise MLP (feed-forward network) with residual connection, followed by layer norm
        # The blocks process the sequence in parallel (unlike RNNs), allowing efficient training.
        # Each block refines the representations, building up increasingly abstract features.
        # ModuleList is used to store the blocks as a list of modules.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.n_embd, 
                config.n_head, 
                config.n_inner, 
                config.resid_pdrop, 
                config.activation
            ) 
            for _ in range(config.n_layer)
        ])

        # Initialize final layer norm: applied after all transformer blocks
        # Layer normalization stabilizes the activations by normalizing across the embedding dimension.
        # This final normalization ensures the representations are in a well-conditioned space
        # before being projected to vocabulary logits. It helps with training stability and
        # can improve generalization by reducing internal covariate shift.
        self.ln_f = LayerNorm(config.n_embd)
        
        # Initialize language modeling head: projects final embeddings to vocabulary logits
        # This is a linear transformation that maps from the embedding dimension (n_embd)
        # to the vocabulary size (vocab_size). The output logits represent unnormalized
        # probabilities for each token in the vocabulary at each position.
        # bias=False: GPT-1 typically doesn't use bias in the output projection
        # The weights of this layer are often tied with the token embedding weights to
        # reduce parameters and improve generalization (not implemented here for simplicity).
        # Pass initializer_range so the Linear layer uses the same initialization scheme as the model.
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False, initializer_range=config.initializer_range)
        
        # Apply weight initialization to all parameters
        # Proper initialization is crucial for training stability and convergence.
        # The _init_weights method will be called recursively on all submodules.
        self.apply(self._init_weights)
        
    def forward(self, input_ids, targets=None):
        """
        Forward pass through the model.
        
        The forward pass follows this pipeline:
        1. Embed input token IDs (token + positional embeddings)
        2. Pass through stack of transformer blocks (attention + MLP)
        3. Apply final layer normalization
        4. Project to vocabulary logits via language modeling head
        5. Optionally compute cross-entropy loss if targets are provided
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
                Each element is an integer token ID from the vocabulary.
            targets: Optional target token indices for loss computation
                Shape: (batch_size, seq_len). If provided, loss is computed.
                Typically, targets are the input_ids shifted by one position
                (next token prediction task).
            
        Returns:
            If targets is None: 
                logits of shape (batch_size, seq_len, vocab_size)
                Logits are unnormalized scores for each vocabulary token at each position.
            If targets is provided: 
                tuple of (logits, loss)
                loss is a scalar tensor containing the average cross-entropy loss.
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Get embeddings from input token IDs
        # The embeddings layer performs two lookups:
        #   - Token embeddings: maps each token ID to its embedding vector
        #   - Positional embeddings: adds position information to each token
        # These are combined (added together) and dropout is applied for regularization.
        # Input shape: (batch_size, seq_len) -> Output shape: (batch_size, seq_len, n_embd)
        embeddings = self.embeddings(input_ids)
        logger.debug(f"Embeddings shape: {embeddings.shape}")

        # Step 2: Pass through transformer blocks sequentially
        # Each transformer block refines the representations through:
        #   - Multi-head causal self-attention: allows tokens to attend to previous tokens
        #   - Layer normalization: stabilizes activations
        #   - Position-wise MLP: applies non-linear transformation
        #   - Layer normalization: further stabilizes activations
        # The blocks are applied sequentially, with each block building on the output
        # of the previous block. This creates a hierarchy of increasingly abstract representations.
        # The causal mask in attention ensures tokens can only attend to previous tokens,
        # maintaining the autoregressive property required for language modeling.
        # Shape remains constant: (batch_size, seq_len, n_embd) through all blocks
        for i, block in enumerate(self.blocks):
            embeddings = block(embeddings)
            logger.debug(f"Embeddings after block {i}: {embeddings.shape}")

        # Step 3: Apply final layer normalization
        # After all transformer blocks, we normalize the final representations.
        # This ensures the activations are in a well-conditioned space before projection.
        # Layer norm normalizes across the embedding dimension (last dimension),
        # computing mean and variance for each position independently.
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_embd)
        embeddings = self.ln_f(embeddings)
        
        # Step 4: Project to vocabulary logits via language modeling head
        # The language modeling head is a linear transformation that maps from the
        # embedding dimension to the vocabulary size. Each position in the sequence
        # gets a vector of logits, one for each token in the vocabulary.
        # These logits represent unnormalized scores - higher values indicate the model
        # thinks that token is more likely at that position.
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(embeddings)
        
        # Step 5: Compute loss if targets are provided (training mode)
        # Cross-entropy loss measures how well the model's predictions match the targets.
        # The loss is computed by comparing the predicted logits to the true next tokens.
        # We reshape logits and targets to 2D tensors for the loss function:
        #   - logits: (batch_size * seq_len, vocab_size)
        #   - targets: (batch_size * seq_len,)
        # The loss function applies softmax internally and computes the negative log
        # likelihood of the correct token at each position.
        if targets is not None:
            # Reshape logits and targets for cross-entropy loss computation
            # Cross-entropy expects logits of shape (N, C) and targets of shape (N,)
            # where N is the number of samples and C is the number of classes
            logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            targets_flat = targets.view(-1)  # (batch_size * seq_len,)
            
            # Compute cross-entropy loss
            # This combines log_softmax and negative log-likelihood in a numerically stable way.
            # The loss is averaged over all positions in the batch.
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
            
            return logits, loss
        
        # If no targets provided (inference/evaluation mode), return only logits
        return logits
    
    def sample_next_token(self, logits, temperature=1.0, top_k=None):
        """
        Sample the next token from logits using temperature and top-k sampling.
        
        Args:
            logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_k: Top-k sampling parameter (None = no top-k filtering)
            
        Returns:
            Sampled token IDs of shape (batch_size,)
        """
        # Extract logits for the last position only (shape: batch_size, vocab_size)
        # We only care about the predictions for the next token, not all positions
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Apply top-k filtering if specified: keep only the top k most likely tokens
        if top_k is not None:
            # Get the k-th largest value for each batch item
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            # Create a mask: set all non-top-k logits to -inf
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_values)
            logits = logits_filtered
        
        # Apply temperature scaling: divide logits by temperature
        # Lower temperature = sharper distribution (more deterministic)
        # Higher temperature = flatter distribution (more random)
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
        
        # Sample one token ID from the probability distribution for each batch item
        # multinomial samples from the distribution independently for each batch
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
        
        # Squeeze to remove the last dimension: (batch_size, 1) -> (batch_size,)
        return next_token.squeeze(-1)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        """
        Generate text from input token IDs using autoregressive sampling.
        
        This method orchestrates the generation loop, calling forward() repeatedly
        to generate tokens one at a time.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (None = no top-k filtering)
            
        Returns:
            Generated text string
        """
        # Autoregressive generation loop: generate tokens one at a time by repeatedly predicting the next token.
        # The process starts with the input prompt and iteratively appends newly sampled tokens until max_length is reached.
        # For each iteration: (1) pass current sequence through model to get logits for next token position,
        # (2) sample next token from logits using temperature and top-k sampling for controlled randomness,
        # (3) append sampled token to sequence. This creates a feedback loop where each new token conditions
        # the prediction of the next token, enabling coherent text generation.
        generated_tokens = input_ids
        for _ in range(max_length):
            logger.debug(f"Iteration {_} of {max_length}")
            logits = self.forward(generated_tokens)
            next_token = self.sample_next_token(logits, temperature, top_k)
            # Reshape next_token from (batch_size,) to (batch_size, 1) for concatenation along sequence dimension
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
        return generated_tokens
    
    def _init_weights(self, module):
        """
        Initialize weights for all modules in the model.
        
        This method is called recursively on all submodules via self.apply().
        Different initialization strategies are used for different layer types:
        - Linear layers: weights initialized from normal distribution, bias to zero
        - Embeddings: initialized from normal distribution
        - LayerNorm: scale initialized to 1, bias to 0 (already done in LayerNorm.__init__)
        
        Proper initialization is crucial for training stability. GPT-1 uses a small
        initialization range to prevent activations from exploding early in training.
        
        Args:
            module: A submodule of the GPT model (Linear, LayerNorm, Parameter, etc.)
        """
        # Initialize PyTorch's built-in Linear layers (but not our custom Linear class)
        # Our custom Linear class handles its own initialization in __init__, so we skip it here.
        # Only initialize nn.Linear layers that don't have the self-initializing behavior.
        if isinstance(module, nn.Linear):
            # Initialize weight matrix from normal distribution
            # The standard deviation is set to initializer_range, which is typically 0.02
            # This creates small random values that prevent activations from being too large
            # at the start of training, which helps with gradient flow and training stability.
            torch.nn.init.normal_(
                module.weight, 
                mean=0.0, 
                std=self.config.initializer_range
            )
            
            # Initialize bias to zero if it exists
            # Starting with zero bias is a common practice - the model will learn
            # appropriate bias values during training through backpropagation.
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # Note: Custom Linear class initializes itself in __init__ with initializer_range=0.02
        # If you want to use a different initializer_range for Linear layers, you can pass it
        # when creating the Linear instance, or modify the Linear class to accept config.
        
        # Initialize embedding parameters (token and positional embeddings)
        # Embeddings in GPTEmbeddings are stored as nn.Parameter attributes
        # We check if the module has parameters named 'token_embeddings' or 'position_embeddings'
        # and initialize them from a normal distribution
        if hasattr(module, 'token_embeddings') and isinstance(module.token_embeddings, nn.Parameter):
            # Token embeddings: shape (vocab_size, n_embd)
            # Initialize from normal distribution with same range as linear layers
            torch.nn.init.normal_(
                module.token_embeddings,
                mean=0.0,
                std=self.config.initializer_range
            )
        
        if hasattr(module, 'position_embeddings') and isinstance(module.position_embeddings, nn.Parameter):
            # Positional embeddings: shape (n_positions, n_embd)
            # Initialize from normal distribution with same range as linear layers
            torch.nn.init.normal_(
                module.position_embeddings,
                mean=0.0,
                std=self.config.initializer_range
            )
        
        # LayerNorm parameters (scale and bias) are already initialized in LayerNorm.__init__
        # Scale is initialized to ones, bias to zeros, which is the standard initialization.
        # We don't need to re-initialize them here.

