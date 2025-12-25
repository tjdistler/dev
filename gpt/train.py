"""
Training script for GPT-1 model.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from gpt.model import GPT
from gpt.config import GPTConfig
from gpt.data import create_dataloader
from tokenizer import GPTTokenizer


def train(model, dataloader, optimizer, device, num_epochs=1):
    """
    Train the GPT model.
    
    Args:
        model: GPT model instance
        dataloader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to train on ('cpu' or 'cuda')
        num_epochs: Number of training epochs
    """
    model.train()
    # TODO: Move model to device
    # TODO: Training loop over epochs
    # TODO: For each batch:
    #   - Move batch to device
    #   - Forward pass
    #   - Compute loss
    #   - Backward pass
    #   - Update weights
    #   - Log progress
    pass


def main():
    """Main training function."""
    # TODO: Set device (cuda if available, else cpu)
    # TODO: Load configuration
    # TODO: Initialize tokenizer
    # TODO: Load training data
    # TODO: Create data loader
    # TODO: Initialize model
    # TODO: Initialize optimizer
    # TODO: Train model
    # TODO: Save model checkpoint
    pass


if __name__ == "__main__":
    main()

