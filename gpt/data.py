"""
Data loading and preprocessing for GPT-1 training.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    Dataset for GPT-1 language modeling.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(self, texts, tokenizer, max_length=512):
        # TODO: Store texts, tokenizer, and max_length
        # TODO: Tokenize all texts
        # TODO: Create training examples (input sequences and targets)
        pass
    
    def __len__(self):
        """Return number of examples in dataset."""
        # TODO: Return dataset size
        pass
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        # TODO: Get tokenized sequence at index idx
        # TODO: Truncate or pad to max_length
        # TODO: Create input_ids (all tokens except last)
        # TODO: Create labels (all tokens except first, shifted by 1)
        # TODO: Return as dictionary
        pass


def create_dataloader(texts, tokenizer, batch_size=32, max_length=512, shuffle=True):
    """
    Create a DataLoader for training.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    # TODO: Create GPTDataset
    # TODO: Create and return DataLoader
    pass

