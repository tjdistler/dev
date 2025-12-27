"""
Unit tests for data loading and preprocessing.
"""

import pytest
import torch

from data import GPTDataset, create_dataloader
from gpt.tokenizer import GPTTokenizer


class TestGPTDataset:
    """Test cases for GPTDataset class."""
    
    def test_dataset_initialization(self):
        """Test GPTDataset initialization."""
        # TODO: Create mock tokenizer
        # TODO: Create list of texts
        # TODO: Create GPTDataset
        # TODO: Assert dataset is created successfully
        pass
    
    def test_dataset_length(self):
        """Test dataset length."""
        # TODO: Create GPTDataset with known number of texts
        # TODO: Assert len(dataset) is correct
        pass
    
    def test_dataset_getitem_shape(self):
        """Test that __getitem__ returns correct shapes."""
        # TODO: Create GPTDataset
        # TODO: Get item from dataset
        # TODO: Assert 'input_ids' and 'labels' have correct shapes
        pass
    
    def test_dataset_input_ids_and_labels(self):
        """Test that labels are shifted by 1 from input_ids."""
        # TODO: Create GPTDataset
        # TODO: Get item from dataset
        # TODO: Assert labels[i] == input_ids[i+1] (for valid indices)
        pass
    
    def test_dataset_max_length_truncation(self):
        """Test that sequences are truncated to max_length."""
        # TODO: Create GPTDataset with max_length
        # TODO: Get item from dataset
        # TODO: Assert sequence length <= max_length
        pass
    
    def test_dataset_padding(self):
        """Test that sequences are padded if needed."""
        # TODO: Create GPTDataset
        # TODO: Get items with different lengths
        # TODO: Verify padding is applied correctly
        pass


class TestDataLoader:
    """Test cases for create_dataloader function."""
    
    def test_dataloader_creation(self):
        """Test that DataLoader is created successfully."""
        # TODO: Create mock tokenizer
        # TODO: Create list of texts
        # TODO: Create DataLoader
        # TODO: Assert DataLoader is created
        pass
    
    def test_dataloader_batch_size(self):
        """Test that batches have correct size."""
        # TODO: Create DataLoader with batch_size
        # TODO: Iterate through batches
        # TODO: Assert batch size is correct (except possibly last batch)
        pass
    
    def test_dataloader_shuffle(self):
        """Test that shuffling works."""
        # TODO: Create DataLoader with shuffle=True
        # TODO: Get two epochs of data
        # TODO: Assert order differs (with high probability)
        pass
    
    def test_dataloader_no_shuffle(self):
        """Test that no shuffling maintains order."""
        # TODO: Create DataLoader with shuffle=False
        # TODO: Get two epochs of data
        # TODO: Assert order is the same
        pass

