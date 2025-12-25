"""
Shared pytest fixtures for testing.
"""

import pytest
import torch

from gpt.config import GPTConfig


@pytest.fixture
def config():
    """Create a default GPTConfig for testing."""
    return GPTConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=4
    )


@pytest.fixture
def device():
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 32

