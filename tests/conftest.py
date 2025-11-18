"""Pytest configuration and shared fixtures."""

import pytest
import torch
from pathlib import Path
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer


# Set up safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals(
    [Preprocessor, LanguageTokenizer, SequenceTokenizer]
)


@pytest.fixture
def models_dir():
    """Return the models directory path."""
    return Path("models")


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_texts():
    """Return sample texts for testing."""
    return [
        "Hello world.",
        "This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing GLaDOS voice generation.",
    ]
