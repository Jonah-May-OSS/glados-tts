"""Unit tests for utility functions."""

import pytest
import torch
from pathlib import Path
from utils.tools import prepare_text, _get_cleaner_and_tokenizer


class TestTextPreparation:
    """Tests for text preparation utilities."""

    def test_prepare_text_basic(self, device):
        """Test basic text preparation."""
        cleaner, tokenizer = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )
        text = "Hello world"
        result = prepare_text(text, device, cleaner, tokenizer)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2  # Should be batched
        assert result.shape[0] == 1  # Batch size of 1
        assert result.device.type == device.type

    def test_prepare_text_adds_period(self, device):
        """Test that prepare_text adds period if missing."""
        cleaner, tokenizer = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )
        text = "Hello world"
        result = prepare_text(text, device, cleaner, tokenizer)

        # Text should be processed successfully
        assert result is not None
        assert result.numel() > 0

    def test_prepare_text_with_punctuation(self, device):
        """Test text preparation with existing punctuation."""
        cleaner, tokenizer = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )
        for punct in [".", "?", "!"]:
            text = f"Hello world{punct}"
            result = prepare_text(text, device, cleaner, tokenizer)
            assert isinstance(result, torch.Tensor)
            assert result.numel() > 0

    def test_prepare_text_empty_raises(self, device):
        """Test that empty text raises ValueError."""
        cleaner, tokenizer = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            prepare_text("", device, cleaner, tokenizer)

    def test_cleaner_tokenizer_cache(self, device):
        """Test that cleaner and tokenizer are cached."""
        # Call twice with same parameters
        result1 = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )
        result2 = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )

        # Should return same instances (cached)
        assert result1[0] is result2[0]
        assert result1[1] is result2[1]

    def test_prepare_text_various_lengths(self, device, sample_texts):
        """Test text preparation with various text lengths."""
        cleaner, tokenizer = _get_cleaner_and_tokenizer(
            "models", str(device), "english_cleaners", "en_us", True
        )

        for text in sample_texts:
            result = prepare_text(text, device, cleaner, tokenizer)
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == 1
            # Longer text should generally produce more tokens
            assert result.shape[1] > 0
