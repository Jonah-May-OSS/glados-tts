"""Text preprocessing helpers shared by GLaDOS TTS modules."""

from functools import lru_cache
from pathlib import Path

import torch

from .text.cleaners import Cleaner
from .text.tokenizer import Tokenizer

# 1) cache Cleaner+Tokenizer singletons


@lru_cache(maxsize=1)
def _get_cleaner_and_tokenizer(
    models_dir: str, device: str, cleaner_name: str, lang: str, use_phonemes: bool
):
    c = Cleaner(
        cleaner_name=cleaner_name,
        use_phonemes=use_phonemes,
        lang=lang,
        models_dir=Path(models_dir),
        device=device,
    )
    t = Tokenizer()
    return c, t


def get_cleaner_and_tokenizer(
    models_dir: str, device: str, cleaner_name: str, lang: str, use_phonemes: bool
):
    """Public wrapper for cached cleaner/tokenizer creation."""
    return _get_cleaner_and_tokenizer(
        models_dir, device, cleaner_name, lang, use_phonemes
    )


def prepare_text(
    text: str,
    device: torch.device,
    cleaner: Cleaner,  # Pass pre-loaded cleaner
    tokenizer: Tokenizer,  # Pass pre-loaded tokenizer
) -> torch.Tensor:
    """Normalize text and return a single-batch tensor of token IDs."""
    if not text:
        raise ValueError("Input text cannot be empty.")
    if text[-1] not in ".?!":
        text += "."
    # pull from cache (phonemizer only initialized once)

    cleaned = cleaner(text)
    tokens = tokenizer(cleaned)
    return torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
