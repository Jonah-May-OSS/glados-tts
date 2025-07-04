import torch
from pathlib import Path
from functools import lru_cache
from .text.cleaners import Cleaner
from .text.tokenizer import Tokenizer
from typing import List

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


def prepare_text(
    text: str,
    device: torch.device,
    cleaner: Cleaner,  # Pass pre-loaded cleaner
    tokenizer: Tokenizer,  # Pass pre-loaded tokenizer
) -> torch.Tensor:
    if not text:
        raise ValueError("Input text cannot be empty.")
    if text[-1] not in ".?!":
        text += "."
    # pull from cache (phonemizer only initialized once)

    cleaned = cleaner(text)
    tokens = tokenizer(cleaned)
    return torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
