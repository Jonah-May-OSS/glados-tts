import re
from pathlib import Path
from typing import Any, Dict, Optional

from unidecode import unidecode

from .numbers import normalize_numbers
from .symbols import phonemes_set

from dp.phonemizer import Phonemizer

# Regular expression matching whitespace

_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations

_abbreviations = []
for abbrev, full_form in [
    ("°F", "degrees Fahrenheit"),
    ("°C", "degrees Celsius"),
    ("deg F", "degrees Fahrenheit"),
    ("deg C", "degrees Celsius"),
    ("mrs", "misses"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]:
    # Match abbreviation as a standalone word, optionally followed by a period
    pattern = re.compile(rf"\b{re.escape(abbrev)}\b\.?", re.IGNORECASE)
    _abbreviations.append((pattern, full_form))


def expand_abbreviations(text: str) -> str:
    """
    Expand common abbreviations in the text.

    Args:
        text: The input text containing abbreviations.

    Returns:
        The text with abbreviations expanded.
    """
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    return text


def collapse_whitespace(text: str) -> str:
    """
    Collapse multiple whitespaces into a single space.

    Args:
        text: The input text with potential multiple spaces.

    Returns:
        The text with whitespace collapsed.
    """
    return _whitespace_re.sub(" ", text).strip()


def no_cleaners(text: str) -> str:
    """
    Return the text unchanged.

    Args:
        text: The input text.

    Returns:
        The unchanged text.
    """
    return text


def english_cleaners(text: str) -> str:
    """
    Clean English text by unidecoding, normalizing numbers, and expanding abbreviations.

    Args:
        text: The input text to clean.

    Returns:
        The cleaned text.
    """
    text = unidecode(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


class Cleaner:
    """Text cleaner that optionally phonemizes the text."""

    def __init__(
        self,
        cleaner_name: str,
        use_phonemes: bool,
        lang: str,
        models_dir: Path,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the Cleaner.

        Args:
            cleaner_name: Name of the cleaning function to use ('english_cleaners' or 'no_cleaners').
            use_phonemes: Whether to convert text to phonemes.
            lang: Language code for phonemization (e.g., 'en_us').
            models_dir: Directory containing model files.
            device: Device to load models onto ('cpu' or 'cuda').
        """
        if cleaner_name == "english_cleaners":
            self.clean_func = english_cleaners
        elif cleaner_name == "no_cleaners":
            self.clean_func = no_cleaners
        else:
            raise ValueError(
                f"Cleaner not supported: {cleaner_name}! "
                f"Currently supported: ['english_cleaners', 'no_cleaners']"
            )
        self.use_phonemes = use_phonemes
        self.lang = lang
        self.device = device

        if use_phonemes:
            # Construct the path to the phonemizer checkpoint
            checkpoint_path = models_dir / "en_us_cmudict_ipa_forward.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Phonemizer checkpoint not found at {checkpoint_path}. "
                    "Ensure that the model file exists."
                )
            # Initialize the phonemizer
            self.phonemizer = Phonemizer.from_checkpoint(
                checkpoint_path, device=self.device
            )

    def __call__(self, text: str) -> str:
        """
        Clean and optionally phonemize the text.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned (and optionally phonemized) text.
        """
        text = self.clean_func(text)

        if self.use_phonemes:
            # Phonemize the text using the appropriate method
            phonemized_text = self.phonemizer(text, lang="en_us")
            # Filter out unwanted phonemes
            text = "".join([p for p in phonemized_text if p in phonemes_set])
        text = collapse_whitespace(text)
        return text

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        models_dir: Path,
        device: str = "cpu",
    ) -> "Cleaner":
        """
        Create a Cleaner instance from a configuration dictionary.

        Args:
            config: Configuration dictionary.
            models_dir: Directory containing model files.
            device: Device to load models onto ('cpu' or 'cuda').

        Returns:
            An instance of Cleaner.
        """
        return cls(
            cleaner_name=config["preprocessing"]["cleaner_name"],
            use_phonemes=config["preprocessing"]["use_phonemes"],
            lang=config["preprocessing"]["language"],
            models_dir=models_dir,
            device=device,
        )
