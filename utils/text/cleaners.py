import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from unidecode import unidecode

from .numbers import normalize_numbers
from .symbols import phonemes_set

from dp.phonemizer import Phonemizer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Abbreviation patterns (specific → generic)
# -----------------------------------------------------------------------------


_abbreviations: List[Tuple[re.Pattern, str]] = []

for pattern, repl in [
    # Textual degree forms (symbolic handled earlier)
    (r"\bdeg\s*F\b", "degrees Fahrenheit"),
    (r"\bdeg\s*C\b", "degrees Celsius"),
    (r"\bdeg\b", "degrees"),
    # Measurement units – only when they directly follow a number
    (r"(?<=\d)\s*ft\b\.?", " feet"),
    (r"(?<=\d)\s*in\b\.?", " inches"),
    (r"(?<=\d)\s*mi\b\.?", " miles"),
    (r"(?<=\d)\s*km\b\.?", " kilometers"),
    (r"(?<=\d)\s*mm\b\.?", " millimeters"),
    (r"(?<=\d)\s*cm\b\.?", " centimeters"),
    (r"(?<=\d)\s*m\b\.?", " meters"),
    (r"(?<=\d)\s*kg\b\.?", " kilograms"),
    (r"(?<=\d)\s*g\b\.?", " grams"),
    (r"(?<=\d)\s*oz\b\.?", " ounces"),
    (r"(?<=\d)\s*lb\b\.?", " pounds"),
    (r"(?<=\d)\s*hr\b\.?", " hours"),
    (r"(?<=\d)\s*min\b\.?", " minutes"),
    (r"(?<=\d)\s*sec\b\.?", " seconds"),
    # Honorifics and titles
    (r"\bmrs?\.?(?=\b)", "misses"),
    (r"\bmr\.?(?=\b)", "mister"),
    (r"\bdr\.?(?=\b)", "doctor"),
    (r"\bst\.?(?=\b)", "saint"),
    (r"\bco\.?(?=\b)", "company"),
    (r"\bjr\.?(?=\b)", "junior"),
    (r"\bmaj\.?(?=\b)", "major"),
    (r"\bgen\.?(?=\b)", "general"),
    (r"\bdrs\.?(?=\b)", "doctors"),
    (r"\brev\.?(?=\b)", "reverend"),
    (r"\blt\.?(?=\b)", "lieutenant"),
    (r"\bhon\.?(?=\b)", "honorable"),
    (r"\bsgt\.?(?=\b)", "sergeant"),
    (r"\bcapt\.?(?=\b)", "captain"),
    (r"\besq\.?(?=\b)", "esquire"),
    (r"\bltd\.?(?=\b)", "limited"),
    (r"\bcol\.?(?=\b)", "colonel"),
    (r"\bglados\b", "glah dos"),
    (r"\bAI\b", "A I"),
    (r"\bAI's\b", "A I's"),
]:
    _abbreviations.append((re.compile(pattern, re.IGNORECASE), repl))
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def expand_abbreviations(text: str) -> str:
    """Apply abbreviation substitutions using precompiled regex patterns."""
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    return text


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def no_cleaners(text: str) -> str:
    return text


def english_cleaners(text: str) -> str:
    """English text cleaning with detailed debug logs."""

    # Unicode → ASCII

    text = unidecode(text)

    # Numeric temperature patterns (handles °, deg, uppercase/lowercase)

    temp_pattern = r"(\d+(?:\.\d+)?)\s*(?:°|deg)\s*([FfCc])"

    def _temp_sub(match: re.Match) -> str:
        num, unit = match.group(1), match.group(2).lower()
        return f"{num} degrees {'fahrenheit' if unit == 'f' else 'celsius'}"

    text = re.sub(temp_pattern, _temp_sub, text)

    # Convert bare numbers → words

    text = normalize_numbers(text)

    # Expand remaining abbreviations

    text = expand_abbreviations(text)

    # Collapse whitespace

    text = collapse_whitespace(text)

    return text


# -----------------------------------------------------------------------------
# Cleaner class
# -----------------------------------------------------------------------------


class Cleaner:
    """Configurable text cleaner/phonemizer."""

    def __init__(
        self,
        cleaner_name: str,
        use_phonemes: bool,
        lang: str,
        models_dir: Path,
        device: str = "cpu",
    ) -> None:
        self.lang = lang.replace("-", "_")
        self.clean_func = (
            english_cleaners if cleaner_name == "english_cleaners" else no_cleaners
        )
        self.use_phonemes = use_phonemes
        self.device = device

        if self.use_phonemes:
            ckpt = models_dir / "en_us_cmudict_ipa_forward.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(f"Phonemizer checkpoint not found at {ckpt}")
            self.phonemizer = Phonemizer.from_checkpoint(ckpt, device=self.device)

    def __call__(self, text: str) -> str:
        cleaned = self.clean_func(text)
        if self.use_phonemes:
            phon = self.phonemizer(cleaned, lang=self.lang)
            cleaned = "".join(p for p in phon if p in phonemes_set)
        return cleaned

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        models_dir: Path,
        device: str = "cpu",
    ) -> "Cleaner":
        cfg = config["preprocessing"]
        return cls(
            cleaner_name=cfg["cleaner_name"],
            use_phonemes=cfg["use_phonemes"],
            lang=cfg["language"],
            models_dir=models_dir,
            device=device,
        )
