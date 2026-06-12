"""Text normalization and optional phonemization utilities."""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dp.phonemizer import Phonemizer
from unidecode import unidecode

from .numbers import normalize_numbers
from .symbols import phonemes_set

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
    # Power & Electrical
    (r"(?<=\d)\s*W\b", " watts"),
    (r"(?<=\d)\s*kW\b", " kilowatts"),
    (r"(?<=\d)\s*V\b", " volts"),
    (r"(?<=\d)\s*A\b", " amps"),
    (r"(?<=\d)\s*Hz\b", " hertz"),
    # Pressure
    (r"(?<=\d)\s*hPa\b", " hectopascals"),
    (r"(?<=\d)\s*bar\b", " bar"),
    (r"(?<=\d)\s*psi\b", " pounds per square inch"),
    # Volume & Flow
    (r"(?<=\d)\s*L\b", " liters"),
    (r"(?<=\d)\s*mL\b", " milliliters"),
    (r"(?<=\d)\s*m3\b", " cubic meters"),
    # Environmental
    (r"(?<=\d)\s*ppm\b", " parts per million"),
    (r"(?<=\d)\s*lux\b", " lux"),
    (r"(?<=\d)\s*dB\b", " decibels"),
    # Time
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
    """Collapse consecutive whitespace into single spaces."""
    return " ".join(text.split())


def no_cleaners(text: str) -> str:
    """Return text unchanged."""
    return text


def english_cleaners(text: str) -> str:
    """English text cleaning with detailed debug logs."""

    # Unicode → ASCII

    text = unidecode(text)

    # ENERGY UNITS: catch them while digits still exist
    # 4.485 kWh → 4.485 kilowatt hours

    text = re.sub(
        r"([0-9]+(?:\.[0-9]+)?)\s*kWh\b",
        r"\1 kilowatt hours",
        text,
        flags=re.IGNORECASE,
    )
    # 4.485 Wh → 4.485 watt hours

    text = re.sub(
        r"([0-9]+(?:\.[0-9]+)?)\s*Wh\b",
        r"\1 watt hours",
        text,
        flags=re.IGNORECASE,
    )

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


# Phonemizer checkpoint filenames.
ENGLISH_CHECKPOINT = "en_us_cmudict_ipa_forward.pt"
# DeepPhonemizer's multilingual "Latin IPA" model (en_us, en_uk, de, fr, es).
# Used only for non-English requests so the English path stays byte-identical.
MULTILINGUAL_CHECKPOINT = "latin_ipa_forward.pt"

# Languages routed through the English (cmudict) phonemizer + english_cleaners.
_ENGLISH_LANGS = {"en", "en_us"}


class Cleaner:
    """Configurable text cleaner/phonemizer.

    English requests use the original cmudict phonemizer and english_cleaners
    so existing GLaDOS output is unchanged. Other languages are routed through
    the multilingual "Latin IPA" checkpoint (loaded lazily on first use) with
    no_cleaners, so accented characters survive to the phonemizer.
    """

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
        self.models_dir = models_dir

        self.phonemizer = None
        self._multilingual_phonemizer = None
        if self.use_phonemes:
            ckpt = models_dir / ENGLISH_CHECKPOINT
            if not ckpt.is_file():
                raise FileNotFoundError(f"Phonemizer checkpoint not found at {ckpt}")
            self.phonemizer = Phonemizer.from_checkpoint(str(ckpt), device=self.device)

    def _get_multilingual_phonemizer(self) -> Phonemizer:
        """Lazily load the multilingual phonemizer on first non-English use."""
        if self._multilingual_phonemizer is None:
            ckpt = self.models_dir / MULTILINGUAL_CHECKPOINT
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"Multilingual phonemizer checkpoint not found at {ckpt}. "
                    "Run download.py to fetch it, or request an English voice."
                )
            _LOGGER.info("Loading multilingual phonemizer from %s", ckpt)
            self._multilingual_phonemizer = Phonemizer.from_checkpoint(
                str(ckpt), device=self.device
            )
        return self._multilingual_phonemizer

    def __call__(self, text: str, lang: str | None = None) -> str:
        """Clean text and optionally phonemize it.

        ``lang`` overrides the instance default for this call (the multilingual
        phonemizer selects its language per call). English keeps the original
        cleaner + cmudict phonemizer; other languages skip english_cleaners so
        accented characters are preserved for the phonemizer.
        """
        effective_lang = (lang or self.lang).replace("-", "_")
        is_english = effective_lang in _ENGLISH_LANGS

        clean_func = self.clean_func if is_english else no_cleaners
        cleaned = clean_func(text)

        if self.use_phonemes:
            if is_english:
                phon = self.phonemizer(cleaned, lang="en_us")
            else:
                phon = self._get_multilingual_phonemizer()(
                    cleaned, lang=effective_lang
                )
            cleaned = "".join(p for p in phon if p in phonemes_set)
        return cleaned

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        models_dir: Path,
        device: str = "cpu",
    ) -> "Cleaner":
        """Build a Cleaner instance from preprocessing configuration."""
        cfg = config["preprocessing"]
        return cls(
            cleaner_name=cfg["cleaner_name"],
            use_phonemes=cfg["use_phonemes"],
            lang=cfg["language"],
            models_dir=models_dir,
            device=device,
        )
