"""
Number normalization utilities for text-to-speech preprocessing.

This module provides functions to convert numbers and numerical expressions in text
to their spoken equivalents, which is essential for generating natural-sounding speech.
"""

import re
from typing import Match

import inflect

# Initialize the inflect engine

_inflect = inflect.engine()

# Regular expressions for different types of numerical patterns

_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9.,]*[0-9]+)")
_ordinal_re = re.compile(r"\b([0-9]+)(st|nd|rd|th)\b")
_number_re = re.compile(r"\b[0-9]+\b")


def _remove_commas(m: Match) -> str:
    """Remove commas from numbers."""
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: Match) -> str:
    """Expand decimal numbers by replacing the decimal point with 'point'."""
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: Match) -> str:
    """Expand dollar amounts to spoken words."""
    match = m.group(1)
    parts = match.replace(",", "").split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0

    dollar_unit = "dollar" if dollars == 1 else "dollars"
    cent_unit = "cent" if cents == 1 else "cents"

    if dollars and cents:
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    elif dollars:
        return f"{dollars} {dollar_unit}"
    elif cents:
        return f"{cents} {cent_unit}"
    else:
        return "zero dollars"


def _expand_pounds(m: Match) -> str:
    """Expand pound amounts to spoken words."""
    amount = m.group(1).replace(",", "")
    number = int(amount)
    pound_unit = "pound" if number == 1 else "pounds"
    return f"{number} {pound_unit}"


def _expand_ordinal(m: Match) -> str:
    """Expand ordinal numbers to words."""
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: Match) -> str:
    """Expand cardinal numbers to words."""
    num_str = m.group(0)
    num_int = int(num_str)
    return _inflect.number_to_words(num_int, andword="")


def normalize_numbers(text: str) -> str:
    """
    Normalize numbers in the input text to their spoken equivalents.

    Args:
        text: The input text containing numbers.

    Returns:
        The text with numbers expanded into words.
    """
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, _expand_pounds, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
