"""
Defines the set of symbols (phonemes and special characters) used as input to the TTS model.

The default set includes English phonemes and symbols that are compatible with the model.
You can modify this set to include other characters or phonemes as needed.
"""

# Special symbols

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"

# Phonemes (International Phonetic Alphabet symbols)

_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧ"
_diacritics = "ɚ˞ɫ"

# Extra phonemes from IPA annotations

_extra_phonemes = ["g", "ɝ", "̃", "̍", "̥", "̩", "̯", "͡"]

# Combine all symbols into a list

phonemes = (
    list(
        _pad
        + _punctuation
        + _special
        + _vowels
        + _non_pulmonic_consonants
        + _pulmonic_consonants
        + _suprasegmentals
        + _other_symbols
        + _diacritics
    )
    + _extra_phonemes
)

# Create a set for faster lookup (e.g., in 'in' checks)

phonemes_set = set(phonemes)

# Indices of silent phonemes (useful for masking or special handling)

silent_phonemes_indices = [
    i for i, p in enumerate(phonemes) if p in (_pad + _punctuation)
]
