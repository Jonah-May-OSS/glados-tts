from typing import List, Dict

from .symbols import phonemes


class Tokenizer:
    """
    Tokenizer for converting text to sequences of token IDs and vice versa.

    The tokenizer uses a predefined list of phonemes to map symbols to IDs and back.
    """

    # Class variables to avoid rebuilding mappings for each instance

    symbol_to_id: Dict[str, int] = {s: i for i, s in enumerate(phonemes)}
    id_to_symbol: Dict[int, str] = {i: s for i, s in enumerate(phonemes)}

    def __call__(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        Args:
            text: The input text string to tokenize.

        Returns:
            A list of integer token IDs corresponding to the symbols in the text.
            Unknown symbols are ignored.
        """
        return [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

    def decode(self, sequence: List[int]) -> str:
        """
        Convert a sequence of token IDs back to text.

        Args:
            sequence: A list of integer token IDs.

        Returns:
            A string reconstructed from the token IDs.
            Unknown IDs are ignored.
        """
        text = [self.id_to_symbol[s] for s in sequence if s in self.id_to_symbol]
        return "".join(text)
