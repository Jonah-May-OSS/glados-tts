import torch
from .symbols import phonemes
from typing import Dict


class Tokenizer:
    """
    Tokenizer for converting text to sequences of token IDs and vice versa.
    Uses torch tensors to improve speed when dealing with large batches.
    """

    # Class variables to avoid rebuilding mappings for each instance

    symbol_to_id: Dict[str, int] = {s: i for i, s in enumerate(phonemes)}
    id_to_symbol: Dict[int, str] = {i: s for i, s in enumerate(phonemes)}

    def __call__(self, text: str) -> torch.Tensor:
        """
        Convert text to a list of token IDs using tensor operations.

        Args:
            text: The input text string to tokenize.

        Returns:
            A tensor of integer token IDs corresponding to the symbols in the text.
            Unknown symbols are ignored.
        """
        # Create a list of token IDs based on the text, and filter any invalid symbols

        valid_tokens = [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

        # Convert the list of token IDs to a tensor in one go

        text_tensor = torch.tensor(valid_tokens, dtype=torch.long)
        return text_tensor

    def decode(self, sequence: torch.Tensor) -> str:
        """
        Convert a sequence of token IDs back to text.

        Args:
            sequence: A tensor of integer token IDs.

        Returns:
            A string reconstructed from the token IDs.
            Unknown IDs are ignored.
        """
        # Decode the tensor back into text by looking up the id_to_symbol mapping

        text = "".join(
            [
                self.id_to_symbol[s.item()]
                for s in sequence
                if s.item() in self.id_to_symbol
            ]
        )
        return text
