"""Static Anole token/support facts derived from its official decoder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AnoleTokenSets:
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int
    boi_id: int
    eoi_id: int
    eot_id: int
    text_token_ids: frozenset[int]
    image_token_ids: frozenset[int]

    @property
    def text_step_support(self) -> frozenset[int]:
        """Official TextDecoder support: text, EOS, and native BOI."""
        return self.text_token_ids | {self.eos_id, self.boi_id}

    @property
    def image_step_support(self) -> frozenset[int]:
        """Official ImageDecoder support before its forced terminal EOI."""
        return self.image_token_ids

    def state_for_prefix(self, token_ids: list[int]) -> str:
        """Track the public decoder's text/image state for a synchronized prefix."""
        if not token_ids:
            return "text"
        if token_ids[-1] == self.boi_id:
            return "image"
        # The public decoder forces EOI after exactly 1024 visual IDs.  A
        # caller that has a full prefix observes EOI then resumes text.
        if token_ids[-1] == self.eoi_id:
            return "text"
        return "image" if self.boi_id in token_ids and self.eoi_id not in token_ids[token_ids.index(self.boi_id) + 1 :] else "text"


class TextEncoder(Protocol):
    def encode(self, text: str): ...


def build_anole_user_turn(prompt: str, suffix: str, tokenizer: TextEncoder, tokens: AnoleTokenSets) -> list[int]:
    """Match Anole's public UI template with a user suffix before EOT.

    The public ``TokenManager.tokens_from_ui`` prepends BOS, tokenizes text,
    and appends the explicit ``<END-OF-TURN>`` sentinel.  The suffix is part
    of the ordinary text payload and never writes BOI/EOT or image IDs.
    """
    if not isinstance(prompt, str) or not isinstance(suffix, str):
        raise TypeError("prompt and suffix must be strings")
    encoded = tokenizer.encode(prompt + suffix).ids
    forbidden = ({tokens.boi_id, tokens.eoi_id, tokens.eot_id} | tokens.image_token_ids) & set(encoded)
    if forbidden:
        raise ValueError(f"ordinary-text prompt tokenized to control/image IDs: {sorted(forbidden)}")
    return [tokens.bos_id, *encoded, tokens.eot_id]


def load_anole_token_sets(tokenizer_path: str | Path) -> AnoleTokenSets:
    """Load vocabulary facts without importing Anole's CUDA runtime."""
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Anole tokenizer not found: {path}")
    vocabulary = json.loads(path.read_text())["model"]["vocab"]
    required = ("<s>", "</s>", "<pad>", "<racm3:break>", "<eoss>", "<reserved08706>")
    missing = [token for token in required if token not in vocabulary]
    if missing:
        raise ValueError(f"Anole tokenizer is missing required tokens: {missing}")
    image_tokens = frozenset(value for name, value in vocabulary.items() if name.startswith("IMGIMG"))
    special_tokens = frozenset(value for name, value in vocabulary.items() if name.startswith("<") and name != "<")
    all_tokens = frozenset(vocabulary.values())
    if len(all_tokens) != len(vocabulary):
        raise ValueError("Anole tokenizer contains duplicate token IDs")
    return AnoleTokenSets(
        vocab_size=len(all_tokens),
        bos_id=vocabulary["<s>"],
        eos_id=vocabulary["</s>"],
        pad_id=vocabulary["<pad>"],
        boi_id=vocabulary["<racm3:break>"],
        eoi_id=vocabulary["<eoss>"],
        eot_id=vocabulary["<reserved08706>"],
        text_token_ids=all_tokens - image_tokens - special_tokens,
        image_token_ids=image_tokens,
    )
