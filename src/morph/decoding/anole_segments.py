"""Structural parsing for the public Anole decoder contract."""
from __future__ import annotations

from dataclasses import dataclass

from morph.adapters.anole import AnoleTokenSets


@dataclass(frozen=True)
class AnoleSequenceSegments:
    route_mask: tuple[bool, ...]
    visual_mask: tuple[bool, ...]
    forced_mask: tuple[bool, ...]
    valid: bool
    error: str | None
    completed_images: int

    @property
    def structure_mask(self) -> tuple[bool, ...]:
        return tuple(route or forced for route, forced in zip(self.route_mask, self.forced_mask))


def analyze_anole_sequence(token_ids: list[int], tokens: AnoleTokenSets, *, image_length: int = 1024) -> AnoleSequenceSegments:
    """Validate generated IDs under Anole's public Text/ImageDecoder state machine.

    BOI is a model-sampled text-state token.  The following 1024 visual IDs
    are model-sampled under the image-only support, while EOI is appended by
    the public decoder.  Keeping the final category separate prevents a
    forced EOI from entering a native route or NLL metric.
    """
    if image_length <= 0:
        raise ValueError("image_length must be positive")
    route_mask = [False] * len(token_ids)
    visual_mask = [False] * len(token_ids)
    forced_mask = [False] * len(token_ids)
    state = "text"
    visual_count = 0
    completed_images = 0
    error: str | None = None
    for index, token_id in enumerate(token_ids):
        if state == "text":
            if token_id == tokens.boi_id:
                route_mask[index] = True
                state = "image"
                visual_count = 0
            elif token_id not in tokens.text_step_support:
                error = f"token {token_id} is illegal in text state at index {index}"
                break
        elif visual_count < image_length:
            if token_id not in tokens.image_step_support:
                error = f"token {token_id} is illegal visual token {visual_count} at index {index}"
                break
            visual_mask[index] = True
            visual_count += 1
        else:
            if token_id != tokens.eoi_id:
                error = f"expected forced EOI {tokens.eoi_id} after {image_length} visual tokens at index {index}"
                break
            forced_mask[index] = True
            completed_images += 1
            state = "text"
    if error is None and state == "image":
        if visual_count < image_length:
            error = f"truncated image: {visual_count}/{image_length} visual tokens"
        else:
            error = "missing forced EOI after complete visual sequence"
    return AnoleSequenceSegments(
        route_mask=tuple(route_mask),
        visual_mask=tuple(visual_mask),
        forced_mask=tuple(forced_mask),
        valid=error is None,
        error=error,
        completed_images=completed_images,
    )
