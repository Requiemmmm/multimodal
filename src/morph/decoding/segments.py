from collections.abc import Iterable

import torch


def segment_mask(length: int, indices: Iterable[int], *, device=None) -> torch.Tensor:
    """Create a boolean segment mask, rejecting out-of-range external indices."""
    if length < 0:
        raise ValueError("length must be non-negative")
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for index in indices:
        if not 0 <= index < length:
            raise ValueError(f"segment index {index} outside length {length}")
        mask[index] = True
    return mask
