from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class StepPolicy:
    """A processed one-step distribution at a decoder boundary."""

    logprobs: torch.Tensor
    raw_logits: torch.Tensor
    finite_mask: torch.Tensor
    decoder_state: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logprobs.ndim != 1 or self.raw_logits.ndim != 1 or self.finite_mask.ndim != 1:
            raise ValueError("StepPolicy tensors must be rank-1 vocab vectors")
        if not (self.logprobs.shape == self.raw_logits.shape == self.finite_mask.shape):
            raise ValueError("StepPolicy tensors must have equal vocabulary shape")
        if self.finite_mask.dtype != torch.bool:
            raise TypeError("finite_mask must be bool")
