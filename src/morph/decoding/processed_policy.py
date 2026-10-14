"""Shared legality and sampling operations for MORPH processed policies."""
from __future__ import annotations

import torch


def apply_repetition_penalty(
    raw_logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply the model's history-dependent repetition penalty to one row.

    Anole's official text decoder applies this operation before temperature
    and top-p.  Teacher and student must each receive the penalty using their
    own prompt history, while top-p remains a single post-bridge operation.
    """
    if raw_logits.ndim != 1 or input_ids.ndim != 1:
        raise ValueError("raw_logits and input_ids must be rank-1 vectors")
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0:
        return raw_logits
    out = raw_logits.clone()
    ids = input_ids.to(device=out.device, dtype=torch.long)
    seen = torch.unique(ids)
    scores = out[seen]
    out[seen] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return out


def masked_logprobs(raw_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Apply a legality mask once, then normalize over the remaining support."""
    if raw_logits.ndim != 1 or legal_mask.ndim != 1 or raw_logits.shape != legal_mask.shape:
        raise ValueError("raw_logits and legal_mask must be equal rank-1 vocab vectors")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must be bool")
    if not legal_mask.any():
        raise ValueError("legality mask excludes the whole vocabulary")
    return torch.log_softmax(raw_logits.masked_fill(~legal_mask, float("-inf")), dim=-1)


def apply_common_warpers(
    logprobs: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply post-bridge temperature/top-k/top-p and return normalized log-probs.

    This accepts a single already-bridged distribution.  Calling it separately
    on teacher/student before ``geometric_bridge`` changes the bridge support
    and is intentionally not part of this API.
    """
    if logprobs.ndim != 1:
        raise ValueError("logprobs must be a rank-1 vocab vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    finite_mask = torch.isfinite(logprobs)
    finite_count = int(finite_mask.sum())
    if finite_count == 0:
        raise ValueError("logprobs has empty finite support")

    warped = logprobs / temperature
    if top_k is not None and top_k < finite_count:
        keep = torch.zeros_like(finite_mask)
        keep[torch.topk(warped, k=top_k).indices] = True
        warped = warped.masked_fill(~keep, float("-inf"))
    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(warped, descending=True)
        sorted_probs = torch.softmax(sorted_values, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Preserve the first token crossing top_p and at least the best token.
        remove_sorted = (cumulative - sorted_probs) >= top_p
        remove_sorted[0] = False
        remove = torch.zeros_like(finite_mask)
        remove.scatter_(0, sorted_indices, remove_sorted)
        warped = warped.masked_fill(remove, float("-inf"))
    return torch.log_softmax(warped, dim=-1)
