"""Route diagnostics computed from one processed next-token policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RouteMetrics:
    boi_logprob: float
    boi_probability: float
    boi_rank: int | None
    boi_margin_to_best_non_boi: float
    boi_finite: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def route_metrics(logprobs: torch.Tensor, boi_id: int) -> RouteMetrics:
    """Report BOI reachability without forcing or biasing the token."""
    if logprobs.ndim != 1:
        raise ValueError("logprobs must be rank-1")
    if not 0 <= boi_id < logprobs.numel():
        raise ValueError("boi_id outside vocabulary")
    boi_logprob = logprobs[boi_id]
    finite = bool(torch.isfinite(boi_logprob))
    if not finite:
        return RouteMetrics(float("-inf"), 0.0, None, float("-inf"), False)
    rank = int((logprobs > boi_logprob).sum().item()) + 1
    comparison = logprobs.clone()
    comparison[boi_id] = float("-inf")
    return RouteMetrics(
        boi_logprob=float(boi_logprob),
        boi_probability=float(boi_logprob.exp()),
        boi_rank=rank,
        boi_margin_to_best_non_boi=float(boi_logprob - comparison.max()),
        boi_finite=True,
    )


def topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    """Fractional overlap among the top-k finite IDs of two policies."""
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("policies must be equal rank-1 vocab vectors")
    if k <= 0:
        raise ValueError("k must be positive")
    available = min(int(torch.isfinite(left).sum()), int(torch.isfinite(right).sum()))
    if available == 0:
        return 0.0
    k = min(k, available)
    left_ids = set(torch.topk(left, k).indices.tolist())
    right_ids = set(torch.topk(right, k).indices.tolist())
    return len(left_ids & right_ids) / k
