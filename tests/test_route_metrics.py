import torch
import pytest

from morph.scoring.route import route_metrics, topk_overlap


def test_route_metrics_for_finite_boi():
    metrics = route_metrics(torch.tensor([-2.0, -0.2, -1.0]), 2)
    assert metrics.boi_finite and metrics.boi_rank == 2
    assert metrics.boi_margin_to_best_non_boi == pytest.approx(-0.8)


def test_route_metrics_for_illegal_boi():
    metrics = route_metrics(torch.tensor([-2.0, float("-inf")]), 1)
    assert not metrics.boi_finite and metrics.boi_rank is None


def test_topk_overlap():
    assert topk_overlap(torch.tensor([3.0, 2.0, 1.0]), torch.tensor([2.0, 3.0, 1.0]), 2) == 1.0
