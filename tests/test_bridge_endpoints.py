import pytest
import torch

from morph.decoding.bridge import geometric_bridge


def test_bridge_endpoints():
    q = torch.tensor([0.0, float("-inf"), -1.0])
    p = torch.tensor([-1.0, float("-inf"), 0.0])
    assert torch.allclose(geometric_bridge(q, p, 1.0), q)
    assert torch.allclose(geometric_bridge(q, p, 0.0), p)


def test_bridge_rejects_support_mismatch():
    q = torch.tensor([0.0, float("-inf")])
    p = torch.tensor([0.0, 0.0])
    with pytest.raises(ValueError, match="support"):
        geometric_bridge(q, p, 0.5)
