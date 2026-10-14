import pytest
import torch

from morph.decoding.processed_policy import (
    apply_common_warpers,
    apply_repetition_penalty,
    masked_logprobs,
)


def test_legality_mask_is_exact_and_normalized():
    result = masked_logprobs(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False, True]))
    assert torch.isneginf(result[1])
    assert torch.isclose(result[torch.isfinite(result)].exp().sum(), torch.tensor(1.0))


def test_common_top_k_is_applied_after_bridge_distribution():
    result = apply_common_warpers(torch.log_softmax(torch.tensor([3.0, 2.0, 1.0]), dim=-1), top_k=2)
    assert torch.isneginf(result[2])
    assert torch.isclose(result[:2].exp().sum(), torch.tensor(1.0))


def test_top_p_keeps_crossing_token():
    result = apply_common_warpers(torch.log_softmax(torch.tensor([3.0, 2.0, 0.0]), dim=-1), top_p=0.9)
    assert torch.isfinite(result[0])
    assert torch.isfinite(result[1])
    assert torch.isneginf(result[2])


def test_repetition_penalty_matches_transformers_sign_rule():
    logits = torch.tensor([-2.0, 3.0, 1.0])
    result = apply_repetition_penalty(logits, torch.tensor([0, 1, 1]), penalty=1.2)
    assert torch.allclose(result, torch.tensor([-2.4, 2.5, 1.0]))


@pytest.mark.parametrize("kwargs", [{"temperature": 0.0}, {"top_k": 0}, {"top_p": 0.0}])
def test_invalid_warper_parameters_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        apply_common_warpers(torch.tensor([0.0, -1.0]), **kwargs)
