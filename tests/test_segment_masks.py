import pytest

from morph.decoding.segments import segment_mask


def test_segment_mask():
    assert segment_mask(4, [0, 2]).tolist() == [True, False, True, False]


def test_segment_mask_bounds():
    with pytest.raises(ValueError, match="outside"):
        segment_mask(2, [2])
