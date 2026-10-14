import json
from pathlib import Path

import pytest

from morph.adapters.anole import build_anole_user_turn, load_anole_token_sets


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text):
        return FakeEncoding(self.ids)


def token_sets(tmp_path: Path):
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<eoss>": 8196, "<racm3:break>": 8197, "<reserved08706>": 8710, "normal": 10000, "IMGIMGAZ": 4}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": vocab}}))
    return load_anole_token_sets(path)


def test_suffix_stays_before_eot(tmp_path: Path):
    tokens = token_sets(tmp_path)
    assert build_anole_user_turn("question", " suffix", FakeTokenizer([10000]), tokens) == [0, 10000, 8710]


@pytest.mark.parametrize("forbidden", [8197, 8196, 8710, 4])
def test_control_or_image_ids_are_rejected(tmp_path: Path, forbidden: int):
    with pytest.raises(ValueError, match="control/image"):
        build_anole_user_turn("question", " suffix", FakeTokenizer([forbidden]), token_sets(tmp_path))
