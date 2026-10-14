import json
from pathlib import Path

from morph.adapters.anole import load_anole_token_sets


def write_tokenizer(path: Path) -> None:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<eoss>": 8196, "<racm3:break>": 8197, "<reserved08706>": 8710, "hello": 10000}
    vocab.update({f"IMGIMG{index}Z": index + 4 for index in range(4)})
    path.write_text(json.dumps({"model": {"vocab": vocab}}))


def test_official_anole_support_has_native_boi(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer.json"
    write_tokenizer(tokenizer)
    tokens = load_anole_token_sets(tokenizer)
    assert tokens.boi_id in tokens.text_step_support
    assert tokens.image_step_support == frozenset({4, 5, 6, 7})
    assert tokens.eoi_id not in tokens.image_step_support
    assert not (tokens.text_step_support & tokens.image_step_support)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer.json"
    write_tokenizer(tokenizer)
    payload = json.loads(tokenizer.read_text())
    payload["model"]["vocab"]["duplicate"] = 10000
    tokenizer.write_text(json.dumps(payload))
    try:
        load_anole_token_sets(tokenizer)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("expected duplicate-ID validation failure")
