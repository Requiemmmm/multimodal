import json
from pathlib import Path

from morph.adapters.anole import load_anole_token_sets
from morph.decoding.anole_segments import analyze_anole_sequence


def tokenizer(tmp_path: Path):
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<eoss>": 8196, "<racm3:break>": 8197, "<reserved08706>": 8710, "hello": 10000}
    vocab.update({f"IMGIMG{index}Z": index + 4 for index in range(4)})
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": vocab}}))
    return load_anole_token_sets(path)


def test_complete_sequence_separates_sampled_and_forced_tokens(tmp_path: Path):
    tokens = tokenizer(tmp_path)
    result = analyze_anole_sequence([10000, tokens.boi_id, 4, 5, 6, 7, tokens.eoi_id, 2], tokens, image_length=4)
    assert result.valid
    assert result.route_mask == (False, True, False, False, False, False, False, False)
    assert result.visual_mask == (False, False, True, True, True, True, False, False)
    assert result.forced_mask == (False, False, False, False, False, False, True, False)
    assert result.completed_images == 1


def test_rejects_non_visual_token_inside_image(tmp_path: Path):
    tokens = tokenizer(tmp_path)
    result = analyze_anole_sequence([tokens.boi_id, 4, 10000], tokens, image_length=4)
    assert not result.valid
    assert "illegal visual" in result.error


def test_rejects_missing_forced_eoi(tmp_path: Path):
    tokens = tokenizer(tmp_path)
    result = analyze_anole_sequence([tokens.boi_id, 4, 5, 6, 7], tokens, image_length=4)
    assert not result.valid
    assert "missing forced EOI" in result.error
