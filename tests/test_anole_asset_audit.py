import json
import subprocess
import sys
from pathlib import Path


def test_anole_audit_rejects_missing_assets(tmp_path: Path):
    script = Path(__file__).parents[1] / "scripts" / "00_audit_anole_assets.py"
    result = subprocess.run(
        [sys.executable, str(script), "--model-path", str(tmp_path), "--code-root", str(tmp_path), "--output", str(tmp_path / "audit.json")],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "missing required files" in result.stderr


def test_audit_fixture_is_structured(tmp_path: Path):
    model = tmp_path / "model"
    (model / "models/7b").mkdir(parents=True)
    (model / "tokenizer").mkdir()
    vocabulary = {"<s>": 0, "</s>": 2, "<pad>": 1, "<racm3:break>": 8197, "<eoss>": 8196, "<reserved08706>": 8710}
    for index in range(8192):
        vocabulary[f"IMGIMG{index}Z"] = index + 4
    (model / "tokenizer/text_tokenizer.json").write_text(json.dumps({"model": {"vocab": vocabulary}}))
    (model / "tokenizer/vqgan.yaml").write_text("resolution: 512\n")
    for name in ("models/7b/consolidated.pth", "models/7b/consolidate_params.json", "tokenizer/vqgan.ckpt"):
        (model / name).write_text("{}")
    (model / "models/7b/params.json").write_text(json.dumps({"dtype": "bf16", "model": {"dim": 1, "n_layers": 1, "n_heads": 1, "vocab_size": 1, "max_length": 1}}))
    script = Path(__file__).parents[1] / "scripts" / "00_audit_anole_assets.py"
    output = tmp_path / "audit.json"
    result = subprocess.run([sys.executable, str(script), "--model-path", str(model), "--code-root", str(tmp_path), "--output", str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["image_vocabulary"]["complete_zero_to_8191"]
