#!/usr/bin/env python
"""CPU-only provenance, vocabulary, and generation-stack audit for Anole."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REQUIRED_FILES = (
    "models/7b/consolidated.pth",
    "models/7b/params.json",
    "models/7b/consolidate_params.json",
    "tokenizer/text_tokenizer.json",
    "tokenizer/vqgan.yaml",
    "tokenizer/vqgan.ckpt",
)
SPECIAL_TOKENS = (
    "<s>",
    "</s>",
    "<pad>",
    "<racm3:break>",
    "<eoss>",
    "<reserved08706>",
)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def decode_image_token(name: str) -> int:
    digit_map = {chr(ord("A") + index): str(index) for index in range(10)}
    return int("".join(digit_map.get(char, char) for char in name[len("IMGIMG") : -1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model_path, code_root, output = map(Path, (args.model_path, args.code_root, args.output))
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise SystemExit(f"Anole checkpoint missing required files: {', '.join(missing)}")

    vocabulary = json.loads((model_path / "tokenizer/text_tokenizer.json").read_text())["model"]["vocab"]
    image_entries = [(name, token_id) for name, token_id in vocabulary.items() if name.startswith("IMGIMG")]
    image_ids = {token_id for _, token_id in image_entries}
    codebook_ids = {decode_image_token(name) for name, _ in image_entries}
    vq_config = (model_path / "tokenizer/vqgan.yaml").read_text()
    params = json.loads((model_path / "models/7b/params.json").read_text())
    model_config = params["model"]
    expected_codebook_size = 8192
    if len(image_entries) != expected_codebook_size or codebook_ids != set(range(expected_codebook_size)):
        raise RuntimeError("Anole image vocabulary is not a complete 8192-entry VQ codebook mapping")

    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "asset_token_audit",
        "hub_id": "GAIR/Anole-7b-v0.1",
        "revision": "b4d4d4d06d02d0b0e95ac321551b79739579130c",
        "model_path": str(model_path),
        "code_root": str(code_root),
        "code_commit": git_commit(code_root),
        "required_files": {name: (model_path / name).stat().st_size for name in REQUIRED_FILES},
        "model": {
            "dtype": params["dtype"],
            "hidden_size": model_config["dim"],
            "num_layers": model_config["n_layers"],
            "num_heads": model_config["n_heads"],
            "vocab_size": model_config["vocab_size"],
            "max_length": model_config["max_length"],
        },
        "tokens": {token: vocabulary[token] for token in SPECIAL_TOKENS},
        "image_vocabulary": {
            "count": len(image_entries),
            "vocab_id_min": min(image_ids),
            "vocab_id_max": max(image_ids),
            "codebook_id_min": min(codebook_ids),
            "codebook_id_max": max(codebook_ids),
            "complete_zero_to_8191": codebook_ids == set(range(expected_codebook_size)),
        },
        "official_decoder_contract": {
            "text_state": "allow text tokens, EOS, and BOI; image route is not forced",
            "image_state": "BOI starts image decoder; CFG then restricts support to 8192 visual tokens",
            "image_length": 1024,
            "image_end": "official decoder appends EOI after 1024 visual tokens",
            "vq_resolution": 512 if "resolution: 512" in vq_config else None,
        },
        "classification": "asset/token audit only; no model rollout or native result",
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
