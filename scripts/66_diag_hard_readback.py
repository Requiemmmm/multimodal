#!/usr/bin/env python
"""Post-hoc readback for script-10 hard-eval outputs: run the reverse
understanding captioner over every saved image and emit readback.jsonl.

Complements the CLIP scoring; the readback caption is the project's semantic
anchor for generated-image content.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))

QUESTION = "Answer in one short phrase: what is in this image?"


def split_segments(token_ids, boi, eoi, length=1024):
    segments = []
    for index, token in enumerate(token_ids):
        if token == boi and index + length + 1 < len(token_ids) and token_ids[index + length + 1] == eoi:
            segments.append(token_ids[index + 1:index + length + 1])
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--eval-dir", required=True,
                        help="script-10 output dir containing per-seed subdirs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from chameleon.inference.chameleon import TokenManager
    from chameleon.inference.loader import load_model
    from morph.adapters.anole import load_anole_token_sets

    torch.set_default_device("cuda")
    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    token_manager = TokenManager(
        str(tokenizer_path), str(model_path / "tokenizer/vqgan.yaml"),
        str(model_path / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    model = load_model(str(model_path / "models/7b"))
    model.eval()
    import chameleon.inference.transformer as transformer_module
    support = p12.build_mask(token_sets, token_sets.text_step_support)

    eval_dir = Path(args.eval_dir)
    records_path = output / "readback.jsonl"
    with records_path.open("w") as handle:
        for sample_dir in sorted(eval_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            images = sorted(sample_dir.glob("*.png"))
            if not images:
                continue
            tokens = json.loads((sample_dir / "generated_tokens.json").read_text())
            segments = split_segments(tokens, token_sets.boi_id, token_sets.eoi_id)
            for index, visual_ids in enumerate(segments):
                caption = p22.reverse_understanding(
                    token_manager, model, transformer_module, token_sets,
                    visual_ids, QUESTION, 48, 1.2, support)
                record = {"sample": sample_dir.name, "image_index": index,
                          "reverse_understanding": caption}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
    print("done:", records_path)


if __name__ == "__main__":
    main()
