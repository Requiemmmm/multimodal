#!/usr/bin/env python
"""Large native random-text-suffix control matrix for a hard projection test."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = {
    "red_apple": "Explain in one sentence why apples are a popular fruit.",
    "golden_retriever": "Explain in one sentence why golden retrievers are suitable family pets.",
    "yellow_school_bus": "Explain in one sentence why school buses are painted yellow.",
    "blue_ceramic_cup": "Explain in one sentence why ceramic cups are useful for hot drinks.",
}


def split_segments(token_ids, boi, eoi, length=1024):
    return [
        token_ids[index + 1 : index + length + 1]
        for index, token in enumerate(token_ids)
        if token == boi and index + length + 1 < len(token_ids) and token_ids[index + length + 1] == eoi
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=list(range(32, 64)))
    parser.add_argument("--max-gen-len", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    if args.suffix_length <= 0:
        raise SystemExit("--suffix-length must be positive")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))
    import torch
    from chameleon.inference.chameleon import ChameleonInferenceModel, Options
    from morph.adapters.anole import load_anole_token_sets

    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(tokenizer_path)
    text_ids = sorted(token_sets.text_token_ids)
    controls = {
        control_seed: random.Random(control_seed).sample(text_ids, args.suffix_length)
        for control_seed in args.control_seeds
    }
    (output / "random_suffixes.json").write_text(json.dumps(controls, indent=2) + "\n")
    model = ChameleonInferenceModel(
        str(model_path / "models/7b"), str(tokenizer_path),
        str(model_path / "tokenizer/vqgan.yaml"), str(model_path / "tokenizer/vqgan.ckpt"),
    )
    records = []
    for control_seed, suffix_ids in controls.items():
        for prompt_id, prompt in PROMPTS.items():
            for seed in args.rollout_seeds:
                options = Options(max_gen_len=args.max_gen_len)
                options.seed = seed
                generated = model.generate(
                    prompt_ui=[
                        {"type": "text", "value": prompt},
                        {"type": "ids", "value": suffix_ids},
                        {"type": "sentinel", "value": "<END-OF-TURN>"},
                    ],
                    options=options,
                )
                raw = generated[0].tolist() if hasattr(generated[0], "tolist") else list(generated[0])
                sample_dir = output / f"control{control_seed}_{prompt_id}_seed{seed}"
                sample_dir.mkdir()
                (sample_dir / "generated_tokens.json").write_text(json.dumps(raw) + "\n")
                image_files = []
                for index, visual_ids in enumerate(split_segments(raw, token_sets.boi_id, token_sets.eoi_id)):
                    image = model.decode_image(torch.tensor([visual_ids], dtype=torch.long, device="cuda"))[0]
                    image_path = sample_dir / f"image_{index:02d}.png"
                    image.save(image_path)
                    image_files.append(image_path.name)
                record = {
                    "kind": "random_text_suffix_student_only_control",
                    "mode": "student_only_native",
                    "control_seed": control_seed,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "suffix_ids": suffix_ids,
                    "suffix_length": args.suffix_length,
                    "control_or_visual_suffix_ids": bool(set(suffix_ids) & ({token_sets.boi_id, token_sets.eoi_id, token_sets.eot_id} | set(token_sets.image_token_ids))),
                    "generated_token_count": len(raw),
                    "model_sampled_boi": token_sets.boi_id in raw,
                    "image_count": len(image_files),
                    "image_files": image_files,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    (output / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
