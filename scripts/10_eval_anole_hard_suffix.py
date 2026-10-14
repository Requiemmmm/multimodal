#!/usr/bin/env python
"""Evaluate a projected ordinary-text suffix with Anole's native decoder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = {
    "red_apple": "Explain in one sentence why apples are a popular fruit.",
    "golden_retriever": "Explain in one sentence why golden retrievers are suitable family pets.",
    "yellow_school_bus": "Explain in one sentence why school buses are painted yellow.",
    "blue_ceramic_cup": "Explain in one sentence why ceramic cups are useful for hot drinks.",
    "sky_blue": "Explain in one sentence why the sky is blue.",
    "strawberry": "Explain in one sentence why strawberries are red.",
    "green_frog": "Explain in one sentence why frogs are green.",
    "tomato": "Explain in one sentence why tomatoes are red.",
}


def load_prompts(prompt_map_path):
    """Optionally replace the built-in prompt table with a JSON prompt-id -> text map."""
    prompts = dict(PROMPTS)
    if prompt_map_path:
        prompts = json.loads(Path(prompt_map_path).read_text())
        if not prompts:
            raise SystemExit(f"prompt map is empty: {prompt_map_path}")
    return prompts


def split_segments(token_ids, boi, eoi, length=1024):
    segments = []
    for index, token in enumerate(token_ids):
        if token == boi and index + length + 1 < len(token_ids) and token_ids[index + length + 1] == eoi:
            segments.append(token_ids[index + 1 : index + length + 1])
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(32, 64)))
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--prompt-map", default=None,
                        help="JSON file mapping prompt_id -> prompt text; defaults to the built-in table")
    args = parser.parse_args()
    prompts = load_prompts(args.prompt_map)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    projection = json.loads(Path(args.projection).read_text())
    suffix_ids = projection.get("token_ids")
    if not isinstance(suffix_ids, list) or not suffix_ids:
        raise SystemExit("projection has no token_ids")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))
    import torch
    from chameleon.inference.chameleon import ChameleonInferenceModel, Options, TokenManager
    from morph.adapters.anole import load_anole_token_sets

    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(tokenizer_path)
    model = ChameleonInferenceModel(
        str(model_path / "models/7b"), str(tokenizer_path),
        str(model_path / "tokenizer/vqgan.yaml"), str(model_path / "tokenizer/vqgan.ckpt"),
    )
    manager = TokenManager(str(tokenizer_path), str(model_path / "tokenizer/vqgan.yaml"), str(model_path / "tokenizer/vqgan.ckpt"), device="cuda")
    records = []
    for prompt_id, prompt in prompts.items():
        for seed in args.seeds:
            options = Options(max_gen_len=args.max_gen_len)
            options.seed = seed
            generated = model.generate(
                prompt_ui=[
                    # Preserve the exact projected IDs.  Feeding a decoded
                    # Unicode string back through BPE is not an inverse for
                    # every token and would invalidate a projection test.
                    {"type": "text", "value": prompt},
                    {"type": "ids", "value": suffix_ids},
                    {"type": "sentinel", "value": "<END-OF-TURN>"},
                ],
                options=options,
            )
            raw = generated[0].tolist() if hasattr(generated[0], "tolist") else list(generated[0])
            sample_dir = output / f"{prompt_id}_seed{seed}"
            sample_dir.mkdir()
            (sample_dir / "generated_tokens.json").write_text(json.dumps(raw) + "\n")
            segments = split_segments(raw, token_sets.boi_id, token_sets.eoi_id)
            image_files = []
            for index, visual_ids in enumerate(segments):
                image = model.decode_image(torch.tensor([visual_ids], dtype=torch.long, device="cuda"))[0]
                path = sample_dir / f"image_{index:02d}.png"
                image.save(path)
                image_files.append(path.name)
            record = {
                "kind": "hard_projected_suffix_student_only",
                "mode": "student_only_native",
                "prompt_id": prompt_id,
                "seed": seed,
                "suffix_ids": suffix_ids,
                "suffix_text": None,
                "suffix_submission": "exact_public_ui_ids_segment",
                "generated_token_count": len(raw),
                "model_sampled_boi": token_sets.boi_id in raw,
                "image_count": len(image_files),
                "image_files": image_files,
                "control_or_visual_suffix_ids": bool(set(suffix_ids) & ({token_sets.boi_id, token_sets.eoi_id, token_sets.eot_id} | set(token_sets.image_token_ids))),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (output / "metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
