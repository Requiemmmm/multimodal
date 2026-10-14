#!/usr/bin/env python
"""One-process Anole baseline matrix for a scarce-GPU window.

Runs authorized image-only teacher baselines and ordinary-text/native student
baselines while loading the 7B checkpoint only once. Every trajectory is saved
with an explicit mode label; native runs never use the image-only mode flag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TEACHER_PROMPTS = {
    "red_apple_plain": "Draw a red apple on a plain light background.",
    "golden_retriever_plain": "Draw a golden retriever standing on a plain light background.",
    "yellow_school_bus_plain": "Draw a yellow school bus on a plain light background.",
    "blue_ceramic_cup_plain": "Draw a blue ceramic cup on a plain light background.",
}
STUDENT_PROMPTS = {
    "apple_text_question": "Explain in one sentence why apples are a popular fruit.",
    "retriever_text_question": "Explain in one sentence why golden retrievers are suitable family pets.",
    "school_bus_text_question": "Explain in one sentence why school buses are painted yellow.",
    "ceramic_cup_text_question": "Explain in one sentence why ceramic cups are useful for hot drinks.",
}


def split_image_segments(token_ids: list[int], boi: int, eoi: int, image_length: int = 1024):
    segments = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != boi:
            index += 1
            continue
        end = index + 1 + image_length
        if end < len(token_ids) and token_ids[end] == eoi:
            segments.append((index, end, token_ids[index + 1 : end]))
            index = end + 1
        else:
            index += 1
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-forced-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--teacher-freeform-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--student-max-gen-len", type=int, default=1026)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))
    import torch
    from chameleon.inference.chameleon import ChameleonInferenceModel, Options
    from morph.adapters.anole import load_anole_token_sets
    from morph.decoding.anole_segments import analyze_anole_sequence

    tokenizer_path = Path(args.model_path) / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(tokenizer_path)
    model = ChameleonInferenceModel(
        str(Path(args.model_path) / "models/7b"),
        str(tokenizer_path),
        str(Path(args.model_path) / "tokenizer/vqgan.yaml"),
        str(Path(args.model_path) / "tokenizer/vqgan.ckpt"),
    )
    records = []

    def run_one(mode: str, prompt_id: str, prompt: str, seed: int, options: Options):
        options.seed = seed
        generated = model.generate(
            prompt_ui=[
                {"type": "text", "value": prompt},
                {"type": "sentinel", "value": "<END-OF-TURN>"},
            ],
            options=options,
        )
        raw_token_ids = generated[0].tolist() if hasattr(generated[0], "tolist") else list(generated[0])
        # In official image-only mode ImageDecoder inserts BOI into its model
        # context before generation. Preserve that fact separately; it is not
        # a model-sampled/native route token.
        forced_boi_prefix = mode == "teacher_forced_image_only"
        token_ids = (
            [token_sets.boi_id, *raw_token_ids, token_sets.eoi_id]
            if forced_boi_prefix
            else raw_token_ids
        )
        sample_id = f"{mode}_{prompt_id}_seed{seed}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir()
        (sample_dir / "generated_tokens.json").write_text(json.dumps(raw_token_ids) + "\n")
        (sample_dir / "structural_tokens.json").write_text(json.dumps(token_ids) + "\n")
        image_segments = split_image_segments(token_ids, token_sets.boi_id, token_sets.eoi_id)
        image_files = []
        for image_index, (_, _, visual_ids) in enumerate(image_segments):
            image = model.decode_image(torch.tensor([visual_ids], dtype=torch.long, device="cuda"))[0]
            image_path = sample_dir / f"image_{image_index:02d}.png"
            image.save(image_path)
            image_files.append(str(image_path.name))
        structure = analyze_anole_sequence(token_ids, token_sets)
        record = {
            "mode": mode,
            "alpha": 1.0 if mode.startswith("teacher_") else 0.0,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "seed": seed,
            "generated_token_count": len(raw_token_ids),
            "structural_token_count": len(token_ids),
            "image_files": image_files,
            "image_count": len(image_files),
            "model_sampled_route_positions": [] if forced_boi_prefix else [i for i, x in enumerate(structure.route_mask) if x],
            "forced_boi_positions": [0] if forced_boi_prefix else [],
            "visual_token_count": sum(structure.visual_mask),
            "forced_eoi_positions": [i for i, x in enumerate(structure.forced_mask) if x],
            "structure_valid": structure.valid,
            "structure_error": structure.error,
        }
        (sample_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    for prompt_id, prompt in TEACHER_PROMPTS.items():
        for seed in args.teacher_forced_seeds:
            # This is the official authorized image-only path (T2/teacher),
            # retained separately from native student results.
            options = Options()
            options.txt = False
            run_one("teacher_forced_image_only", prompt_id, prompt, seed, options)
    for prompt_id, prompt in TEACHER_PROMPTS.items():
        for seed in args.teacher_freeform_seeds:
            # Authorized prompt, but normal free-form TextDecoder: BOI remains
            # a model-sampled route decision rather than a decoder insertion.
            run_one("teacher_freeform_authorized", prompt_id, prompt, seed, Options(max_gen_len=args.student_max_gen_len))
    for prompt_id, prompt in STUDENT_PROMPTS.items():
        for seed in args.student_seeds:
            # Native student: default free-form Options, no mode flag, BOI
            # forcing, assistant prefill, or output control token injection.
            run_one("student_only", prompt_id, prompt, seed, Options(max_gen_len=args.student_max_gen_len))
    (output_dir / "metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output_dir / "summary.json").write_text(json.dumps({"count": len(records), "records": records}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
