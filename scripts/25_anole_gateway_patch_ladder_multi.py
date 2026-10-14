#!/usr/bin/env python
"""P1 gateway patch ladder amplification (run 25).

Multi-prompt, multi-seed repetition of the run 12 ladder with CLIP semantic
evaluation on every completed image.  The run 12 slice (n=4, red apple only)
showed the BOI-position readout patch pulls pixel MSE ~4x closer to the
target; this run asks whether that separation holds across prompts and seeds.

Intervention ladder (per prompt, from scripts/12):
1. student_none                    ordinary free decoding (control)
2. boundary_patch_final_resid      teacher EOT-position final residual into the
                                   student boundary position
3. force_boi_only                  force BOI as the first generated token
4. force_boi_patch_final_resid     force BOI + patch final residual at BOI
5. force_boi_patch_late_layers     force BOI + patch late-layer inputs at BOI
6. force_boi_patch_soi_vis1        force BOI + late-layer patches at BOI/vis1

All interventions are T3 mechanism diagnostics; student_none is the control.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p12)

PROMPTS = {
    "red_apple": (
        "Draw a red apple on a plain light background.",
        "Explain in one sentence why apples are a popular fruit.",
    ),
    "golden_retriever": (
        "Draw a golden retriever standing on a plain light background.",
        "Explain in one sentence why golden retrievers are suitable family pets.",
    ),
    "yellow_school_bus": (
        "Draw a yellow school bus on a plain light background.",
        "Explain in one sentence why school buses are yellow.",
    ),
    "blue_ceramic_cup": (
        "Draw a blue ceramic cup on a plain light background.",
        "Explain in one sentence why ceramic cups are useful.",
    ),
}
TARGET_CAPTIONS = {
    "red_apple": "a photo of a red apple",
    "golden_retriever": "a photo of a golden retriever dog",
    "yellow_school_bus": "a photo of a yellow school bus",
    "blue_ceramic_cup": "a photo of a blue ceramic cup",
}
DISTRACTOR_CAPTION = "a photo of a car"

KINDS = ["student_none", "boundary_patch_final_resid",
         "force_boi_only", "force_boi_patch_final_resid",
         "force_boi_patch_late_layers", "force_boi_patch_soi_vis1"]
FORCED_BOI_KINDS = {
    "force_boi_only", "force_boi_patch_final_resid",
    "force_boi_patch_late_layers", "force_boi_patch_soi_vis1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=list(PROMPTS))
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--patch-layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--max-text-steps", type=int, default=128)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--image-length", type=int, default=1024)
    parser.add_argument("--guidance-text", type=float, default=3.0)
    parser.add_argument("--guidance-image", type=float, default=1.2)
    parser.add_argument("--image-temp", type=float, default=0.7)
    parser.add_argument("--image-top-p", type=float, default=0.9)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--text-top-p", type=float, default=0.9)
    parser.add_argument("--clip-dir", default=os.environ.get("CLIP_DIR", "") or None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    (output / "images").mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from PIL import Image
    from chameleon.inference import transformer
    from chameleon.inference.loader import load_model
    from chameleon.inference.chameleon import TokenManager
    from morph.adapters.anole import load_anole_token_sets

    torch.set_default_device("cuda")
    tokenizer_path = Path(args.model_path) / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    token_manager = TokenManager(
        str(tokenizer_path), str(Path(args.model_path) / "tokenizer/vqgan.yaml"),
        str(Path(args.model_path) / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    model = load_model(str(Path(args.model_path) / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    clip_model, clip_proc = None, None
    if Path(args.clip_dir).is_dir():
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(args.clip_dir).to("cuda").eval()
        clip_proc = CLIPProcessor.from_pretrained(args.clip_dir)

    def clip_similarity(image, caption):
        inputs = clip_proc(text=[caption], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            feat_i = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            feat_t = clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            return float(torch.nn.functional.cosine_similarity(feat_i, feat_t).item())

    masks = (
        p12.build_mask(token_sets, token_sets.text_step_support),
        p12.build_mask(token_sets, token_sets.image_token_ids),
    )
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for prompt_id in args.prompt_ids:
        if prompt_id not in PROMPTS:
            raise SystemExit(f"unknown prompt id: {prompt_id}")
        teacher_text, student_text = PROMPTS[prompt_id]
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": teacher_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": student_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])

        donor_record = p12.run_trajectory(
            model, transformer, token_manager, token_sets, masks, args,
            "teacher_freeform_control", teacher_ids, None, args.donor_seed, output,
        )
        donor_record["prompt_id"] = prompt_id
        append(donor_record)
        if not donor_record.get("boi_sampled"):
            print(f"donor did not sample BOI for {prompt_id}; skipping", flush=True)
            continue
        boi_step = donor_record["boi_step"]
        positions = {
            "eot": len(teacher_ids) - 1,
            "boi": len(teacher_ids) + boi_step,
            "vis1": len(teacher_ids) + boi_step + 1,
        }
        capture_ids = teacher_ids + donor_record["generated_tokens"][:boi_step + 2]
        donor_inputs, donor_finals = p12.capture_states(
            model, transformer, capture_ids, positions, set(args.patch_layers)
        )
        donor = {
            "meta": {"prompt_id": f"{prompt_id}_teacher", "seed": args.donor_seed,
                     "boi_step": boi_step, "capture_positions": positions},
            "inputs": donor_inputs,
            "finals": donor_finals,
            "patched_positions": [],
        }

        for kind in KINDS:
            if kind in FORCED_BOI_KINDS:
                donor["patched_positions"] = ["boi"] if kind != "force_boi_patch_soi_vis1" else ["boi", "vis1"]
            else:
                donor["patched_positions"] = ["eot"] if kind == "boundary_patch_final_resid" else []
            for seed in args.seeds:
                t0 = time.time()
                record = p12.run_trajectory(
                    model, transformer, token_manager, token_sets, masks, args,
                    kind, student_ids, donor, seed, output,
                )
                record["prompt_id"] = prompt_id
                if record.get("image_path"):
                    image = Image.open(output / record["image_path"])
                    if clip_model is not None:
                        record["clip_target_similarity"] = clip_similarity(
                            image, TARGET_CAPTIONS[prompt_id])
                        record["clip_distractor_similarity"] = clip_similarity(
                            image, DISTRACTOR_CAPTION)
                record["seconds"] = round(time.time() - t0, 1)
                append(record)

    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
