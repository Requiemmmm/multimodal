#!/usr/bin/env python
"""Forced-BOI no-suffix control (run 56): what does the model draw by
default when forced into image mode on a given prompt, with NO suffix?

Control for the content-flip claims: if the strawberry prompt's forced-BOI
default is already apples, the 13/16 flip is the model's own prior, not the
suffix.  Uses p12.run_trajectory("force_boi_only"), which advances the forced
BOI (faithful post-BOI protocol), plus CLIP and Self-SpectraReward readback.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

p12 = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py")))
importlib.util.spec_from_file_location(
    "ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py")).loader.exec_module(p12)
p22 = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py")))
importlib.util.spec_from_file_location(
    "probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py")).loader.exec_module(p22)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-id", default="strawberry")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--patch-layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--clip-dir", default=os.environ.get("CLIP_DIR", "") or None)
    parser.add_argument("--extra-captions", nargs="*", default=None)
    parser.add_argument("--max-text-steps", type=int, default=128)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--image-length", type=int, default=1024)
    parser.add_argument("--guidance-text", type=float, default=3.0)
    parser.add_argument("--guidance-image", type=float, default=1.2)
    parser.add_argument("--image-temp", type=float, default=0.7)
    parser.add_argument("--image-top-p", type=float, default=0.9)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--text-top-p", type=float, default=0.9)
    parser.add_argument("--caption-steps", type=int, default=48)
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
    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": p22.STUDENT_PROMPTS[args.prompt_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for seed in args.seeds:
        t0 = time.time()
        record = p12.run_trajectory(
            model, transformer, token_manager, token_sets, masks, args,
            "force_boi_only", prompt_ids, None, seed, output)
        record["prompt_id"] = args.prompt_id
        if record.get("image_path"):
            image = Image.open(output / record["image_path"])
            if clip_model is not None:
                record["clip_target_similarity"] = clip_similarity(
                    image, p22.TARGET_CAPTIONS[args.prompt_id])
                record["clip_distractor_similarity"] = clip_similarity(image, "a photo of a car")
            if args.extra_captions:
                record["extra_clip"] = {caption: clip_similarity(image, caption)
                                        for caption in args.extra_captions}
            visual = [t for t in record["generated_tokens"] if t in token_sets.image_token_ids]
            record["reverse_understanding"] = p22.reverse_understanding(
                token_manager, model, transformer, token_sets, visual, p22.QUESTION,
                args.caption_steps, args.text_repetition_penalty, masks[0])
        record["seconds"] = round(time.time() - t0, 1)
        append(record)
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
