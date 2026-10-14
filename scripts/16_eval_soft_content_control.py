#!/usr/bin/env python
"""Scaled content-control evaluation of a trained soft suffix.

For each (prompt, seed): student-only text phase with the soft suffix, image
phase through the replicated official decoder, then two semantic evaluations:
1. Self-SpectraReward: feed the completed image back into Anole's own
   understanding branch with a question and greedily decode the answer.
2. CLIP: cosine similarity of the image with the target caption and with a
   fixed distractor caption.

One process per prompt is expected (--prompt-id), run in parallel across
GPUs.  Held-out prompts (--heldout) test route generalization of the suffix
on prompts it was never trained against.  Everything remains soft-control
evidence; hard-text claims are out of scope for this script.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


distill = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))

from morph.decoding.processed_policy import apply_common_warpers, apply_repetition_penalty, masked_logprobs

TARGET_CAPTIONS = {
    "red_apple": "a photo of a red apple",
    "golden_retriever": "a photo of a golden retriever dog",
    "yellow_school_bus": "a photo of a yellow school bus",
    "blue_ceramic_cup": "a photo of a blue ceramic cup",
    "green_frog": "a photo of a green frog",
    "strawberry": "a photo of a red strawberry",
}
DISTRACTOR_CAPTION = "a photo of a car"
QUESTION = "Answer in one short phrase: what is in this image?"


def reverse_understanding(token_manager, model, transformer, token_sets, visual_ids,
                          question, max_steps, penalty, support_mask):
    import torch
    boi_id, eoi_id, eos_id = token_sets.boi_id, token_sets.eoi_id, token_sets.eos_id
    ids = token_manager.tokens_from_ui([
        {"type": "text", "value": question},
        {"type": "ids", "value": [boi_id, *visual_ids, eoi_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    branch = p12.Branch(model, transformer, ids, len(ids) + max_steps + 8)
    out = []
    for _ in range(max_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(branch.logits, history, penalty=penalty),
            support_mask)
        token = int(torch.argmax(policy))
        out.append(token)
        if token == eos_id:
            break
        branch.advance(token)
    return token_manager.decode_text(torch.tensor([out], dtype=torch.long))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--soft-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-id", default=None, help="run only this prompt")
    parser.add_argument("--heldout", action="store_true",
                        help="evaluate on held-out prompts instead of training prompts")
    parser.add_argument("--clip-dir", default=os.environ.get("CLIP_DIR", "") or None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--image-length", type=int, default=1024)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--caption-steps", type=int, default=48)
    parser.add_argument("--guidance-text", type=float, default=3.0)
    parser.add_argument("--guidance-image", type=float, default=1.2)
    parser.add_argument("--image-temp", type=float, default=0.7)
    parser.add_argument("--image-top-p", type=float, default=0.9)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--text-top-p", type=float, default=0.9)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    (output / "images").mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from chameleon.inference import transformer
    from chameleon.inference.loader import load_model
    from chameleon.inference.chameleon import TokenManager
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
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = torch.load(args.soft_state, map_location="cuda")
    suffix = state["suffix"].to("cuda")
    text_support = distill.legal_mask(token_sets)
    image_mask = p12.build_mask(token_sets, token_sets.image_token_ids)

    if args.heldout:
        prompt_set = {
            "green_frog": "Explain in one sentence why frogs are green.",
            "strawberry": "Explain in one sentence why strawberries are red.",
        }
    else:
        prompt_set = distill.PROMPTS
    if args.prompt_id:
        if args.prompt_id not in prompt_set:
            raise SystemExit(f"--prompt-id {args.prompt_id} not in selected prompt set")
        prompt_set = {args.prompt_id: prompt_set[args.prompt_id]}
    prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in prompt_set.items()
    }

    # CLIP semantic scorer
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

    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for prompt_id, ids in prompt_ids.items():
        for seed in args.seeds:
            torch.manual_seed(seed)
            t0 = time.time()
            record = {"kind": "soft_suffix_content_control", "evidence_class": "soft-control",
                      "prompt_id": prompt_id, "seed": seed, "heldout": args.heldout}
            context = distill.SoftTextContext(
                model, transformer, token_sets, ids, suffix,
                args.max_steps + args.image_length + args.post_image_text_steps + 8)
            sampled, boi_step = [], None
            for step in range(args.max_steps):
                history = torch.tensor(context.history, dtype=torch.long, device="cuda")
                policy = masked_logprobs(
                    apply_repetition_penalty(context.logits, history, penalty=1.2),
                    text_support)
                warped = apply_common_warpers(policy, top_p=args.text_top_p)
                token = int(torch.multinomial(warped.exp(), 1).item())
                sampled.append(token)
                if token == token_sets.boi_id:
                    boi_step = step
                    break
                if token == token_sets.eos_id:
                    break
                context.advance(token)
            record.update({"boi_sampled": boi_step is not None, "boi_step": boi_step})
            if boi_step is None:
                record.update({"visual_count": 0, "image_path": None,
                               "seconds": round(time.time() - t0, 1)})
                append(record)
                continue

            image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, token_sets.boi_id, token_sets.eoi_id}
            imgcond_ids = [t for t in ids if t in image_cond_allowed]
            if not imgcond_ids or imgcond_ids[-1] != token_sets.boi_id:
                imgcond_ids.append(token_sets.boi_id)
            uncond_ids = [token_sets.bos_id, token_sets.boi_id]
            aux_length = args.image_length + args.post_image_text_steps + 8
            imgcond = p12.Branch(model, transformer, imgcond_ids, aux_length)
            uncond = p12.Branch(model, transformer, uncond_ids, aux_length)
            visual = []
            for _ in range(args.image_length):
                mixed = (uncond.logits
                         + args.guidance_image * (imgcond.logits - uncond.logits)
                         + args.guidance_text * (context.logits - imgcond.logits))
                policy = p12.image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
                token = p12.sample_token(policy)
                visual.append(token)
                context.advance(token)
                imgcond.advance(token)
                uncond.advance(token)
            context.advance(token_sets.eoi_id)
            sampled.extend(visual)
            sampled.append(token_sets.eoi_id)
            post = []
            for _ in range(args.post_image_text_steps):
                history = torch.tensor(context.history, dtype=torch.long, device="cuda")
                policy = p12.text_policy(context.logits, history, text_support,
                                         args.text_repetition_penalty, args.text_top_p)
                token = p12.sample_token(policy)
                post.append(token)
                sampled.append(token)
                if token == token_sets.eos_id:
                    break
                context.advance(token)

            from PIL import ImageStat
            image = token_manager.pil_from_bpe_tokens(
                torch.tensor(visual, dtype=torch.long, device="cuda"))
            image_path = f"images/{prompt_id}_seed{seed}.png"
            image.save(output / image_path)
            s = ImageStat.Stat(image.convert("RGB"))
            caption = reverse_understanding(
                token_manager, model, transformer, token_sets, visual, QUESTION,
                args.caption_steps, args.text_repetition_penalty, text_support)
            clip_target = clip_similarity(image, TARGET_CAPTIONS[prompt_id]) if clip_model else None
            clip_distractor = clip_similarity(image, DISTRACTOR_CAPTION) if clip_model else None
            post_text = ""
            if post:
                post_text = token_manager.decode_text(torch.tensor([post], dtype=torch.long))[0]
            record.update({
                "sampled_tokens": sampled,
                "visual_count": len(visual),
                "image_path": image_path,
                "image_stats": {"mean": [round(v, 2) for v in s.mean],
                                "stddev": [round(v, 2) for v in s.stddev]},
                "reverse_understanding": caption,
                "clip_target_similarity": clip_target,
                "clip_distractor_similarity": clip_distractor,
                "post_image_text": post_text,
                "seconds": round(time.time() - t0, 1),
            })
            append(record)
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
