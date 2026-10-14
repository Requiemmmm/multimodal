#!/usr/bin/env python
"""Full-image student-only evaluation of a trained soft suffix (soft-control).

Continues the retry8-style student-only rollout past BOI through the
replicated official image decoder (three-branch InstructCFG, forced EOI),
saving the completed image.  The soft suffix is continuous embeddings, so
every result here is soft-control evidence, never a T1 hard-text claim.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))

from morph.decoding.processed_policy import apply_common_warpers, apply_repetition_penalty, masked_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--soft-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--image-length", type=int, default=1024)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
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
    prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in distill.PROMPTS.items()
    }
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
            record = {"kind": "soft_suffix_full_image_rollout", "evidence_class": "soft-control",
                      "prompt_id": prompt_id, "seed": seed}
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
            record.update({"boi_sampled": boi_step is not None, "boi_step": boi_step,
                           "text_preamble_tokens": sampled})
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
            post_text = ""
            if post:
                post_text = token_manager.decode_text(torch.tensor([post], dtype=torch.long))[0]
            record.update({
                "sampled_tokens": sampled,
                "visual_count": len(visual),
                "image_path": image_path,
                "image_stats": {"mean": [round(v, 2) for v in s.mean],
                                "stddev": [round(v, 2) for v in s.stddev]},
                "post_image_text": post_text,
                "semantics": "placeholder_manual_inspection",
                "seconds": round(time.time() - t0, 1),
            })
            append(record)
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
