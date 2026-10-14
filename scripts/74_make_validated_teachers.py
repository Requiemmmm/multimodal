#!/usr/bin/env python
"""Build a content target from teacher trajectories that are *verified* apples.

Why this exists (run 116): the K=1 mass search -- the one setting where the
teacher-forced and deployed contexts are identical, so the objective is honest
-- maximised the deployed probability of the teacher's first visual token (id
3162) 16x, and the images came back as **glasses/sunglasses 6/7** by readback and
by CLIP, with zero apples.  The "forced-BOI visual contrast" objective family
assumes the teacher's visual tokens faithfully carry the target concept; run 116
falsifies that for the token it can test cleanly.  So the target itself has to be
validated before any further content search means anything.

This script generates full authorized teacher rollouts ("Draw a red apple"),
decodes them, scores each against a caption set with CLIP, keeps only the ones
that are actually apples, and writes a ``teachers.json`` (same schema as run 83)
whose ``target_visual`` comes from those verified trajectories.

Selection is deliberately conservative and recorded per seed, so the filter can
be audited or loosened without regenerating.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

CAPTIONS = [
    "a photo of a red apple",
    "a photo of a strawberry",
    "a photo of a pair of glasses",
    "a photo of a dog",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--teacher-text", default="Draw a red apple")
    parser.add_argument("--victim-text", default="Draw a strawberry")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--visual-steps", type=int, default=1024)
    parser.add_argument("--target-steps", type=int, default=32,
                        help="visual tokens kept in the emitted teachers.json")
    parser.add_argument("--clip-dir",
                        default=os.environ.get("CLIP_DIR", "") or None)
    parser.add_argument("--min-margin", type=float, default=0.02,
                        help="required CLIP margin of apple over the runner-up")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    import chameleon.inference.transformer as transformer_module

    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(args.clip_dir).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_dir)

    boi_id = token_sets.boi_id
    eos_id = token_sets.eos_id
    text_mask = p12.build_mask(token_sets, token_sets.text_step_support)
    image_mask = p12.build_mask(token_sets, token_sets.image_token_ids)

    def teacher_trajectory(text, k_visual):
        """Freeform teacher rollout: preamble (ending in BOI) + k_visual tokens."""
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        branch = p12.Branch(model, transformer_module, teacher_ids,
                            len(teacher_ids) + 256 + k_visual + 8)
        preamble = []
        for _step in range(128):
            history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
            policy = p12.text_policy(branch.logits, history, text_mask, 1.2, 0.9)
            token = p12.sample_token(policy)
            preamble.append(token)
            branch.advance(token)
            if token == boi_id:
                break
            if token == eos_id:
                raise RuntimeError(f"teacher hit EOS without BOI: {text}")
        image_cond_allowed = set(token_sets.image_token_ids) | {
            token_sets.bos_id, boi_id, token_sets.eoi_id}
        imgcond_ids = [t for t in teacher_ids if t in image_cond_allowed]
        if not imgcond_ids or imgcond_ids[-1] != boi_id:
            imgcond_ids.append(boi_id)
        aux = k_visual + 8
        imgcond = p12.Branch(model, transformer_module, imgcond_ids, aux)
        uncond = p12.Branch(model, transformer_module,
                            [token_sets.bos_id, boi_id], aux)
        visual = []
        for _ in range(k_visual):
            mixed = (uncond.logits
                     + 1.2 * (imgcond.logits - uncond.logits)
                     + 3.0 * (branch.logits - imgcond.logits))
            policy = p12.image_policy(mixed, image_mask, 0.7, 0.9)
            token = p12.sample_token(policy)
            visual.append(token)
            branch.advance(token)
            imgcond.advance(token)
            uncond.advance(token)
        return preamble, visual

    def clip_scores(image):
        inputs = clip_proc(text=CAPTIONS, images=image, return_tensors="pt",
                           padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            fi = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            ft = clip_model.get_text_features(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])
        return torch.nn.functional.cosine_similarity(fi, ft).tolist()

    records, validated = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        preamble, visual = teacher_trajectory(args.teacher_text, args.visual_steps)
        image = token_manager.decode_image(
            torch.tensor([visual], dtype=torch.long, device="cuda"))[0]
        image_path = output / f"teacher_seed{seed}.png"
        image.save(image_path)
        sims = clip_scores(image)
        order = sorted(range(len(sims)), key=lambda i: -sims[i])
        apple_margin = sims[0] - max(sims[1:])
        ok = order[0] == 0 and apple_margin >= args.min_margin
        record = {
            "kind": "validated_teacher_rollout", "seed": seed,
            "clip_apple": round(sims[0], 4),
            "clip_runner_up": round(max(sims[1:]), 4),
            "apple_margin": round(apple_margin, 4),
            "argmax_caption": CAPTIONS[order[0]],
            "validated_apple": bool(ok),
            "image": image_path.name,
            "preamble_len": len(preamble),
            "visual_head": visual[:args.target_steps],
        }
        records.append(record)
        if ok:
            validated.append(record)
        print(json.dumps({k: v for k, v in record.items()
                          if k != "visual_head"}), flush=True)

    validated.sort(key=lambda r: -r["apple_margin"])
    n_keep = min(4, len(validated))
    if n_keep == 0:
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
        raise SystemExit(
            "no teacher rollout passed the apple filter; do not build a target "
            "from unvalidated trajectories -- report and stop")

    target_visual = validated[0]["visual_head"]
    # Victim trajectory: same generator, unvalidated on purpose (it only has to
    # be a plausible strawberry, and the search pushes away from it).
    torch.manual_seed(args.seeds[0] + 10000)
    _vp, victim_visual = teacher_trajectory(args.victim_text, args.target_steps)
    payload = {
        "target_text": args.teacher_text,
        "victim_text": args.victim_text,
        "target_visual": target_visual,
        "victim_visual": victim_visual,
        "target_visual_trajectories": [r["visual_head"] for r in validated[:n_keep]],
        "n_target_trajectories": n_keep,
        "n_victim_trajectories": 1,
        "provenance": {
            "builder": "scripts/74_make_validated_teachers.py",
            "selection": f"CLIP argmax == apple and margin >= {args.min_margin}",
            "n_rollouts": len(records),
            "n_validated": len(validated),
            "validated_seeds": [r["seed"] for r in validated[:n_keep]],
            "captions": CAPTIONS,
        },
    }
    (output / "teachers.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps({"kind": "validated_teachers_written",
                      "n_rollouts": len(records), "n_validated": len(validated),
                      "kept_seeds": [r["seed"] for r in validated[:n_keep]],
                      "target_margins": [round(r["apple_margin"], 4)
                                         for r in validated[:n_keep]],
                      "teachers_json": str(output / "teachers.json")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
