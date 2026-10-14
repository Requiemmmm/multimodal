#!/usr/bin/env python
"""P2: commitment horizon on Anole (T3 diagnostics).

Inject an authorized trajectory's output prefix -- the BOI plus the first
``k`` visual tokens -- under the ordinary student prompt, then continue the
image freely with the replicated official decoder (three-branch InstructCFG,
temp 0.7, top-p 0.9, forced EOI after 1024 visual IDs).  Measures, per k:

- first divergence position / token agreement against the donor continuation;
- mean sampled-token log-prob over the free continuation;
- pixel MSE between the completed image and the donor's completed image;
- structural validity and post-image text.

The injection is forced-structure T3 diagnosis, never a T1 native claim.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEACHER_PROMPT = "Draw a red apple on a plain light background."
STUDENT_PROMPT = "Explain in one sentence why apples are a popular fruit."


def build_mask(token_sets, ids):
    import torch
    mask = torch.zeros(token_sets.vocab_size, dtype=torch.bool, device="cuda")
    mask[list(ids)] = True
    return mask


class Branch:
    """No-grad cache state for one decoder branch (official kernels)."""

    def __init__(self, model, transformer, token_ids, max_length):
        import torch
        self.model = model
        self.history = list(token_ids)
        self.cache = transformer.make_cache(
            model.args, length=max_length, device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        self.bias = transformer.AttnBias.from_seqlens(
            q_seqlen=[len(token_ids)], kv_seqlen=[len(token_ids)],
            kv_padding=max_length,
        )
        hidden = model.tok_embeddings(
            torch.tensor(token_ids, dtype=torch.long, device="cuda")
        )
        with torch.no_grad():
            for index, layer in enumerate(model.layers):
                hidden = layer(hidden, self.cache[index], self.bias)
            self.logits = model.output(model.norm(hidden)).float()[-1]
        self.bias.q_seqinfo.seqstart.copy_(
            torch.arange(2, dtype=torch.int, device="cuda")
        )
        self.bias.q_seqinfo.max_seqlen = 1
        self.bias.q_seqinfo.seqstart_py = self.bias.q_seqinfo.seqstart.tolist()
        self.bias.k_seqinfo.seqlen.add_(1)

    def advance(self, token):
        import torch
        hidden = self.model.tok_embeddings(
            torch.tensor([token], dtype=torch.long, device="cuda")
        )
        with torch.no_grad():
            for index, layer in enumerate(self.model.layers):
                hidden = layer(hidden, self.cache[index], self.bias)
            self.logits = self.model.output(self.model.norm(hidden)).float()[0]
        self.bias.k_seqinfo.seqlen.add_(1)
        self.history.append(token)


def text_policy(raw_logits, history, support_mask, penalty, top_p):
    from morph.decoding.processed_policy import (
        apply_common_warpers, apply_repetition_penalty, masked_logprobs,
    )
    policy = masked_logprobs(
        apply_repetition_penalty(raw_logits, history, penalty=penalty),
        support_mask,
    )
    return apply_common_warpers(policy, temperature=1.0, top_p=top_p)


def image_policy(mixed_logits, image_mask, temperature, top_p):
    from morph.decoding.processed_policy import apply_common_warpers, masked_logprobs
    policy = masked_logprobs(mixed_logits, image_mask)
    return apply_common_warpers(policy, temperature=temperature, top_p=top_p)


def sample_token(warped_logprobs):
    import torch
    return int(torch.multinomial(warped_logprobs.exp(), 1).item())


def run_authorized(model, transformer, token_manager, token_sets, masks, args,
                   teacher_ids, seed):
    """Teacher free-form trajectory with the replicated official decoder."""
    import torch
    torch.manual_seed(seed)
    text_support, image_mask = masks
    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    branch = Branch(model, transformer, teacher_ids,
                    len(teacher_ids) + args.max_text_steps + args.image_length + args.post_image_text_steps + 8)
    generated = []
    boi_step = None
    for step in range(args.max_text_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = text_policy(branch.logits, history, text_support,
                             args.text_repetition_penalty, args.text_top_p)
        token = sample_token(policy)
        generated.append(token)
        if token == boi_id:
            boi_step = step
            branch.advance(token)
            break
        if token == eos_id:
            break
        branch.advance(token)
    if boi_step is None:
        raise SystemExit("authorized trajectory did not sample BOI natively")
    image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
    imgcond_ids = [t for t in teacher_ids if t in image_cond_allowed]
    if not imgcond_ids or imgcond_ids[-1] != boi_id:
        imgcond_ids.append(boi_id)
    uncond_ids = [token_sets.bos_id, boi_id]
    aux_length = args.image_length + args.post_image_text_steps + 8
    imgcond = Branch(model, transformer, imgcond_ids, aux_length)
    uncond = Branch(model, transformer, uncond_ids, aux_length)
    visual = []
    for _ in range(args.image_length):
        mixed = (uncond.logits
                 + args.guidance_image * (imgcond.logits - uncond.logits)
                 + args.guidance_text * (branch.logits - imgcond.logits))
        policy = image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
        token = sample_token(policy)
        visual.append(token)
        branch.advance(token)
        imgcond.advance(token)
        uncond.advance(token)
    branch.advance(eoi_id)
    generated.extend(visual)
    generated.append(eoi_id)
    post = []
    for _ in range(args.post_image_text_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = text_policy(branch.logits, history, text_support,
                             args.text_repetition_penalty, args.text_top_p)
        token = sample_token(policy)
        post.append(token)
        generated.append(token)
        if token == eos_id:
            break
        branch.advance(token)
    donor_image = token_manager.pil_from_bpe_tokens(
        torch.tensor(visual, dtype=torch.long, device="cuda"))
    return {"boi_step": boi_step, "generated": generated, "visual": visual,
            "image": donor_image, "post_text": post}


def run_horizon(model, transformer, token_manager, token_sets, masks, args,
                student_ids, donor, k, seed, output_dir):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    t0 = time.time()
    text_support, image_mask = masks
    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    record = {
        "kind": "commitment_horizon",
        "evidence_class": "T3",
        "prompt_id": "red_apple",
        "seed": seed,
        "donor_seed": args.donor_seed,
        "injected_visual_prefix": k,
    }
    branch = Branch(model, transformer, student_ids,
                    len(student_ids) + args.image_length + args.post_image_text_steps + 8)
    generated = [boi_id]
    branch.advance(boi_id)

    injected = donor["visual"][:k]
    image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
    imgcond_ids = [t for t in student_ids if t in image_cond_allowed]
    if not imgcond_ids or imgcond_ids[-1] != boi_id:
        imgcond_ids.append(boi_id)
    imgcond_ids.extend(injected)
    uncond_ids = [token_sets.bos_id, boi_id, *injected]
    aux_length = args.image_length + args.post_image_text_steps + 8
    imgcond = Branch(model, transformer, imgcond_ids, aux_length)
    uncond = Branch(model, transformer, uncond_ids, aux_length)
    for token in injected:
        branch.advance(token)
    generated.extend(injected)

    sampled, logprobs = [], []
    for _ in range(args.image_length - k):
        mixed = (uncond.logits
                 + args.guidance_image * (imgcond.logits - uncond.logits)
                 + args.guidance_text * (branch.logits - imgcond.logits))
        policy = image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
        token = sample_token(policy)
        logprobs.append(float(policy[token]))
        sampled.append(token)
        branch.advance(token)
        imgcond.advance(token)
        uncond.advance(token)
    branch.advance(eoi_id)
    generated.extend(sampled)
    generated.append(eoi_id)
    post = []
    for _ in range(args.post_image_text_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = text_policy(branch.logits, history, text_support,
                             args.text_repetition_penalty, args.text_top_p)
        token = sample_token(policy)
        post.append(token)
        generated.append(token)
        if token == eos_id:
            break
        branch.advance(token)

    completed_visual = injected + sampled
    donor_visual = donor["visual"]
    first_divergence = None
    for t in range(len(sampled)):
        if sampled[t] != donor_visual[k + t]:
            first_divergence = t
            break
    agreement = sum(1 for t in range(len(sampled)) if sampled[t] == donor_visual[k + t])

    image = token_manager.pil_from_bpe_tokens(
        torch.tensor(completed_visual, dtype=torch.long, device="cuda"))
    image_path = f"images/k{k:04d}_seed{seed}.png"
    image.save(output_dir / image_path)
    mse = float(np.mean(
        (np.asarray(image, dtype=float) - np.asarray(donor["image"], dtype=float)) ** 2))
    from PIL import ImageStat
    s = ImageStat.Stat(image.convert("RGB"))
    post_text = ""
    if post:
        post_text = token_manager.decode_text(torch.tensor([post], dtype=torch.long))[0]
    record.update({
        "generated_tokens": generated,
        "token_count": len(generated),
        "visual_count": len(completed_visual),
        "injected_visual_ids_match_donor": injected == donor_visual[:k],
        "first_divergence": first_divergence,
        "token_agreement": agreement,
        "agreement_fraction": round(agreement / max(len(sampled), 1), 4),
        "mean_continuation_logprob": round(sum(logprobs) / max(len(logprobs), 1), 4),
        "pixel_mse_vs_donor": round(mse, 2),
        "image_path": image_path,
        "image_stats": {"mean": [round(v, 2) for v in s.mean],
                        "stddev": [round(v, 2) for v in s.stddev]},
        "post_image_text": post_text,
        "semantics": "placeholder_manual_inspection",
        "seconds": round(time.time() - t0, 1),
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--max-text-steps", type=int, default=128)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--image-length", type=int, default=1024)
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

    teacher_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": TEACHER_PROMPT},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    student_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": STUDENT_PROMPT},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    masks = (
        build_mask(token_sets, token_sets.text_step_support),
        build_mask(token_sets, token_sets.image_token_ids),
    )
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    donor = run_authorized(model, transformer, token_manager, token_sets, masks,
                           args, teacher_ids, args.donor_seed)
    append({"kind": "donor_note", "seed": args.donor_seed,
            "boi_step": donor["boi_step"], "visual_count": len(donor["visual"])})
    for k in args.k_values:
        for seed in args.seeds:
            append(run_horizon(model, transformer, token_manager, token_sets, masks,
                               args, student_ids, donor, k, seed, output))
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
