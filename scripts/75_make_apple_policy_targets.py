#!/usr/bin/env python
"""Build a *distributional* apple target: the model's own next-token image policy
while it is drawing an apple.

Why not a trajectory (run 117/118): the 14 CLIP-verified apple images in run 117
start with **five different first visual tokens**, and the trajectories are
dominated by recurring filler tokens (5799/7426).  A token-level target therefore
has to pick one arbitrary member of a family and then spends most of its
gradient on filler -- which is what every content arm so far has done, and why
run 118 improved the margin more than any other arm and still produced 0 apples.

This script instead records, at each of the first K visual positions, the whole
distribution over image tokens that the model has under the authorized
instruction "Draw a red apple", and averages that over the validated rollouts.
Matching a distribution rather than a token sidesteps the "which token is the
apple token" problem entirely: whatever the model's apple-drawing policy puts
mass on is the target, including all five first-token variants.

Only K positions are generated, not 1024: position k's policy depends solely on
the prefix up to k, so the target costs seconds per seed rather than minutes.
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

from morph.decoding.processed_policy import masked_logprobs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--teacher-text", default="Draw a red apple")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="validated seeds to average the policy over")
    parser.add_argument("--content-steps", type=int, default=32)
    parser.add_argument("--top-n", type=int, default=512,
                        help="tokens kept per position in the emitted target")
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

    boi_id, eos_id = token_sets.boi_id, token_sets.eos_id
    K = args.content_steps
    text_mask = p12.build_mask(token_sets, token_sets.text_step_support)
    image_mask = p12.build_mask(token_sets, token_sets.image_token_ids)

    def teacher_policies(text, seed):
        """First K image-policy distributions of an authorized teacher rollout."""
        torch.manual_seed(seed)
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        branch = p12.Branch(model, transformer_module, teacher_ids,
                            len(teacher_ids) + 256 + K + 8)
        for _step in range(128):
            history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
            policy = p12.text_policy(branch.logits, history, text_mask, 1.2, 0.9)
            token = p12.sample_token(policy)
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
        imgcond = p12.Branch(model, transformer_module, imgcond_ids, K + 8)
        uncond = p12.Branch(model, transformer_module,
                            [token_sets.bos_id, boi_id], K + 8)
        policies, sampled = [], []
        for _ in range(K):
            mixed = (uncond.logits
                     + 1.2 * (imgcond.logits - uncond.logits)
                     + 3.0 * (branch.logits - imgcond.logits))
            # Unwarped masked log-softmax: the student side in
            # morph_hard_objectives.content() uses the same processing, so the
            # two distributions are directly comparable.
            lp = masked_logprobs(mixed, image_mask)
            policies.append(lp)
            token = p12.sample_token(p12.image_policy(mixed, image_mask, 0.7, 0.9))
            sampled.append(int(token))
            branch.advance(token)
            imgcond.advance(token)
            uncond.advance(token)
        return policies, sampled

    stacked = None
    per_seed = []
    for seed in args.seeds:
        policies, sampled = teacher_policies(args.teacher_text, seed)
        stack = torch.stack(policies)  # [K, V]
        stacked = stack if stacked is None else stacked + stack
        per_seed.append({"seed": seed, "sampled_head": sampled[:8]})
        print(json.dumps({"kind": "apple_policy_seed", "seed": seed,
                          "sampled_head": sampled[:8]}), flush=True)

    mean_policy = stacked / len(args.seeds)          # [K, V] mean log-prob
    mean_probs = mean_policy.exp()
    mean_probs = mean_probs / mean_probs.sum(-1, keepdim=True)
    entropy = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(-1)

    # Keep only the top-N tokens per position: the full [K, V] table is ~2M
    # floats (~50 MB of JSON) and everything outside the head carries
    # negligible mass.  Mass kept is reported so the truncation is auditable.
    n_keep = min(args.top_n, mean_probs.shape[-1])
    top_probs, top_ids = mean_probs.topk(n_keep, dim=-1)
    mass_kept = top_probs.sum(-1)

    payload = {
        "kind": "apple_policy_targets",
        "teacher_text": args.teacher_text,
        "content_steps": K,
        "n_rollouts": len(args.seeds),
        "seeds": list(args.seeds),
        "vocab_size": int(mean_probs.shape[-1]),
        "top_n": int(n_keep),
        "target_top_ids": [[int(t) for t in row] for row in top_ids],
        "target_top_probs": [[float(p) for p in row] for row in top_probs],
        "mass_kept_by_position": [round(float(m), 6) for m in mass_kept],
        "per_position_entropy_nats": [round(float(e), 4) for e in entropy],
        "top1_token_by_position": [int(i) for i in mean_probs.argmax(-1)],
        "top1_prob_by_position": [round(float(p), 4)
                                  for p in mean_probs.max(-1).values],
        "sampled_heads": per_seed,
        "note": ("mean over validated apple rollouts of the model's own "
                 "unwarped image policy at each of the first K visual positions; "
                 "distribution truncated to the top-N tokens per position"),
    }
    (output / "apple_policy_targets.json").write_text(
        json.dumps(payload) + "\n")
    print(json.dumps({
        "kind": "apple_policy_targets_written",
        "n_rollouts": len(args.seeds), "content_steps": K,
        "mean_entropy_nats": round(float(entropy.mean()), 4),
        "top1_tokens": payload["top1_token_by_position"][:8],
        "top1_probs": payload["top1_prob_by_position"][:8],
        "output": str(output / "apple_policy_targets.json")}, indent=2),
        flush=True)


if __name__ == "__main__":
    main()
