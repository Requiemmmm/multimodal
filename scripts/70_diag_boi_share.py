#!/usr/bin/env python
"""Sampling-share diagnostic: for given hard suffixes, compute the official
processed sampling probability of BOI at the first free generation position
(deployment processing: rep penalty over the FULL history incl. suffix,
legality mask, top-p 0.9, temp 1.0).  Answers whether a converged BOI-margin
search actually concentrates sampling mass on BOI.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))

from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--projections", nargs="+", required=True)
    parser.add_argument("--prompt-id", default="strawberry")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    model.eval()
    support = d08.legal_mask(token_sets)

    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": p22.STUDENT_PROMPTS[args.prompt_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])

    records = []
    for projection_path in args.projections:
        projection = json.loads(Path(projection_path).read_text())
        ids = [int(t) for t in projection["token_ids"]]
        suffix = model.tok_embeddings(
            torch.tensor(ids, dtype=torch.long, device="cuda")).detach().float()
        embeds = d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix)
        logits = p14.feasibility.exact_forward(model, transformer, embeds)
        position = len(prompt_ids) + suffix.shape[0] - 1
        row = logits[position]
        # Deployment history (full): prompt + suffix (+ EOT implicitly in prompt_ids).
        full_history = torch.tensor(prompt_ids + ids, dtype=torch.long, device="cuda")
        policy_full = masked_logprobs(
            apply_repetition_penalty(row, full_history, penalty=1.2), support)
        # Search-objective history (prompt only, as script 65 used).
        prompt_history = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        policy_prompt = masked_logprobs(
            apply_repetition_penalty(row, prompt_history, penalty=1.2), support)
        boi = token_sets.boi_id
        for label, policy in [("full_history", policy_full), ("prompt_history", policy_prompt)]:
            sorted_lp, sorted_idx = policy.sort(descending=True)
            keep = (sorted_lp.exp().cumsum(0) <= args.top_p)
            keep[0] = True  # top-1 always in the nucleus
            nucleus = sorted_idx[keep]
            share = float(policy[boi].exp() / policy[nucleus].exp().sum())
            rank = int((policy > policy[boi]).sum().item()) + 1
            records.append({
                "projection": Path(projection_path).name,
                "history": label,
                "boi_lp": round(float(policy[boi]), 4),
                "boi_rank": rank,
                "boi_share_top_p": round(share, 4),
                "top_p": args.top_p,
            })
            print(json.dumps(records[-1]), flush=True)
    (output / "boi_share.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    main()
