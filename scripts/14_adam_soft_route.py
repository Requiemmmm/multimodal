#!/usr/bin/env python
"""P3 upgrade: exact-forward soft-suffix M-step with true Adam gradients.

Replaces the SPSA fallback.  The forward is the bit-exact official Anole
decoder path (verified max abs diff 0 vs the official layer calls), and the
backward is the analytic construction from P0_backward_feasibility.py:
official rope_padded forward + flash attention (BHMK causal layout, same
kernel as the decoder) with a cache-gradient-routed inverse RoPE and analytic
RMSNorm gradients.  Model weights stay frozen; only the soft suffix moves.

Same bridge elites and evaluation protocol as retry7 for comparability, but
the optimizer is Adam and every reported number is labeled exact-forward.
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


feasibility = _load("p0_backward", str(PROJECT_ROOT / "scripts/P0_backward_feasibility.py"))
distill = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))

from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs


def differentiable_route_nll(model, transformer, token_sets, prompt_ids, suffix,
                             target_ids, support):
    """Exact-forward route NLL (BOI-weighted) with a live graph to the suffix."""
    import torch
    if not target_ids or target_ids[-1] != token_sets.boi_id:
        raise ValueError("route elite must end in BOI")
    embeddings = distill.soft_input_embeddings(
        model, prompt_ids[:-1], token_sets.eot_id, suffix, target_ids[:-1]
    )
    logits = feasibility.exact_forward(model, transformer, embeddings)
    first_prediction = len(prompt_ids) + suffix.shape[0] - 1
    history = list(prompt_ids)
    losses = []
    for offset, target in enumerate(target_ids):
        past = torch.tensor(history, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[first_prediction + offset], past, penalty=1.2),
            support,
        )
        weight = 4.0 if target == token_sets.boi_id else 1.0
        losses.append(-weight * policy[target])
        history.append(target)
    return torch.stack(losses).sum() / sum(
        4.0 if token == token_sets.boi_id else 1.0 for token in target_ids
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preferred-alpha", type=float, default=0.5)
    parser.add_argument("--fallback-alphas", type=float, nargs="*", default=[0.75])
    parser.add_argument("--elites-per-prompt", type=int, default=1)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, nargs="*", default=[0, 10, 20, 40])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    if not Path(args.bridge_metrics).is_file():
        raise SystemExit(f"bridge metrics file does not exist: {args.bridge_metrics}")
    output.mkdir(parents=True)
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

    prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in distill.PROMPTS.items()
    }
    elites = distill.select_elites(
        args.bridge_metrics, distill.PROMPTS, args.preferred_alpha,
        args.fallback_alphas, args.elites_per_prompt,
    )
    (output / "selected_elites.json").write_text(json.dumps(elites, indent=2) + "\n")
    support = distill.legal_mask(token_sets)
    torch.manual_seed(args.init_seed)
    mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
    suffix = (
        mean_embedding.repeat(args.suffix_length, 1)
        + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda")
    )
    suffix = suffix.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
    flat_elites = [(prompt_id, elite) for prompt_id, rows in elites.items() for elite in rows]
    metrics_path = output / "metrics.jsonl"

    def append(records):
        with metrics_path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    def evaluate(step, phase):
        records = distill.evaluate_student_only(
            model, transformer, token_sets, prompt_ids, suffix, args.eval_seeds,
            args.max_steps,
        )
        for record in records:
            record["phase"] = phase
            record["step"] = step
        append(records)

    for step in range(args.steps):
        if step in args.eval_steps:
            evaluate(step, "student_only_eval")
        t0 = time.time()
        losses = [
            differentiable_route_nll(model, transformer, token_sets,
                                     prompt_ids[prompt_id], suffix,
                                     elite["target_ids"], support)
            for prompt_id, elite in flat_elites
        ]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(suffix.grad.norm())
        if not torch.isfinite(suffix.grad).all():
            raise RuntimeError("non-finite suffix gradient; aborting M-step")
        optimizer.step()
        record = {
            "kind": "soft_suffix_route_distillation",
            "optimizer": "adam_exact_forward",
            "step": step,
            "student_route_nll": float(loss.detach()),
            "suffix_grad_norm": grad_norm,
            "per_elite_nll": [float(item.detach()) for item in losses],
            "seconds": round(time.time() - t0, 2),
        }
        append([record])
        print(json.dumps(record), flush=True)
    evaluate(args.steps, "trained_soft_suffix")
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length},
               output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
