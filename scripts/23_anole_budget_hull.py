#!/usr/bin/env python
"""Budget-constrained hull distance follow-up (run 23).

Run 21 established the per-slot optimal soft directions are ~0.94 far from
the single-token hull (random level ~0.956) while the unconstrained cone
spans the whole space (distance 0 for everything, including random).  The
remaining fair readings for hard-text feasibility:

1. budget-16 distance: min ||R^T a - u|| s.t. a >= 0, sum(a) <= 16 -- a
   16-slot hard suffix can co-align all its tokens on one direction, so this
   is the honest per-direction budget;
2. minimal representation cost: the smallest budget B* at which the distance
   drops to ~0 (LP-type threshold), found by budget bisection.

Inputs are the saved direction tensors from run 21 (no re-optimization):
rho_directions_*.pt, the retry8 route suffix, and random calibrations.
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

spec = importlib.util.spec_from_file_location(
    "hull21", str(PROJECT_ROOT / "scripts/21_anole_hull_distance.py"))
h21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h21)


def project_capped_simplex(a, budget):
    """Project onto {a >= 0, sum(a) <= budget}."""
    import torch
    a = torch.clamp(a, min=0)
    total = a.sum()
    if total > budget:
        sorted_a, _ = torch.sort(a, descending=True)
        cssv = torch.cumsum(sorted_a, 0) - budget
        ind = torch.arange(1, a.numel() + 1, device=a.device)
        cond = sorted_a - cssv / ind > 0
        rho = int(ind[cond].max()) if cond.any() else 1
        theta = cssv[rho - 1] / rho
        a = torch.clamp(a - theta, min=0)
    return a


def solve_budgeted(rows, u, budget, iters=4000):
    """min ||rows^T a - u||^2 s.t. a >= 0, sum(a) <= budget (FISTA)."""
    import torch
    v_count = rows.shape[0]
    probe = torch.randn(v_count, device=rows.device)
    for _ in range(30):
        probe = rows @ (rows.T @ probe)
        probe = probe / probe.norm()
    largest = probe @ (rows @ (rows.T @ probe))
    lr = 1.0 / (2.0 * largest + 1e-6)
    a = torch.zeros(v_count, device=rows.device)
    y = a.clone()
    momentum = 1.0
    for _ in range(iters):
        residual = rows.T @ y - u
        grad = rows @ residual
        a_new = project_capped_simplex(y - lr * grad, budget)
        momentum_new = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        y = a_new + ((momentum - 1.0) / momentum_new) * (a_new - a)
        a, momentum = a_new, momentum_new
    return float((rows.T @ a - u).square().sum().sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--directions-dir", required=True,
                        help="run 21 results dir with rho_directions_*.pt")
    parser.add_argument("--extra-suffix-path", default=None)
    parser.add_argument("--budget", type=float, default=16.0)
    parser.add_argument("--calibration-seeds", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from chameleon.inference.loader import load_model
    from morph.adapters.anole import load_anole_token_sets

    torch.set_default_device("cuda")
    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    model = load_model(str(model_path / "models/7b"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    embeddings = model.tok_embeddings.weight.detach().float()
    center = embeddings.mean(dim=0)
    text_rows = sorted(token_sets.text_token_ids)
    centered = embeddings[torch.tensor(text_rows, device="cuda")] - center
    rays = torch.nn.functional.normalize(centered, dim=-1)

    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    directions_dir = Path(args.directions_dir)
    sources = {}
    for path in sorted(directions_dir.glob("rho_directions_*.pt")):
        name = "rho_" + path.stem.replace("rho_directions_", "")
        sources[name] = torch.load(path, map_location="cpu")["directions"].float().to("cuda")
    if args.extra_suffix_path:
        state = torch.load(args.extra_suffix_path, map_location="cpu")
        extra = state["suffix"].float().to("cuda")
        torch.manual_seed(0)
        mean_embedding = embeddings.mean(dim=0)
        w_init = (mean_embedding.repeat(extra.shape[0], 1)
                  + 0.01 * torch.randn(extra.shape[0], mean_embedding.numel(), device="cuda"))
        sources["retry8_route"] = torch.nn.functional.normalize(extra - w_init, dim=-1)
    for seed in range(args.calibration_seeds):
        torch.manual_seed(1000 + seed)
        sources[f"random_{seed}"] = torch.nn.functional.normalize(
            torch.randn(16, rays.shape[1], device="cuda"), dim=-1)

    budgets = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    for name, slots in sources.items():
        mean_direction = torch.nn.functional.normalize(slots.mean(dim=0), dim=-1)
        # Minimal-cost bisection on the mean direction.
        distances_by_budget = {}
        for budget in budgets:
            distances_by_budget[budget] = solve_budgeted(rays, mean_direction, budget)
        minimal_budget = None
        for budget in budgets:
            if distances_by_budget[budget] < 0.05:
                minimal_budget = budget
                break
        # Budget-16 (the honest per-suffix budget) for every slot.
        slot_distances = [solve_budgeted(rays, slot, args.budget) for slot in slots]
        append({
            "kind": "budget_hull_summary", "source": name,
            "mean_direction_distances_by_budget": distances_by_budget,
            "minimal_budget_05": minimal_budget,
            "per_slot_budget_distance_mean": sum(slot_distances) / len(slot_distances),
            "per_slot_budget_distance_min": min(slot_distances),
            "per_slot_budget_distance_max": max(slot_distances),
            "per_slot_budget_distance": slot_distances,
            "budget": args.budget,
        })
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
