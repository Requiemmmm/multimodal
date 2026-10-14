#!/usr/bin/env python
"""Hull/cone distance of the optimal soft directions (run 21).

Final adjudication for hard-text feasibility.  The rho probe (run 17) found a
16-slot soft suffix whose per-slot directions realize the gateway displacement
(rho ~ 0.998) but did not persist the suffix.  This script re-runs the same
optimization, saves the suffix, and measures how far each per-slot optimal
direction u_i = normalize(w_i - w_init_i) lies from the text-vocabulary cone

    C = { sum_v alpha_v (E_v - E_center) : alpha_v >= 0 }

i.e. the rays of per-slot displacements a single text token (or a mixture)
could realize, where E is restricted to TEXT tokens only (image/VQ and special
rows excluded; run 17's nearest-cosine used the full matrix).  E_center is the
full-matrix embedding mean, matching the soft-suffix initialization.

Distance ~ 0   -> a text suffix can in principle reproduce the soft direction
                  (then hardening via hull mixture is on the table).
Distance ~ 1   -> the random-direction null level: hard text is structurally
                  infeasible at this position (T1 disfavored even before
                  projection tricks).

The retry8 Adam route suffix (scripts/14) is measured identically, connecting
the content direction to the route direction and the 1/128 NN projection.

Solvers:
- cone distance: projected-gradient NNLS on the nonneg orthant (convex, exact
  optimum) with an active-set least-squares polish;
- hull distance: Frank-Wolfe on the simplex (sum alpha = 1); the sparse alpha
  support gives the minimal "vocabulary mixture" realization.

Random unit directions provide the null calibration; a single text row provides
a solver sanity check (distance must be ~ 0).
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

feas = importlib.util.spec_from_file_location(
    "p0_backward", str(PROJECT_ROOT / "scripts/P0_backward_feasibility.py"))
p0 = importlib.util.module_from_spec(feas)
feas.loader.exec_module(p0)
spec = importlib.util.spec_from_file_location(
    "ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p12)
spec = importlib.util.spec_from_file_location(
    "distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
d08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d08)

TEACHER_PROMPTS = {
    "red_apple": "Draw a red apple on a plain light background.",
    "golden_retriever": "Draw a golden retriever standing on a plain light background.",
}
STUDENT_PROMPTS = {
    "red_apple": "Explain in one sentence why apples are a popular fruit.",
    "golden_retriever": "Explain in one sentence why golden retrievers are suitable family pets.",
}


def solve_cone(rows, u, iters=3000):
    """min ||rows^T a - u||^2 s.t. a >= 0 (convex NNLS via FISTA).

    rows: (V, d) unit-norm rays; u: (d,) unit direction.  Returns the final
    distance, the support size, and the coefficient sum.  FISTA converges
    O(1/t^2); plain projected gradient is far too slow here because the Gram
    matrix top eigenvalue (anisotropic text embeddings) is ~hundreds.
    """
    import torch
    v_count = rows.shape[0]
    # Lipschitz constant of the gradient map via power iteration on rows @ rows.T.
    probe = torch.randn(v_count, device=rows.device)
    for _ in range(30):
        probe = rows @ (rows.T @ probe)
        probe = probe / probe.norm()
    largest = probe @ (rows @ (rows.T @ probe))
    # Gradient of f(a) = ||rows^T a - u||^2 is 2 rows (rows^T a - u); the
    # Hessian is 2 rows rows^T, so L_f = 2 * largest.  The computed `grad`
    # below is half the true gradient, so a step of lr = 1/(2 * largest)
    # equals a true-gradient step of 1/(2 L_f) -- inside the FISTA safe zone.
    lr = 1.0 / (2.0 * largest + 1e-6)
    a = torch.zeros(v_count, device=rows.device)
    y = a.clone()
    momentum = 1.0
    for _ in range(iters):
        residual = rows.T @ y - u
        grad = rows @ residual
        a_new = torch.clamp(y - lr * grad, min=0)
        momentum_new = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        y = a_new + ((momentum - 1.0) / momentum_new) * (a_new - a)
        a, momentum = a_new, momentum_new
    # Active-set polish on the support.
    loss = (rows.T @ a - u).square().sum()
    support = torch.nonzero(a > 1e-5).flatten()
    if support.numel() > 0 and support.numel() <= rows.shape[1]:
        sub = rows[support].T  # (d, |s|)
        x, _ = torch.linalg.lstsq(sub, u.unsqueeze(1))
        x = x.flatten()
        if x.min() >= 0:
            a_polished = a.clone()
            a_polished[support] = x
            candidate = (rows.T @ a_polished - u).square().sum()
            if candidate < loss:
                a = a_polished
    distance = float((rows.T @ a - u).square().sum().sqrt())
    return distance, int((a > 1e-5).sum()), float(a.sum())


def solve_hull(rows, u, iters=2000, tol=1e-14):
    """min ||rows^T a - u||^2 s.t. a >= 0, sum a = 1 (Frank-Wolfe on simplex)."""
    import torch
    v_count = rows.shape[0]
    a = torch.zeros(v_count, device=rows.device)
    a[0] = 1.0
    residual = rows.T @ a - u
    prev = None
    for _ in range(iters):
        loss = residual.square().sum()
        if prev is not None and abs(float(prev - loss)) < tol * max(1.0, float(prev)):
            break
        prev = loss
        grad = rows @ residual
        vertex = int(grad.argmin())
        direction = torch.zeros(v_count, device=rows.device)
        direction[vertex] = 1.0
        direction -= a
        rows_d = rows.T @ direction
        if rows_d.square().sum() <= 0:
            break
        step = float(-(residual @ rows_d) / rows_d.square().sum())
        step = max(0.0, min(1.0, step))
        a = a + step * direction
        residual = rows.T @ a - u
    distance = float(residual.square().sum().sqrt())
    return distance, int((a > 1e-4).sum())


def rho_suffix_directions(model, token_manager, token_sets, prompt_ids,
                          suffix_length, steps, learning_rate,
                          init_seed, donor_seed):
    """Re-run the run 17 optimization per prompt; return per-slot directions."""
    import torch
    from chameleon.inference import transformer
    text_embeddings = model.tok_embeddings.weight.detach().float()

    def donor_state(prompt_id):
        ids = token_manager.tokens_from_ui([
            {"type": "text", "value": TEACHER_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        torch.manual_seed(donor_seed)
        mask = p12.build_mask(token_sets, token_sets.text_step_support)
        branch = p12.Branch(model, transformer, ids, len(ids) + 256)
        generated = []
        boi_step = None
        for step in range(128):
            history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
            policy = p12.text_policy(branch.logits, history, mask, 1.2, 0.9)
            token = p12.sample_token(policy)
            generated.append(token)
            if token == token_sets.boi_id:
                boi_step = step
                break
            if token == token_sets.eos_id:
                break
            branch.advance(token)
        if boi_step is None:
            raise SystemExit(f"donor did not sample BOI for {prompt_id}")
        capture_ids = ids + generated[:boi_step + 1]
        _, finals = p12.capture_states(
            model, transformer, capture_ids, {"boi": len(capture_ids) - 1}, set())
        return finals["boi"].detach().float(), boi_step

    directions = {}
    for prompt_id in prompt_ids:
        s_donor, boi_step = donor_state(prompt_id)
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": STUDENT_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        torch.manual_seed(init_seed)
        mean_embedding = text_embeddings.mean(dim=0)
        w = (mean_embedding.repeat(suffix_length, 1)
             + 0.01 * torch.randn(suffix_length, mean_embedding.numel(), device="cuda"))
        w = w.detach().requires_grad_(True)
        w_init = w.detach().clone()
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        for step in range(steps):
            embeddings = d08.soft_input_embeddings(
                model, student_ids[:-1], token_sets.eot_id, w,
                continuation=[token_sets.boi_id])
            _, hidden = p0.exact_forward(model, transformer, embeddings, return_hidden=True)
            s_w = hidden[-1].float()
            loss = (s_w - s_donor).square().sum() / s_donor.square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        displacement = w.detach() - w_init
        directions[prompt_id] = torch.nn.functional.normalize(displacement, dim=-1)
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["red_apple", "golden_retriever"])
    parser.add_argument("--extra-suffix-path", default=None,
                        help="additional soft_suffix.pt (retry8 route suffix) to measure")
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--calibration-seeds", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
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

    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    # Text-only embedding rays, centered by the full-matrix mean (the same
    # center the soft suffix is initialized at), row-normalized.
    embeddings = model.tok_embeddings.weight.detach().float()
    center = embeddings.mean(dim=0)
    text_rows = sorted(token_sets.text_token_ids)
    centered = embeddings[torch.tensor(text_rows, device="cuda")] - center
    rays = torch.nn.functional.normalize(centered, dim=-1)
    append({"kind": "setup", "text_vocab_size": rays.shape[0],
            "embedding_dim": rays.shape[1],
            "note": "text-token rows only; centered by full-matrix mean; row-normalized"})

    # Sanity checks: a single text ray and a three-ray mixture are exactly
    # reachable by construction; both solvers must find ~0 distance.
    idx = torch.randint(0, rays.shape[0], (3,))
    sample_ray = rays[int(idx[0])]
    cone_self, _, _ = solve_cone(rays, sample_ray, iters=500)
    hull_self, _ = solve_hull(rays, sample_ray, iters=500)
    mixed = torch.nn.functional.normalize(rays[idx].sum(dim=0), dim=-1)
    cone_mixed, _, _ = solve_cone(rays, mixed, iters=500)
    append({"kind": "solver_sanity", "single_ray_cone_distance": cone_self,
            "single_ray_hull_distance": hull_self,
            "three_ray_mixture_cone_distance": cone_mixed})
    if max(cone_self, hull_self, cone_mixed) > 1e-3:
        raise SystemExit(f"solver failed sanity check: cone {cone_self}, "
                         f"hull {hull_self}, mixture {cone_mixed}")

    directions = rho_suffix_directions(
        model, token_manager, token_sets, args.prompt_ids,
        args.suffix_length, args.steps, args.learning_rate,
        args.init_seed, args.donor_seed,
    )
    for prompt_id, slots in directions.items():
        torch.save({"directions": slots.detach().cpu()}, output / f"rho_directions_{prompt_id}.pt")
        append({"kind": "rho_directions_saved", "prompt_id": prompt_id,
                "slots": slots.shape[0]})

    sources = [("rho_" + pid, directions[pid]) for pid in args.prompt_ids]

    if args.extra_suffix_path:
        state = torch.load(args.extra_suffix_path, map_location="cpu")
        extra = state["suffix"].float().to("cuda")
        torch.manual_seed(args.init_seed)
        mean_embedding = embeddings.mean(dim=0)
        w_init = (mean_embedding.repeat(extra.shape[0], 1)
                  + 0.01 * torch.randn(extra.shape[0], mean_embedding.numel(), device="cuda"))
        sources.append(("retry8_route", torch.nn.functional.normalize(extra - w_init, dim=-1)))
        append({"kind": "extra_suffix_loaded", "slots": extra.shape[0],
                "path": args.extra_suffix_path})

    for seed in range(args.calibration_seeds):
        torch.manual_seed(1000 + seed)
        sources.append((f"random_{seed}",
                        torch.nn.functional.normalize(torch.randn(16, rays.shape[1], device="cuda"), dim=-1)))

    for name, slots in sources:
        cone_distances, hull_distances = [], []
        nearest_cosines = []
        supports, alpha_sums = [], []
        for slot in slots:
            nearest_cosines.append(float((rays @ slot).max()))
            cone_distance, support, alpha_sum = solve_cone(rays, slot)
            cone_distances.append(cone_distance)
            hull_distance, _ = solve_hull(rays, slot)
            hull_distances.append(hull_distance)
            supports.append(support)
            alpha_sums.append(alpha_sum)
        append({
            "kind": "hull_distance_summary", "source": name,
            "nearest_text_cosine_mean": sum(nearest_cosines) / len(nearest_cosines),
            "nearest_text_cosine_min": min(nearest_cosines),
            "cone_distance_mean": sum(cone_distances) / len(cone_distances),
            "cone_distance_min": min(cone_distances),
            "cone_distance_max": max(cone_distances),
            "hull_distance_mean": sum(hull_distances) / len(hull_distances),
            "hull_distance_min": min(hull_distances),
            "cone_support_mean": sum(supports) / len(supports),
            "cone_alpha_sum_mean": sum(alpha_sums) / len(alpha_sums),
            "per_slot_cone": cone_distances,
            "per_slot_hull": hull_distances,
            "per_slot_nn_cosine": nearest_cosines,
        })
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
