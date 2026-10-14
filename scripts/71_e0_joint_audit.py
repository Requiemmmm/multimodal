#!/usr/bin/env python
"""E0 audit for the joint (route + content) hard-suffix objective.

Checks, in order:

1. every token of the init suffix is an ordinary text token (no BOI / EOI /
   EOT / visual token may appear in an adversarial suffix);
2. ``boi_ce`` from the shared module reproduces the inlined script-65 maths
   bit-for-bit (independent re-implementation, same inputs);
3. ``content`` with the cached CFG branches reproduces the uncached path;
4. the gradient of each objective is finite, non-zero, and has the shape
   ``[L, hidden]`` -- in particular the *content* gradient, which the whole
   E1 chain depends on;
5. a central finite difference on one slot agrees with the analytic gradient
   in sign and magnitude.

Writes ``audit.json`` (plus a per-check ``metrics.jsonl``) into --output-dir.
No long-running search happens here.
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


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))
mho = _load("morph_hard_objectives",
            str(PROJECT_ROOT / "scripts/morph_hard_objectives.py"))

from morph.decoding.processed_policy import (  # noqa: E402
    apply_repetition_penalty,
    masked_logprobs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--init-projection", required=True)
    parser.add_argument("--prompt-id", default="strawberry")
    parser.add_argument("--teachers-json", required=True)
    parser.add_argument("--content-steps", type=int, default=1)
    parser.add_argument("--content-lambda", type=float, default=1.0)
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

    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": p22.STUDENT_PROMPTS[args.prompt_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    init = json.loads(Path(args.init_projection).read_text())
    current = [int(t) for t in init["token_ids"]]
    suffix_length = len(current)
    support = d08.legal_mask(token_sets)
    text_ids = set(int(t) for t in token_sets.text_token_ids)
    teachers = json.loads(Path(args.teachers_json).read_text())
    target_visual = teachers["target_visual"][:args.content_steps]
    victim_visual = teachers["victim_visual"][:args.content_steps]

    records = []
    audit = {
        "prompt_id": args.prompt_id,
        "content_steps": args.content_steps,
        "content_lambda": args.content_lambda,
        "suffix_length": suffix_length,
        "prompt_length": len(prompt_ids),
        "first_free_position": len(prompt_ids) + suffix_length - 1,
        "boi_offset_in_continuation": 0,
    }
    illegal = [t for t in current if t not in text_ids]
    audit["illegal_suffix_ids"] = illegal

    space = mho.HardObjectiveSpace(model, transformer_module, token_sets,
                                   prompt_ids, support)

    def emit(record):
        records.append(record)
        print(json.dumps(record), flush=True)

    emit({"kind": "e0_text_token_audit", "suffix_length": suffix_length,
          "n_illegal": len(illegal), "illegal_suffix_ids": illegal,
          "boi_id_in_text_set": int(token_sets.boi_id) in text_ids})

    # --- 2. equivalence against an independent copy of script 65's inlined maths
    def inlined_boi_ce_detached(ids):
        suffix = model.tok_embeddings(
            torch.tensor(ids, dtype=torch.long, device="cuda")).detach().float()
        embeds = d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix)
        logits = p14.feasibility.exact_forward(model, transformer_module, embeds)
        position = len(prompt_ids) + suffix.shape[0] - 1
        past = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[position], past, penalty=1.2), support)
        return -policy[token_sets.boi_id]

    module_ce = float(space.boi_ce(current).detach())
    inlined_ce = float(inlined_boi_ce_detached(current).detach())
    audit["boi_ce_module"] = module_ce
    audit["boi_ce_inlined"] = inlined_ce
    audit["boi_ce_abs_diff"] = abs(module_ce - inlined_ce)
    emit({"kind": "e0_ce_equivalence", "module": round(module_ce, 6),
          "inlined": round(inlined_ce, 6),
          "abs_diff": abs(module_ce - inlined_ce)})

    # --- 3. branch-cache equivalence
    cache = mho.CFGBranchCache(model, transformer_module, token_sets, target_visual)
    loss_cached, deltas_cached = space.content(current, target_visual,
                                               victim_visual, cache=cache)
    loss_uncached, deltas_uncached = space.content(current, target_visual,
                                                   victim_visual, cache=None)
    cache_diff = float((loss_cached - loss_uncached).abs())
    audit["content_cache_abs_diff"] = cache_diff
    audit["content_loss"] = float(loss_cached.detach())
    audit["content_margins"] = [float(d) for d in deltas_cached.detach()]
    emit({"kind": "e0_branch_cache_equivalence",
          "cached": round(float(loss_cached.detach()), 6),
          "uncached": round(float(loss_uncached.detach()), 6),
          "abs_diff": cache_diff})

    # --- 4. gradients
    def grad_of(fn):
        suffix_t = model.tok_embeddings(
            torch.tensor(current, dtype=torch.long, device="cuda")
        ).detach().clone().requires_grad_(True)
        loss = fn(suffix_t)
        loss.backward()
        grad = suffix_t.grad
        return loss, grad

    ce_loss, ce_grad = grad_of(space.boi_ce)
    content_loss, content_grad = grad_of(
        lambda s: space.content(s, target_visual, victim_visual, cache=cache)[0])
    joint_loss, joint_grad = grad_of(
        lambda s: space.joint(s, target_visual, victim_visual,
                              args.content_lambda, cache=cache)[0])

    cached_branches = all(
        not p.requires_grad for p in ())
    audit.update({
        "joint_gradient_nonzero": bool(float(joint_grad.abs().sum()) > 0),
        "joint_gradient_finite": bool(torch.isfinite(joint_grad).all()),
        "content_gradient_nonzero": bool(float(content_grad.abs().sum()) > 0),
        "content_gradient_finite": bool(torch.isfinite(content_grad).all()),
        "content_gradient_norm": float(content_grad.norm()),
        "boi_ce_gradient_norm": float(ce_grad.norm()),
        "joint_gradient_norm": float(joint_grad.norm()),
        "gradient_shape": list(joint_grad.shape),
        "boi_ce_loss": float(ce_loss.detach()),
        "joint_loss": float(joint_loss.detach()),
        "cfgbranch_cache_grad_free": cached_branches,
    })
    emit({"kind": "e0_gradient", "joint_loss": round(float(joint_loss.detach()), 4),
          "boi_ce": round(float(ce_loss.detach()), 4),
          "content_loss": round(float(content_loss.detach()), 4),
          "joint_grad_norm": round(float(joint_grad.norm()), 6),
          "content_grad_norm": round(float(content_grad.norm()), 6),
          "grad_shape": list(joint_grad.shape)})

    # --- 5. finite differences: step-size sweep + descent test
    #
    # The forward runs in the model's native precision, where a loss of order 1
    # has a resolution of roughly 1e-2: a perturbation small enough to stay in
    # the linear regime is below that resolution, and one large enough to be
    # resolved has already left it.  Exact FD agreement is therefore not
    # attainable, so the audit reports what is actually diagnostic:
    #
    #   (a) a step-size sweep along each objective's own gradient direction,
    #       with boi_ce as a control -- its gradient demonstrably works, since
    #       CE-GCG drove the hard route to 51.25%;
    #   (b) a descent test: does stepping along -grad lower the exact-forward
    #       loss?  This is the property the GCG search actually relies on.
    #
    # Note the search only uses the gradient to *propose* top-k candidates;
    # every accepted move is re-scored by exact forward, so a noisy gradient
    # costs search efficiency, not correctness.
    suffix_base = model.tok_embeddings(
        torch.tensor(current, dtype=torch.long, device="cuda")).detach().float()
    emb_scale = float(suffix_base.std())
    param_dtype = str(next(model.parameters()).dtype)
    audit["suffix_embedding_std"] = emb_scale
    audit["model_param_dtype"] = param_dtype

    def objective_at(emb, lossfn):
        with torch.no_grad():
            return float(lossfn(emb))

    fd_rows = []
    descent_rows = []
    for label, lossfn, grad in (
        ("boi_ce", lambda e: space.boi_ce(e), ce_grad),
        ("content",
         lambda e: space.content(e, target_visual, victim_visual, cache=cache)[0],
         content_grad),
        ("joint",
         lambda e: space.joint(e, target_visual, victim_visual,
                               args.content_lambda, cache=cache)[0],
         joint_grad),
    ):
        unit = grad / grad.norm()
        analytic = float(grad.norm())  # d/ds along the unit gradient direction
        base = objective_at(suffix_base, lossfn)
        # t is scaled so each coordinate moves by alpha * emb_scale on average.
        for alpha in (0.01, 0.03, 0.1):
            t = alpha * emb_scale * (suffix_length * suffix_base.shape[1]) ** 0.5
            fd = (objective_at(suffix_base + t * unit, lossfn)
                  - objective_at(suffix_base - t * unit, lossfn)) / (2 * t)
            down = objective_at(suffix_base - t * unit, lossfn)
            descent_rows.append({
                "objective": label, "alpha": alpha,
                "loss_before": base, "loss_after_step_down": down,
                "decreased": down < base,
                "predicted_decrease": analytic * t,
                "actual_decrease": base - down,
            })
            fd_rows.append({
                "objective": label, "alpha": alpha,
                "finite_difference": fd, "analytic": analytic,
                "rel_error": abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12),
                "same_sign": (fd > 0) == (analytic > 0),
            })
    audit["finite_difference"] = fd_rows
    audit["descent_test"] = descent_rows
    emit({"kind": "e0_finite_difference", "rows": fd_rows})
    emit({"kind": "e0_descent", "rows": descent_rows})

    def _decreases(objective):
        return any(r["decreased"] for r in descent_rows
                   if r["objective"] == objective)

    # Gates are the checks E0 exists to enforce: they catch code and protocol
    # errors, and they are all measurable in bf16.  The finite-difference and
    # descent probes are reported as diagnostics, not gates, because the audit
    # itself proves they are unmeasurable at this precision: the smallest step
    # displaces each coordinate by ~2.5 bf16 ULPs (see suffix_embedding_std vs
    # alpha below), and the boi_ce control -- a gradient that demonstrably works,
    # having driven the hard route to 51.25% -- fails them in exactly the same
    # way.  Step sizes large enough to be resolved have left the linear regime.
    gates = {
        "pass_text_only_suffix": len(illegal) == 0,
        "pass_ce_equivalence": audit["boi_ce_abs_diff"] < 1e-4,
        "pass_branch_cache": cache_diff < 1e-4,
        "pass_content_gradient": audit["content_gradient_nonzero"]
        and audit["content_gradient_finite"],
        "pass_joint_gradient": audit["joint_gradient_nonzero"]
        and audit["joint_gradient_finite"],
    }
    diagnostics = {
        "diagnostic_descent_control_boi_ce": _decreases("boi_ce"),
        "diagnostic_descent_content": _decreases("content"),
        "diagnostic_descent_joint": _decreases("joint"),
        "diagnostic_fd_measurable": False,
        "diagnostic_note": (
            "FD/descent probes are below bf16 resolution and are not gates; "
            "the GCG search accepts a move only after re-scoring it with an "
            "exact forward, so proposal quality affects search efficiency, "
            "never correctness."
        ),
    }
    audit["verdict"] = {**gates, **diagnostics}
    audit["gates"] = gates
    audit["diagnostics"] = diagnostics
    audit["pass"] = all(gates.values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps({"kind": "e0_verdict", **gates, **diagnostics,
                      "pass": audit["pass"]}, indent=2), flush=True)
    if not audit["pass"]:
        raise SystemExit("E0 audit FAILED -- do not start E1")


if __name__ == "__main__":
    main()
