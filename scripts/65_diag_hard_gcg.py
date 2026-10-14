#!/usr/bin/env python
"""Hard-suffix diagnostic: GCG-style coordinate search over the full text
vocabulary.  Per iteration: exact-forward route NLL -> analytic gradient at
each suffix position -> top-k vocab candidates by gradient score -> evaluate
each candidate by swap + exact forward, keep the best.

Seeded from a prior hard suffix (e.g., the greedy rerank of script 64).  The
surviving suffix must still be re-evaluated with the official decoder
(script 10 protocol); this script only optimizes the route-NLL objective.
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

from morph.suffix.constraints import route_floor_from_optional


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))
mho = _load("morph_hard_objectives",
            str(PROJECT_ROOT / "scripts/morph_hard_objectives.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--init-projection", required=True,
                        help="starting hard suffix (projection JSON with token_ids)")
    parser.add_argument("--elites", default=None,
                        help="selected_elites.json of the soft suffix's training run "
                             "(required for --objective preamble-nll)")
    parser.add_argument("--prompt-id", default="sky_blue")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--objective", choices=["preamble-nll", "boi-margin", "boi-ce", "joint"],
                        default="preamble-nll",
                        help="boi-margin = processed BOI logit margin at the first free "
                             "generation position (after prompt+suffix+EOT); "
                             "boi-ce = processed BOI cross-entropy (concentrates "
                             "sampling mass on BOI, the DeGCG first-target loss); "
                             "joint = boi-ce + lambda * forced-BOI visual contrast "
                             "(target-vs-victim teacher visual tokens, CFG policy) "
                             "-- hard-space route+content search")
    parser.add_argument("--teachers-json", default=None,
                        help="teachers.json with target_visual/victim_visual "
                             "(required for --objective joint)")
    parser.add_argument("--content-steps", type=int, default=1,
                        help="first K visual tokens in the joint content term")
    parser.add_argument("--content-lambda", type=float, default=1.0)
    parser.add_argument("--route-floor", type=float, default=None,
                        help="maximum processed BOI cross-entropy allowed for a "
                             "candidate (lower is better).  Candidates above "
                             "this floor are rejected during joint/content "
                             "search; use this to prevent content optimization "
                             "from silently destroying modality routing.")
    parser.add_argument("--content-objective", choices=["margin", "ce", "distill"],
                        default="margin",
                        help="margin = target-vs-victim log-prob margin (the "
                             "original joint term); ce = target log-prob only, "
                             "i.e. concentrate mass on the target visual tokens "
                             "-- the visual analogue of the boi-margin -> boi-ce "
                             "switch that took the hard route from 0/16 to 51.25%%; "
                             "distill = soft cross-entropy against the model's "
                             "own apple-drawing policy (--apple-policy-json), "
                             "which matches a distribution instead of naming a "
                             "token")
    parser.add_argument("--apple-policy-json", default=None,
                        help="JSON from script 75 with the model's own "
                             "apple-drawing image policy per visual position; "
                             "required for --content-objective distill")
    parser.add_argument("--sampled-prefix-json", default=None,
                        help="JSON from script 73 with model-sampled visual "
                             "prefixes.  Scores the content term against the "
                             "prefixes the deployed rollout actually visits "
                             "instead of the teacher's own trajectory (the "
                             "exposure-bias fix); the target/victim comparison "
                             "tokens stay the teacher's.")
    parser.add_argument("--max-slot", type=int, default=None,
                        help="only search the first N suffix slots (route/content "
                             "signal lives in the early slots per mixq_16)")
    args = parser.parse_args()
    if args.route_floor is not None and args.route_floor < 0:
        raise SystemExit("--route-floor must be non-negative")
    if args.route_floor is not None and args.objective == "preamble-nll":
        raise SystemExit("--route-floor requires a BOI-aware objective, not "
                         "--objective preamble-nll")
    route_floor = route_floor_from_optional(args.route_floor)
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
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    import chameleon.inference.transformer as transformer_module

    prompt_text = p22.STUDENT_PROMPTS[args.prompt_id]
    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": prompt_text},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    init = json.loads(Path(args.init_projection).read_text())
    current = [int(t) for t in init["token_ids"]]
    suffix_length = len(current)
    support = d08.legal_mask(token_sets)
    vocab = model.tok_embeddings.weight.detach().float()
    text_ids = sorted(token_sets.text_token_ids)
    allowed = torch.zeros(vocab.shape[0], dtype=torch.bool, device="cuda")
    allowed[torch.tensor(text_ids, device="cuda")] = True

    from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs

    if args.objective == "preamble-nll":
        if not args.elites:
            raise SystemExit("--elites required for --objective preamble-nll")
        target_lists = [elite["target_ids"]
                        for elite in json.loads(Path(args.elites).read_text())]
    else:
        target_lists = []

    def route_nll_detached(ids):
        """Route NLL of a hard suffix (token ids), no grad graph retained."""
        suffix = model.tok_embeddings(
            torch.tensor(ids, dtype=torch.long, device="cuda")).detach().float()
        losses = [
            p14.differentiable_route_nll(
                model, transformer_module, token_sets,
                prompt_ids, suffix, target_ids, support)
            for target_ids in target_lists
        ]
        return torch.stack(losses).mean()

    def route_nll_live(suffix_t):
        """Route NLL with a live graph to the suffix embeddings tensor."""
        losses = [
            p14.differentiable_route_nll(
                model, transformer_module, token_sets,
                prompt_ids, suffix_t, target_ids, support)
            for target_ids in target_lists
        ]
        return torch.stack(losses).mean()

    def boi_margin_detached(ids):
        """Processed BOI logit margin at the first free generation position."""
        suffix = model.tok_embeddings(
            torch.tensor(ids, dtype=torch.long, device="cuda")).detach().float()
        embeds = d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix)
        logits = p14.feasibility.exact_forward(model, transformer_module, embeds)
        position = len(prompt_ids) + suffix.shape[0] - 1
        past = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[position], past, penalty=1.2), support)
        return -(policy[token_sets.boi_id] - policy.max())

    def boi_margin_live(suffix_t):
        embeds = d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix_t)
        logits = p14.feasibility.exact_forward(model, transformer_module, embeds)
        position = len(prompt_ids) + suffix_t.shape[0] - 1
        past = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[position], past, penalty=1.2), support)
        return -(policy[token_sets.boi_id] - policy.max())

    def boi_ce_detached(ids):
        """Processed cross-entropy of BOI at the first free position."""
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

    def boi_ce_live(suffix_t):
        embeds = d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix_t)
        logits = p14.feasibility.exact_forward(model, transformer_module, embeds)
        position = len(prompt_ids) + suffix_t.shape[0] - 1
        past = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[position], past, penalty=1.2), support)
        return -policy[token_sets.boi_id]

    joint_parts = None
    if args.objective == "boi-margin":
        loss_detached, loss_live = boi_margin_detached, boi_margin_live
    elif args.objective == "boi-ce":
        loss_detached, loss_live = boi_ce_detached, boi_ce_live
    elif args.objective == "joint":
        if not args.teachers_json:
            raise SystemExit("--teachers-json required for --objective joint")
        teachers = json.loads(Path(args.teachers_json).read_text())
        target_visual = teachers["target_visual"][:args.content_steps]
        victim_visual = teachers["victim_visual"][:args.content_steps]
        # Objectives live in morph_hard_objectives so the E0 audit (script 71)
        # and this search cannot drift apart.  The CFG branches depend only on
        # the target visual prefix, so they are built once and reused for every
        # candidate instead of twice per candidate.
        space = mho.HardObjectiveSpace(model, transformer_module, token_sets,
                                       prompt_ids, support)
        # One (continuation, CFG cache) pair per scored context.  Default is the
        # teacher's own trajectory; --sampled-prefix-json swaps in the prefixes
        # the deployed rollout actually visits, which is the exposure-bias fix
        # the divergence diagnostic (script 72) showed is needed.
        if args.sampled_prefix_json:
            payload = json.loads(Path(args.sampled_prefix_json).read_text())
            prefixes = [[int(t) for t in p[:args.content_steps]]
                        for p in payload["prefixes"]]
            prefixes = [p for p in prefixes
                        if len(p) >= args.content_steps]
            if not prefixes:
                raise SystemExit("--sampled-prefix-json has no prefix long enough "
                                 f"for content-steps={args.content_steps}")
            contexts = [[token_sets.boi_id] + p[:args.content_steps - 1]
                        for p in prefixes]
            caches = [mho.CFGBranchCache(model, transformer_module, token_sets,
                                         p[:args.content_steps])
                      for p in prefixes]
        else:
            contexts = [None]
            caches = [mho.CFGBranchCache(model, transformer_module, token_sets,
                                         target_visual)]

        # Distributional target for --content-objective distill: scatter the
        # truncated per-position top-N mass back into a dense [K, V] table.
        apple_policy = None
        apple_policy_mass = None
        if args.content_objective == "distill":
            if not args.apple_policy_json:
                raise SystemExit("--apple-policy-json required for "
                                 "--content-objective distill")
            payload = json.loads(Path(args.apple_policy_json).read_text())
            head = args.content_steps
            if len(payload["target_top_ids"]) < head:
                raise SystemExit("apple policy JSON does not cover "
                                 f"content-steps={head}")
            vocab_size = int(support.shape[0])
            ids = torch.tensor(payload["target_top_ids"][:head],
                               dtype=torch.long, device="cuda")
            probs = torch.tensor(payload["target_top_probs"][:head],
                                 dtype=torch.float32, device="cuda")
            apple_policy = torch.zeros(head, vocab_size, device="cuda")
            apple_policy.scatter_(1, ids, probs)
            apple_policy_mass = [round(float(m), 6) for m in
                                 payload["mass_kept_by_position"][:head]]

        def joint_parts(ids):
            """(loss, breakdown) -- breakdown is what gets logged."""
            ce = space.boi_ce(ids)
            losses, deltas_all = [], []
            for context, cache in zip(contexts, caches):
                loss_k, deltas_k = space.content(
                    ids, target_visual, victim_visual, cache=cache,
                    mode=args.content_objective, continuation=context,
                    target_policy=apple_policy)
                losses.append(loss_k)
                deltas_all.append(deltas_k)
            content_loss = torch.stack(losses).mean()
            return ce + args.content_lambda * content_loss, {
                "boi_ce": ce,
                "content_loss": content_loss,
                "content_margins": torch.stack(deltas_all).mean(0),
                "n_prefixes": len(contexts),
            }

        def joint_detached(ids):
            return joint_parts(ids)[0]

        def joint_live(suffix_t):
            return joint_parts(suffix_t)[0]

        loss_detached, loss_live = joint_detached, joint_live
    else:
        loss_detached, loss_live = route_nll_detached, route_nll_live

    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    def breakdown_fields():
        """Route/content split of the joint loss, or {} for the other objectives."""
        if joint_parts is None:
            return {}
        with torch.no_grad():
            _, parts = joint_parts(current)
        return {
            "boi_ce": round(float(parts["boi_ce"].detach()), 4),
            "content_loss": round(float(parts["content_loss"].detach()), 4),
            "content_margins": [round(float(d), 4)
                                for d in parts["content_margins"].detach()],
            "content_steps": len(target_visual),
            "content_lambda": args.content_lambda,
            "content_objective": args.content_objective,
            "n_prefixes": parts.get("n_prefixes", 1),
        }

    def route_floor_value(ids):
        """Return the exact processed BOI CE used by the route constraint."""
        if args.route_floor is None:
            return None
        # Reuse the same detached exact-forward route objective used by the
        # BOI-CE search.  This deliberately re-scores each candidate after the
        # proposed token replacement; gradient scores are never treated as a
        # feasibility certificate.
        return float(boi_ce_detached(ids).detach())

    t0 = time.time()
    start_nll = float(loss_detached(current).detach())
    append({"kind": "gcg_hard_init", "loss": round(start_nll, 4),
            "suffix_length": suffix_length, "n_elites": len(target_lists),
            "top_k": args.top_k, "iterations": args.iterations,
            "objective": args.objective,
            "route_floor": args.route_floor,
            "route_floor_value": route_floor_value(current),
            **breakdown_fields()})

    for iteration in range(args.iterations):
        # Gradient ranking of the full text vocabulary at every position.
        suffix_t = model.tok_embeddings(
            torch.tensor(current, dtype=torch.long, device="cuda")
        ).detach().clone().requires_grad_(True)
        loss = loss_live(suffix_t)
        loss.backward()
        grad = suffix_t.grad.float()  # [L, 4096]
        scores = grad @ vocab.T  # [L, V]
        scores[:, ~allowed] = float("-inf")
        top_scores, top_ids = torch.topk(scores, k=args.top_k, dim=-1)
        gradient_row = {"kind": "gcg_hard_gradient", "iteration": iteration,
                        "loss": round(float(loss.detach()), 4),
                        "seconds": round(time.time() - t0, 1)}
        # Route/content split: answers whether BOI-CE swamps the content term
        # (the E1 question) instead of only logging their sum.
        gradient_row.update(breakdown_fields())
        append(gradient_row)
        improved = 0
        rejected_route_floor = 0
        search_slots = min(suffix_length, args.max_slot or suffix_length)
        # Hoist baseline loss out of slot loop: it's invariant within the slot.
        baseline_value = loss_detached(current)
        for slot in range(search_slots):
            best_value, best_id = baseline_value, current[slot]
            for candidate in top_ids[slot].tolist():
                if candidate == current[slot]:
                    continue
                trial = current.copy()
                trial[slot] = candidate
                # Check loss first: only evaluate route constraint if this
                # candidate would actually become the new optimum.  Avoids
                # O(top_k) extra forward passes per slot.
                value = loss_detached(trial)
                if value < best_value:
                    if args.route_floor is not None:
                        candidate_route = route_floor_value(trial)
                        if not route_floor.accepts(candidate_route):
                            rejected_route_floor += 1
                            continue
                    best_value, best_id = value, candidate
            if best_id != current[slot]:
                current[slot] = best_id
                baseline_value = best_value  # Update for next slot
                improved += 1
                append({"kind": "gcg_hard_step", "iteration": iteration,
                        "slot": slot, "token": best_id,
                        "loss": round(float(best_value.detach()), 4),
                        "route_floor_value": route_floor_value(current),
                        "seconds": round(time.time() - t0, 1)})
        if improved == 0:
            append({"kind": "gcg_hard_converged", "iteration": iteration,
                    "loss": round(float(loss_detached(current).detach()), 4),
                    "route_floor_rejections": rejected_route_floor,
                    "seconds": round(time.time() - t0, 1)})
            break
        append({"kind": "gcg_hard_iteration_done", "iteration": iteration,
                "route_floor_rejections": rejected_route_floor,
                "route_floor_value": route_floor_value(current),
                "seconds": round(time.time() - t0, 1)})
    final_nll = float(loss_detached(current).detach())
    append({"kind": "gcg_hard_done", "loss": round(final_nll, 4),
            "start_loss": round(start_nll, 4), **breakdown_fields(),
            "route_floor": args.route_floor,
            "route_floor_value": route_floor_value(current),
            "delta": round(start_nll - final_nll, 4),
            "seconds": round(time.time() - t0, 1)})
    result = {
        "kind": "soft_to_hard_projection",
        "projection": "nearest_text_embedding + greedy rerank + GCG gradient search",
        "source_soft_state": init.get("source_soft_state"),
        "source_projection": args.init_projection,
        "objective": args.objective,
        "prompt_id": args.prompt_id,
        "top_k": args.top_k,
        "iterations": args.iterations,
        "max_slot": args.max_slot,
        "teachers_json": args.teachers_json,
        "content_steps": args.content_steps if args.objective == "joint" else None,
        "content_lambda": args.content_lambda if args.objective == "joint" else None,
        "content_objective": (args.content_objective
                              if args.objective == "joint" else None),
        "route_floor": args.route_floor,
        "sampled_prefix_json": (args.sampled_prefix_json
                                if args.objective == "joint" else None),
        "token_ids": current,
        "loss_start": start_nll,
        "loss_final": final_nll,
        "note": "GCG-reranked hard suffix; re-evaluate with script 10 before any claim.",
    }
    (output / "gcg_projection.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
