#!/usr/bin/env python
"""Exposure-bias diagnostic for the hard-suffix content boundary.

The joint search scores the content term on a *teacher-forced* prefix: it asks
whether the apple trajectory's token k is favoured given the apple trajectory's
own tokens 0..k-1.  The deployed rollout never sees that prefix -- it sees the
tokens the model itself sampled.  If the model's own prefix drifts off the
teacher trajectory, the teacher-forced margin is measuring a context the attack
never visits, and a converging search can leave behaviour untouched.

This script reads the native rollouts already produced by script 10 (no new
sampling) and, for every completed image, compares the per-position
target-vs-victim margin under two contexts:

  teacher : [BOI] + teacher_target[:k]
  sampled : [BOI] + the model's own visual tokens[:k]

It also records how often the model's own token at position k *is* the teacher's
target token.  A margin that is large under ``teacher`` and collapses under
``sampled`` is exposure bias made quantitative -- the visual analogue of the
margin-vs-share gap that explained the route ceiling.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import json
import sys
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


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))
mho = _load("morph_hard_objectives",
            str(PROJECT_ROOT / "scripts/morph_hard_objectives.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--eval-dir", required=True,
                        help="script-10 output dir with per-seed generated_tokens.json")
    parser.add_argument("--projection", required=True,
                        help="the suffix that produced those rollouts")
    parser.add_argument("--teachers-json", required=True)
    parser.add_argument("--prompt-id", default="strawberry")
    parser.add_argument("--content-steps", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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

    K = args.content_steps
    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": p22.STUDENT_PROMPTS[args.prompt_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    suffix_ids = [int(t) for t in
                  json.loads(Path(args.projection).read_text())["token_ids"]]
    teachers = json.loads(Path(args.teachers_json).read_text())
    target_visual = list(teachers["target_visual"])
    victim_visual = list(teachers["victim_visual"])
    support = d08.legal_mask(token_sets)

    space = mho.HardObjectiveSpace(model, transformer_module, token_sets,
                                   prompt_ids, support)
    boi = token_sets.boi_id
    eoi = token_sets.eoi_id

    # Suffix-independent caches: one per context, reused across rollouts.
    teacher_cache = mho.CFGBranchCache(model, transformer_module, token_sets,
                                       target_visual[:K])
    K_teacher = min(K, len(target_visual), len(victim_visual))

    eval_dir = Path(args.eval_dir)
    rows = []
    skipped = {"no_boi": 0, "short_visual": 0}
    for sample_dir in sorted(eval_dir.glob("*_seed*")):
        tokens_path = sample_dir / "generated_tokens.json"
        if not tokens_path.exists():
            continue
        raw = json.loads(tokens_path.read_text())
        if boi not in raw:
            skipped["no_boi"] += 1
            continue
        start = raw.index(boi) + 1
        end = start + 1024
        visual = raw[start:end]
        if eoi in visual:
            visual = visual[:visual.index(eoi)]
        if len(visual) < K_teacher:
            skipped["short_visual"] += 1
            continue

        # teacher-forced context
        cont_t = [boi] + target_visual[:K_teacher - 1]
        _, deltas_t = space.content(suffix_ids, target_visual[:K_teacher],
                                    victim_visual[:K_teacher],
                                    cache=teacher_cache, mode="margin",
                                    continuation=cont_t)
        # the model's own sampled context
        cont_s = [boi] + visual[:K_teacher - 1]
        # The cache needs one row per scored position, so its prefix must be at
        # least K long -- pass the sampled visual tokens themselves (the branch
        # for position k is conditioned on visual[:k]).
        sampled_cache = mho.CFGBranchCache(model, transformer_module, token_sets,
                                           visual[:K_teacher])
        _, deltas_s = space.content(suffix_ids, target_visual[:K_teacher],
                                    victim_visual[:K_teacher],
                                    cache=sampled_cache, mode="margin",
                                    continuation=cont_s)
        rows.append({
            "sample": sample_dir.name,
            "teacher_margins": [round(float(d), 4) for d in deltas_t.detach()],
            "sampled_margins": [round(float(d), 4) for d in deltas_s.detach()],
            "agreement": [int(visual[k] == target_visual[k])
                          for k in range(K_teacher)],
            "sampled_tokens": [int(t) for t in visual[:K_teacher]],
        })
        print(json.dumps({"kind": "prefix_divergence", **rows[-1]}), flush=True)

    n = len(rows)
    summary = {"kind": "prefix_divergence_summary", "n_rollouts": n,
               "content_steps": K_teacher, "skipped": skipped,
               "n_target_trajectories": 1}
    if n:
        for label, key in (("teacher", "teacher_margins"),
                           ("sampled", "sampled_margins")):
            per_k = [sum(r[key][k] for r in rows) / n for k in range(K_teacher)]
            summary[f"mean_margin_{label}_by_step"] = [round(v, 4) for v in per_k]
            summary[f"mean_margin_{label}"] = round(sum(per_k) / K_teacher, 4)
        agree_by_k = [sum(r["agreement"][k] for r in rows) / n
                      for k in range(K_teacher)]
        summary["agreement_by_step"] = [round(v, 4) for v in agree_by_k]
        summary["mean_agreement"] = round(sum(agree_by_k) / K_teacher, 4)
        # How often does the sampled prefix still favour the target token?
        fav = [sum(1 for r in rows if r["sampled_margins"][k] > 0) / n
               for k in range(K_teacher)]
        summary["sampled_favours_target_by_step"] = [round(v, 4) for v in fav]
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
