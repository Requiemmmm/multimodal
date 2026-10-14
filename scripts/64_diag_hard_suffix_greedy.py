#!/usr/bin/env python
"""Hard-suffix diagnostic: greedy coordinate search over per-slot top-k
projection candidates, scored by the exact-forward route NLL (the M-step
objective).  Answers the question: does ANY hard realization of the 64-slot
suffix carry route signal, or is the NN projection already optimal?

Not an attack evaluation: the surviving suffix must still be re-evaluated with
the official decoder (script 10 protocol).
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
import importlib.util
import json
import sys
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--projection", required=True,
                        help="script 09 output with topk_token_ids")
    parser.add_argument("--elites", required=True,
                        help="selected_elites.json of the soft suffix's training run")
    parser.add_argument("--prompt-id", default="sky_blue")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--init-seed", type=int, default=0)
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
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompt_text = p22.STUDENT_PROMPTS[args.prompt_id]
    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": prompt_text},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    projection = json.loads(Path(args.projection).read_text())
    candidates = projection["topk_token_ids"]
    suffix_length = len(candidates)
    elites = json.loads(Path(args.elites).read_text())
    target_lists = [elite["target_ids"] for elite in elites]
    support = d08.legal_mask(token_sets)

    def route_nll(ids):
        suffix = model.tok_embeddings(
            torch.tensor(ids, dtype=torch.long, device="cuda")).detach().float()
        losses = [
            p14.differentiable_route_nll(
                model, transformer_module, token_sets,
                prompt_ids, suffix, target_ids, support)
            for target_ids in target_lists
        ]
        return float(torch.stack(losses).mean().detach())

    import chameleon.inference.transformer as transformer_module

    current = [int(projection["token_ids"][slot]) for slot in range(suffix_length)]
    initial_nll = route_nll(current)
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    append({"kind": "greedy_hard_route_init", "route_nll": initial_nll,
            "suffix_length": suffix_length, "n_elites": len(target_lists)})
    t0 = time.time()
    for passage in range(args.passes):
        improved_slots = []
        for slot in range(suffix_length):
            best_value, best_id = route_nll(current), current[slot]
            for candidate in candidates[slot]:
                if candidate == current[slot]:
                    continue
                trial = current.copy()
                trial[slot] = candidate
                value = route_nll(trial)
                if value < best_value:
                    best_value, best_id = value, candidate
            if best_id != current[slot]:
                current[slot] = best_id
                improved_slots.append(slot)
                append({"kind": "greedy_hard_route_step", "pass": passage,
                        "slot": slot, "token": best_id,
                        "route_nll": best_value,
                        "seconds": round(time.time() - t0, 1)})
    final_nll = route_nll(current)
    append({"kind": "greedy_hard_route_done", "route_nll": final_nll,
            "initial_route_nll": initial_nll,
            "delta": round(initial_nll - final_nll, 4),
            "seconds": round(time.time() - t0, 1)})
    result = {
        "kind": "soft_to_hard_projection",
        "projection": "nearest_text_embedding + greedy route-NLL rerank",
        "source_soft_state": projection["source_soft_state"],
        "token_ids": current,
        "route_nll_initial": initial_nll,
        "route_nll_final": final_nll,
        "note": "Reranked hard suffix; re-evaluate with script 10 before any claim.",
    }
    (output / "reranked_projection.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
