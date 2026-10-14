#!/usr/bin/env python
"""Multi-prompt state injection M-step (run 34) -- the content-hijacking test.

The mismatch experiment (run 31) showed a state-matched suffix trained ONLY on
the apple prompt cannot inject apple content into a non-apple prompt: the
suffix stores a prompt-conditional residual.  The untested mechanism for
ATTACKER-SPECIFIED content is prompt-INDEPENDENT injection: train one suffix to
push the BOI-position final residual toward the apple donor state from MANY
non-apple prompts, so the apple plan lands regardless of context (P1: the BOI
state carries the plan; the rho probe: the suffix has high leverage on it).

Training prompts: golden_retriever / yellow_school_bus / blue_ceramic_cup /
green_frog (no apple concept).  Evaluation (separate run): sky_blue -- an
unseen prompt with no apple concept.  If the suffix produces apple images
there, content hijacking holds at the soft layer.
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

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p0 = _load("p0_backward", str(PROJECT_ROOT / "scripts/P0_backward_feasibility.py"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))

TRAIN_PROMPTS = ["golden_retriever", "yellow_school_bus", "blue_ceramic_cup", "green_frog"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--start-suffix-path", required=True,
                        help="retry8 route suffix (route already solves; "
                             "this M-step adds prompt-independent content)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=0.5)
    parser.add_argument("--donor-seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
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

    # Apple donor: authorized teacher trajectory -> BOI-position final residual.
    teacher_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": p22.TEACHER_PROMPTS["red_apple"]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    torch.manual_seed(args.donor_seed)
    mask = p12.build_mask(token_sets, token_sets.text_step_support)
    branch = p12.Branch(model, transformer, teacher_ids, len(teacher_ids) + 256)
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
        raise SystemExit("apple donor did not sample BOI")
    capture_ids = teacher_ids + generated[:boi_step + 1]
    _, donor_finals = p12.capture_states(
        model, transformer, capture_ids, {"boi": len(capture_ids) - 1}, set())
    s_donor = donor_finals["boi"][0].detach().float()

    prompt_ids = {}
    for prompt_id in TRAIN_PROMPTS:
        if prompt_id in d08.PROMPTS:
            text = d08.PROMPTS[prompt_id]
        else:
            text = p22.STUDENT_PROMPTS[prompt_id]
        prompt_ids[prompt_id] = token_manager.tokens_from_ui([
            {"type": "text", "value": text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
    support = d08.legal_mask(token_sets)
    elites = d08.select_elites(args.bridge_metrics, d08.PROMPTS, 0.5, [0.75], 1)

    state = torch.load(args.start_suffix_path, map_location="cuda")
    suffix = state["suffix"].to("cuda").detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for step in range(args.steps):
        t0 = time.time()
        state_terms, route_terms = [], []
        for prompt_id in TRAIN_PROMPTS:
            ids = prompt_ids[prompt_id]
            embeddings = d08.soft_input_embeddings(
                model, ids[:-1], token_sets.eot_id, suffix,
                continuation=[token_sets.boi_id])
            _, hidden = p0.exact_forward(model, transformer, embeddings, return_hidden=True)
            s_w = hidden[-1].float()
            state_terms.append((s_w - s_donor).square().sum() / s_donor.square().sum())
            if prompt_id in elites:
                route_terms.append(p14.differentiable_route_nll(
                    model, transformer, token_sets, ids, suffix,
                    elites[prompt_id][0]["target_ids"], support))
        loss = args.state_weight * torch.stack(state_terms).mean()
        if route_terms:
            loss = loss + args.route_weight * torch.stack(route_terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(suffix.grad).all():
            raise RuntimeError("non-finite suffix gradient")
        optimizer.step()
        append({"kind": "multiprompt_state_inject", "step": step,
                "loss": float(loss.detach()),
                "per_prompt_state_loss": [float(v.detach()) for v in state_terms],
                "route_mean": float(torch.stack(route_terms).mean().detach()) if route_terms else None,
                "seconds": round(time.time() - t0, 2)})
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": suffix.shape[0]},
               output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
