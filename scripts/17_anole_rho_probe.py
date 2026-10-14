#!/usr/bin/env python
"""Jacobian reachability probe (expert doc section 8.2) for the gateway state.

Question: can a 16-slot soft suffix at the user position span the gateway
displacement d_c = s_donor - s_ordinary, where s is the final (pre-norm)
residual at the BOI position -- the coordinate the P1 ladder showed carries
the visual plan?

Method: optimize w (soft suffix perturbation, same init as the M-step) with
exact-forward Adam to minimize ||s(w) - s_donor||^2 / ||s_donor||^2 and report:
- rho_achieved = 1 - final relative loss  (>= true projection ratio of d_c
  onto the suffix Jacobian column space; small rho => direction unreachable
  by ANY soft suffix, hence by any hard text suffix)
- control energy E = mean ||w - w_init||^2 per slot
- nearest-text-embedding cosine distance of the matched w (hard-reachability
  proxy: if even the optimal continuous direction is far from the text
  vocabulary region, hardening is structurally hard)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["red_apple", "golden_retriever"])
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=0)
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
    text_embeddings = model.tok_embeddings.weight.detach().float()

    def donor_state(prompt_id):
        """Authorized teacher trajectory -> BOI-position final residual.

        The captured state includes attention over the teacher's natural text
        preamble, so the probe asks whether the suffix can substitute for the
        plan-building role of that preamble."""
        ids = token_manager.tokens_from_ui([
            {"type": "text", "value": TEACHER_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        torch.manual_seed(args.donor_seed)
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

    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for prompt_id in args.prompt_ids:
        s_donor, boi_step = donor_state(prompt_id)
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": STUDENT_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        torch.manual_seed(args.init_seed)
        mean_embedding = text_embeddings.mean(dim=0)
        w = (mean_embedding.repeat(args.suffix_length, 1)
             + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda"))
        w = w.detach().requires_grad_(True)
        w_init = w.detach().clone()
        optimizer = torch.optim.Adam([w], lr=args.learning_rate)
        initial_loss = None
        for step in range(args.steps):
            embeddings = d08.soft_input_embeddings(
                model, student_ids[:-1], token_sets.eot_id, w,
                continuation=[token_sets.boi_id])
            _, hidden = p0.exact_forward(model, transformer, embeddings, return_hidden=True)
            s_w = hidden[-1].float()
            loss = (s_w - s_donor).square().sum() / s_donor.square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = float(loss.detach())
            append({
                "kind": "rho_probe_step", "prompt_id": prompt_id, "step": step,
                "relative_state_loss": float(loss.detach()),
                "grad_norm": float(w.grad.norm()),
            })
        final_loss = float(loss.detach())
        rho = max(0.0, 1.0 - final_loss / max(initial_loss, 1e-12)) if initial_loss else None
        energy = float((w.detach() - w_init).square().mean())
        norms_w = torch.nn.functional.normalize(w.detach(), dim=-1)
        norms_e = torch.nn.functional.normalize(text_embeddings, dim=-1)
        sims = norms_w @ norms_e.T
        nearest = sims.max(dim=-1).values
        append({
            "kind": "rho_probe_summary", "prompt_id": prompt_id,
            "donor_boi_step": boi_step, "initial_relative_loss": initial_loss,
            "final_relative_loss": final_loss, "rho_achieved": rho,
            "control_energy_per_slot": energy,
            "nearest_text_cosine_mean": float(nearest.mean()),
            "nearest_text_cosine_min": float(nearest.min()),
            "note": "rho_achieved >= true projection ratio; small rho means the "
                    "gateway direction is unreachable by any suffix at this position",
        })
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
