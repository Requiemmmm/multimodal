#!/usr/bin/env python
"""Apple-only route M-step (run 37): train the suffix on red_apple elites ONLY.

retry8 (mixed 4-prompt elites) flipped strawberry prompts to apples 10/15 --
the slot residuals carry the elite preamble text.  A single-concept elite
training set should sharpen that bias: the hypothesis is a suffix trained on
apple preambles alone carries a stronger, purer apple content signal, flipping
mismatched prompts more reliably (evaluated separately on strawberry / sky).
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
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--target-prompt", default="red_apple",
                        help="d08.PROMPTS key whose elites train the suffix "
                             "(the concept the suffix will carry)")
    parser.add_argument("--teacher-text", default=None,
                        help="authorized teacher instruction for the target "
                             "concept; overrides bridge elites (elite = the "
                             "teacher's own freeform preamble + BOI)")
    parser.add_argument("--train-prompt", default=None,
                        help="student prompt key (or plain text if no key "
                             "matches) used as the route-NLL context; "
                             "default = the target's own student prompt")
    parser.add_argument("--train-prompts-file", default=None,
                        help="JSON list of student prompt texts; the route "
                             "NLL is averaged over (prompt x elite) pairs "
                             "-- victim-context ensemble against memorizing "
                             "a single sentence")
    parser.add_argument("--boi-only", action="store_true",
                        help="ablation: target is just [BOI] (no preamble "
                             "text residual) -- isolates the content carrier")
    parser.add_argument("--quant-penalty", type=float, default=0.0,
                        help="quantization-aware training: pull each slot "
                             "toward its nearest text token (max-cosine "
                             "penalty). 0 disables. Retrains a vocab-near "
                             "suffix so hard projection loses less signal.")
    parser.add_argument("--init-suffix", default=None,
                        help="soft_suffix.pt to warm-start from (projection "
                             "fine-tuning) instead of random init")
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

    prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in d08.PROMPTS.items()
    }
    if args.boi_only:
        target_elites = [{"target_ids": [token_sets.boi_id]}]
    elif args.teacher_text:
        # Elite = the teacher's own freeform preamble + BOI (no bridge needed).
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": args.teacher_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        torch.manual_seed(0)
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
            raise SystemExit("teacher did not sample BOI for the given text")
        target_elites = [{"target_ids": generated[:boi_step + 1]}]
    else:
        elites = d08.select_elites(
            args.bridge_metrics, d08.PROMPTS, 0.5, [0.75], 1)
        target_elites = elites.get(args.target_prompt, [])
        if not target_elites:
            raise SystemExit(f"no {args.target_prompt} elites found in bridge metrics")
    (output / "selected_elites.json").write_text(json.dumps(target_elites, indent=2) + "\n")
    # Route-NLL context: the deployed victim prompt.
    train_ids_list = []
    if args.train_prompts_file:
        train_texts = json.loads(Path(args.train_prompts_file).read_text())
        if not train_texts:
            raise SystemExit(f"empty train prompts file: {args.train_prompts_file}")
        for text in train_texts:
            train_ids_list.append(token_manager.tokens_from_ui([
                {"type": "text", "value": text},
                {"type": "sentinel", "value": "<END-OF-TURN>"},
            ]))
    elif args.train_prompt:
        if args.train_prompt in d08.PROMPTS:
            train_text = d08.PROMPTS[args.train_prompt]
        elif args.train_prompt in p22.STUDENT_PROMPTS:
            train_text = p22.STUDENT_PROMPTS[args.train_prompt]
        else:
            train_text = args.train_prompt
        train_ids_list.append(token_manager.tokens_from_ui([
            {"type": "text", "value": train_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ]))
    else:
        train_ids_list.append(prompt_ids[args.target_prompt])
    support = d08.legal_mask(token_sets)
    if args.init_suffix:
        init_payload = torch.load(args.init_suffix, map_location="cpu", weights_only=True)
        suffix = init_payload["suffix"].float().to("cuda")
        if suffix.shape[0] != args.suffix_length:
            raise SystemExit(
                f"init suffix has {suffix.shape[0]} slots, expected {args.suffix_length}")
        suffix = suffix.detach().requires_grad_(True)
    else:
        torch.manual_seed(args.init_seed)
        mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
        suffix = (mean_embedding.repeat(args.suffix_length, 1)
                  + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda"))
        suffix = suffix.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for step in range(args.steps):
        t0 = time.time()
        losses = [
            p14.differentiable_route_nll(
                model, transformer, token_sets,
                train_ids, suffix, elite["target_ids"], support)
            for train_ids in train_ids_list
            for elite in target_elites
        ]
        loss = torch.stack(losses).mean()
        route_nll = float(loss.detach())
        quant_loss = None
        if args.quant_penalty:
            vocab_norm = torch.nn.functional.normalize(
                model.tok_embeddings.weight.detach().float(), dim=-1)
            slot_cos = torch.nn.functional.normalize(suffix, dim=-1) @ vocab_norm.T
            quant_loss = -slot_cos.max(dim=-1).values.mean()
            loss = loss + args.quant_penalty * quant_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(suffix.grad).all():
            raise RuntimeError("non-finite suffix gradient")
        optimizer.step()
        record = {"kind": "apple_route_mstep", "step": step,
                  "route_nll": route_nll,
                  "seconds": round(time.time() - t0, 2)}
        if quant_loss is not None:
            record["quant_penalty"] = args.quant_penalty
            record["max_cos_mean"] = float(-quant_loss.detach())
            record["total_loss"] = float(loss.detach())
        append(record)
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length},
               output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
