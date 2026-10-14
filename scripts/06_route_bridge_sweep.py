#!/usr/bin/env python
"""No-suffix Anole teacher/student local bridge route sweep.

This deliberately measures route only (first model-sampled BOI within a short
horizon). It uses the official text-state legality support and common post-
bridge top-p warper, with no BOI forcing, output prefill, or mode flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = {
    "red_apple": ("Draw a red apple on a plain light background.", "Explain in one sentence why apples are a popular fruit."),
    "golden_retriever": ("Draw a golden retriever standing on a plain light background.", "Explain in one sentence why golden retrievers are suitable family pets."),
    "yellow_school_bus": ("Draw a yellow school bus on a plain light background.", "Explain in one sentence why school buses are painted yellow."),
    "blue_ceramic_cup": ("Draw a blue ceramic cup on a plain light background.", "Explain in one sentence why ceramic cups are useful for hot drinks."),
}


def sample_one(model, token_manager, model_adapter_cls, token_sets, teacher_prompt, student_prompt, alpha, seed, max_steps, suffix=""):
    import torch
    from morph.decoding.bridge import geometric_bridge
    from morph.decoding.processed_policy import (
        apply_common_warpers,
        apply_repetition_penalty,
        masked_logprobs,
    )

    torch.manual_seed(seed)
    teacher_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": teacher_prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
    ])
    student_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": student_prompt + suffix}, {"type": "sentinel", "value": "<END-OF-TURN>"}
    ])
    teacher = model_adapter_cls(model, max_seq_len=len(teacher_ids) + max_steps + 2)
    student = model_adapter_cls(model, max_seq_len=len(student_ids) + max_steps + 2)
    teacher.initialize([teacher_ids])
    student.initialize([student_ids])
    teacher_seq = torch.tensor([teacher_ids], dtype=torch.long, device="cuda")
    student_seq = torch.tensor([student_ids], dtype=torch.long, device="cuda")
    # Official TextDecoder legality: text tokens + EOS + native BOI.
    legal = torch.zeros(token_sets.vocab_size, dtype=torch.bool, device="cuda")
    legal[list(token_sets.text_step_support)] = True
    teacher_logits = teacher(teacher_seq)[0, -1]
    student_logits = student(student_seq)[0, -1]
    sampled = []
    boi_step = None
    boi_logprob = float("-inf")
    boi_logprob_max = float("-inf")
    boi_rank_at_max = None
    topk_overlaps = []
    for step in range(max_steps):
        # Match Anole's official TextDecoder: legality first, then the
        # history-dependent repetition penalty on each branch.  The bridge is
        # formed over these processed distributions, and only then receives
        # one common top-p warper.
        q_logits = apply_repetition_penalty(teacher_logits, teacher_seq[0], penalty=1.2)
        p_logits = apply_repetition_penalty(student_logits, student_seq[0], penalty=1.2)
        q = masked_logprobs(q_logits, legal)
        p = masked_logprobs(p_logits, legal)
        if not torch.equal(torch.isfinite(q), torch.isfinite(p)):
            raise RuntimeError("Anole teacher/student text support mismatch")
        bridge = geometric_bridge(q, p, alpha)
        bridge = apply_common_warpers(bridge, top_p=0.9)
        boi_logprob = float(bridge[token_sets.boi_id])
        if boi_logprob > boi_logprob_max:
            boi_logprob_max = boi_logprob
            boi_rank_at_max = int((bridge > bridge[token_sets.boi_id]).sum().item()) + 1
        k = min(50, int(torch.isfinite(q).sum()), int(torch.isfinite(p).sum()))
        q_top = set(torch.topk(q, k=k).indices.tolist())
        p_top = set(torch.topk(p, k=k).indices.tolist())
        topk_overlaps.append(len(q_top & p_top) / max(1, k))
        token = int(torch.multinomial(bridge.exp(), 1).item())
        sampled.append(token)
        teacher_seq = torch.cat([teacher_seq, torch.tensor([[token]], device="cuda")], dim=1)
        student_seq = torch.cat([student_seq, torch.tensor([[token]], device="cuda")], dim=1)
        if token == token_sets.boi_id:
            boi_step = step
            break
        if token == token_sets.eos_id:
            break
        teacher_logits = teacher(teacher_seq)[0, -1]
        student_logits = student(student_seq)[0, -1]
    return {
        "alpha": alpha,
        "seed": seed,
        "sampled_tokens": sampled,
        "token_count": len(sampled),
        "boi_sampled": boi_step is not None,
        "boi_step": boi_step,
        "boi_logprob_at_last_step": boi_logprob,
        "boi_logprob_max": boi_logprob_max,
        "boi_rank_at_max": boi_rank_at_max,
        "mean_top50_support_overlap": sum(topk_overlaps) / len(topk_overlaps),
        "suffix": suffix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--prompt-ids", nargs="+", choices=sorted(PROMPTS), default=sorted(PROMPTS))
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))
    import torch
    from chameleon.inference import loader
    from chameleon.inference.chameleon import TokenManager
    from chameleon.inference.model_adapter import ChameleonModelAdapter
    from morph.adapters.anole import load_anole_token_sets

    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    tokens = load_anole_token_sets(tokenizer_path)
    # The public Anole inference service sets the CUDA device before creating
    # the Transformer.  This direct single-GPU diagnostic path must do the
    # same: checkpoint tensors alone being mapped to CUDA does not move module
    # parameters that were constructed on CPU.
    torch.set_default_device("cuda")
    token_manager = TokenManager(str(tokenizer_path), str(model_path / "tokenizer/vqgan.yaml"), str(model_path / "tokenizer/vqgan.ckpt"), device="cuda")
    model = loader.load_model(str(model_path / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    records = []
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
      for prompt_id in args.prompt_ids:
        teacher_prompt, student_prompt = PROMPTS[prompt_id]
        for alpha in args.alphas:
            for seed in args.seeds:
                result = sample_one(model, token_manager, ChameleonModelAdapter, tokens, teacher_prompt, student_prompt, alpha, seed, args.max_steps, args.suffix)
                result.update({"mode": "bridge_route_only", "prompt_id": prompt_id, "teacher_prompt": teacher_prompt, "student_prompt": student_prompt})
                records.append(result)
                line = json.dumps(result)
                metrics_file.write(line + "\n")
                metrics_file.flush()
                print(line, flush=True)
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
