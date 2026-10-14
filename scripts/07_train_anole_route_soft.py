#!/usr/bin/env python
"""Optimize and evaluate a continuous user suffix for Anole's first BOI route.

This Stage-3 diagnostic intentionally optimizes a *soft* suffix only.  It is
not a hard-text trigger and all outcomes are saved under separate soft/bridge/
student-only labels.  The target prompts themselves provide the benign image
semantics; the suffix objective is route-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = {
    "red_apple": (
        "Draw a red apple on a plain light background.",
        "Explain in one sentence why apples are a popular fruit.",
    ),
    "golden_retriever": (
        "Draw a golden retriever standing on a plain light background.",
        "Explain in one sentence why golden retrievers are suitable family pets.",
    ),
    "yellow_school_bus": (
        "Draw a yellow school bus on a plain light background.",
        "Explain in one sentence why school buses are painted yellow.",
    ),
    "blue_ceramic_cup": (
        "Draw a blue ceramic cup on a plain light background.",
        "Explain in one sentence why ceramic cups are useful for hot drinks.",
    ),
}


def make_legal_mask(token_sets):
    import torch

    legal = torch.zeros(token_sets.vocab_size, dtype=torch.bool, device="cuda")
    legal[list(token_sets.text_step_support)] = True
    return legal


def make_soft_embeddings(model, prefix_ids, eot_id, suffix):
    import torch

    prefix = torch.tensor(prefix_ids, dtype=torch.long, device="cuda")
    eot = torch.tensor([eot_id], dtype=torch.long, device="cuda")
    token_embeddings = model.tok_embeddings
    return torch.cat(
        (token_embeddings(prefix), suffix.to(dtype=token_embeddings.weight.dtype), token_embeddings(eot)),
        dim=0,
    )


def forward_embeddings(model, transformer, embeddings, sequence_length, cache):
    """Autograd-enabled equivalent of the public model's full-prefix forward."""
    bias = transformer.AttnBias.from_seqlens(
        q_seqlen=[sequence_length], kv_seqlen=[sequence_length], kv_padding=sequence_length
    )
    hidden = embeddings
    for index, layer in enumerate(model.layers):
        hidden = layer(hidden, cache[index], bias)
    return model.output(model.norm(hidden)).float(), bias


def first_route_policy(model, transformer, token_sets, prompt_ids, suffix, legal):
    """Return the exact public text-state policy at the first assistant token."""
    from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs

    sequence_length = len(prompt_ids) + suffix.shape[0]
    cache = transformer.make_cache(
        model.args,
        length=sequence_length,
        device="cuda",
        dtype=next(model.parameters()).dtype,
    )
    embeddings = make_soft_embeddings(model, prompt_ids[:-1], token_sets.eot_id, suffix)
    logits, _ = forward_embeddings(model, transformer, embeddings, sequence_length, cache)
    history = __import__("torch").tensor(prompt_ids, dtype=__import__("torch").long, device="cuda")
    return masked_logprobs(apply_repetition_penalty(logits[-1], history, penalty=1.2), legal)


class SoftContext:
    """One no-grad cached text decoder state with continuous suffix embeddings."""

    def __init__(self, model, transformer, token_sets, prompt_ids, suffix, max_steps):
        import torch

        self.model = model
        self.transformer = transformer
        self.token_sets = token_sets
        self.history = list(prompt_ids)
        sequence_length = len(prompt_ids) + suffix.shape[0]
        self.max_seq_len = sequence_length + max_steps + 2
        self.cache = transformer.make_cache(
            model.args,
            length=self.max_seq_len,
            device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        embeddings = make_soft_embeddings(model, prompt_ids[:-1], token_sets.eot_id, suffix.detach())
        with torch.no_grad():
            self.logits, self.bias = forward_embeddings(
                model, transformer, embeddings, sequence_length, self.cache
            )
            self.logits = self.logits[-1]
        # This is the state transition performed by ChameleonModelAdapter after
        # its prompt pass: each later call consumes precisely one sampled token.
        self.bias.q_seqinfo.seqstart.copy_(torch.arange(2, dtype=torch.int, device="cuda"))
        self.bias.q_seqinfo.max_seqlen = 1
        self.bias.q_seqinfo.seqstart_py = self.bias.q_seqinfo.seqstart.tolist()
        self.bias.k_seqinfo.seqlen.add_(1)

    def advance(self, token_id):
        import torch

        token = torch.tensor([token_id], dtype=torch.long, device="cuda")
        with torch.no_grad():
            self.logits = self.model.forward_with_attn_bias(token, self.bias, self.cache).float()[0]
        self.bias.k_seqinfo.seqlen.add_(1)
        self.history.append(token_id)


def route_snapshot(policy, boi_id):
    boi_logprob = float(policy[boi_id])
    return {
        "boi_logprob_pre_warp": boi_logprob,
        "boi_probability_pre_warp": math.exp(boi_logprob),
        "boi_rank_pre_warp": int((policy > policy[boi_id]).sum().item()) + 1,
    }


def evaluate_student_only(model, transformer, token_sets, prompt_ids, suffix, seeds, max_steps):
    """Native-policy rollouts: no teacher, bridge, injected BOI, or mode flag."""
    import torch
    from morph.decoding.processed_policy import apply_common_warpers, apply_repetition_penalty, masked_logprobs

    legal = make_legal_mask(token_sets)
    records = []
    for prompt_id, ids in prompt_ids.items():
        for seed in seeds:
            torch.manual_seed(seed)
            context = SoftContext(model, transformer, token_sets, ids, suffix, max_steps)
            generated, boi_step = [], None
            max_snapshot = None
            for step in range(max_steps):
                history = torch.tensor(context.history, dtype=torch.long, device="cuda")
                policy = masked_logprobs(
                    apply_repetition_penalty(context.logits, history, penalty=1.2), legal
                )
                snapshot = route_snapshot(policy, token_sets.boi_id)
                if max_snapshot is None or snapshot["boi_logprob_pre_warp"] > max_snapshot["boi_logprob_pre_warp"]:
                    max_snapshot = snapshot
                token = int(torch.multinomial(apply_common_warpers(policy, top_p=0.9).exp(), 1).item())
                generated.append(token)
                if token == token_sets.boi_id:
                    boi_step = step
                    break
                if token == token_sets.eos_id:
                    break
                context.advance(token)
            records.append({
                "kind": "soft_suffix_student_only_rollout",
                "prompt_id": prompt_id,
                "seed": seed,
                "sampled_tokens": generated,
                "token_count": len(generated),
                "boi_sampled": boi_step is not None,
                "boi_step": boi_step,
                "max_route": max_snapshot,
            })
    return records


def nearest_text_tokens(suffix, embedding_weight, text_ids, tokenizer, k=4):
    import torch

    candidates = embedding_weight[list(sorted(text_ids))].float()
    candidates = torch.nn.functional.normalize(candidates, dim=-1)
    normalized = torch.nn.functional.normalize(suffix.detach().float(), dim=-1)
    scores, positions = torch.topk(normalized @ candidates.T, k=k, dim=-1)
    ordered_ids = sorted(text_ids)
    result = []
    for row_scores, row_positions in zip(scores.cpu().tolist(), positions.cpu().tolist()):
        ids = [ordered_ids[position] for position in row_positions]
        result.append({
            "token_ids": ids,
            "cosine": row_scores,
            "decoded": [tokenizer.decode([token_id]) for token_id in ids],
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    if args.suffix_length <= 0 or args.steps <= 0 or args.learning_rate <= 0:
        raise SystemExit("suffix length, steps, and learning rate must be positive")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(Path(args.code_root)))
    import torch
    from chameleon.inference import loader, transformer
    from chameleon.inference.chameleon import TokenManager
    from morph.adapters.anole import load_anole_token_sets

    torch.set_default_device("cuda")
    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(tokenizer_path)
    token_manager = TokenManager(
        str(tokenizer_path), str(model_path / "tokenizer/vqgan.yaml"),
        str(model_path / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    model = loader.load_model(str(model_path / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": student_prompt},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        for prompt_id, (_, student_prompt) in PROMPTS.items()
    }
    legal = make_legal_mask(token_sets)
    torch.manual_seed(0)
    initial = model.tok_embeddings.weight.detach().float().mean(dim=0)
    suffix = torch.nn.Parameter(
        initial.repeat(args.suffix_length, 1)
        + 0.01 * torch.randn(args.suffix_length, initial.numel(), device="cuda")
    )
    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
    metrics = []

    # Evaluate the same randomly initialized soft prompt before updates.  This
    # is a soft-control baseline, distinct from the earlier hard/no-suffix one.
    initial_eval = evaluate_student_only(
        model, transformer, token_sets, prompt_ids, suffix, args.eval_seeds, args.max_steps
    )
    for record in initial_eval:
        record["phase"] = "initial_soft_control"
        metrics.append(record)

    for step in range(args.steps):
        policies = [first_route_policy(model, transformer, token_sets, ids, suffix, legal) for ids in prompt_ids.values()]
        loss = -torch.stack([policy[token_sets.boi_id] for policy in policies]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(suffix.grad.norm())
        optimizer.step()
        snapshots = {prompt_id: route_snapshot(policy.detach(), token_sets.boi_id) for prompt_id, policy in zip(prompt_ids, policies)}
        record = {"kind": "soft_suffix_optimization", "step": step, "loss": float(loss.detach()), "suffix_grad_norm": grad_norm, "route": snapshots}
        metrics.append(record)
        print(json.dumps(record), flush=True)

    final_eval = evaluate_student_only(
        model, transformer, token_sets, prompt_ids, suffix, args.eval_seeds, args.max_steps
    )
    for record in final_eval:
        record["phase"] = "trained_soft_suffix"
        metrics.append(record)
        print(json.dumps(record), flush=True)

    with (output / "metrics.jsonl").open("w") as handle:
        for record in metrics:
            handle.write(json.dumps(record) + "\n")
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length}, output / "soft_suffix.pt")
    (output / "nearest_text_tokens.json").write_text(json.dumps(
        nearest_text_tokens(suffix, model.tok_embeddings.weight.detach(), token_sets.text_token_ids, token_manager.tokenizer),
        ensure_ascii=False, indent=2,
    ) + "\n")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
