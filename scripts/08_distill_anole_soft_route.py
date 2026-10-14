#!/usr/bin/env python
"""MORPH-v0 route-only soft-suffix distillation from completed bridge elites.

The bridge trajectories are fixed, stop-gradient proposals from a separately
recorded run.  This M-step freezes every Anole weight and optimizes only an
ordinary user-message soft suffix before EOT.  Native student-only rollouts
are evaluated separately and never use teacher logits, bridge interpolation,
BOI injection, or an image mode flag.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = {
    "red_apple": "Explain in one sentence why apples are a popular fruit.",
    "golden_retriever": "Explain in one sentence why golden retrievers are suitable family pets.",
    "yellow_school_bus": "Explain in one sentence why school buses are painted yellow.",
    "blue_ceramic_cup": "Explain in one sentence why ceramic cups are useful for hot drinks.",
}


def _rms_norm(x, weight, eps):
    """Elementwise RMSNorm matching xformers' float32 Triton accumulation."""
    import torch
    # xformers loads activations and weights as float32, multiplies in float32,
    # and writes the result to bf16.  Casting rstd to bf16 before multiplication
    # creates a small error at every norm that compounds across all 32 layers.
    normalized = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    )
    return normalized.to(dtype=x.dtype)


def _rope(x, freqs_cis):
    """Apply RoPE rotation.  x: [seq, heads, head_dim]; freqs_cis: [seq, head_dim//2] complex."""
    import torch
    # xformers' kernel evaluates sin/cos and products in the bf16 input path;
    # materializing a complex float32 product changes rounding before the cache
    # write.  Keep trig values and pairwise products in the model dtype.
    angles = torch.angle(freqs_cis).to(device=x.device, dtype=x.dtype)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    real = x[..., 0::2]
    imag = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = real * cos - imag * sin
    out[..., 1::2] = imag * cos + real * sin
    return out


def legal_mask(token_sets):
    import torch

    mask = torch.zeros(token_sets.vocab_size, dtype=torch.bool, device="cuda")
    mask[list(token_sets.text_step_support)] = True
    return mask


def soft_input_embeddings(model, prompt_without_eot, eot_id, suffix, continuation=()):
    import torch

    ids = torch.tensor([*prompt_without_eot, eot_id, *continuation], dtype=torch.long, device="cuda")
    prefix_count = len(prompt_without_eot)
    embedded_ids = model.tok_embeddings(ids)
    return torch.cat((embedded_ids[:prefix_count], suffix.to(dtype=embedded_ids.dtype), embedded_ids[prefix_count:]), dim=0)


def precompute_freqs_cis(dim, end, theta=10000.0):
    """Precompute RoPE rotation matrices."""
    import torch

    positions = torch.arange(end, device="cuda", dtype=torch.float32).unsqueeze(1)
    powers = torch.arange(0, dim, 2, device="cuda", dtype=torch.float32).unsqueeze(0)
    return positions * torch.pow(torch.tensor(theta, device="cuda"), powers / -dim)


def full_forward(model, transformer, embeddings):
    """Exact deployed Anole forward for gradient-free suffix optimization."""
    import torch

    length = embeddings.shape[0]
    cache = transformer.make_cache(model.args, length=length, device="cuda", dtype=embeddings.dtype)
    bias = transformer.AttnBias.from_seqlens(q_seqlen=[length], kv_seqlen=[length], kv_padding=length)
    with torch.no_grad():
        hidden = embeddings.to(dtype=next(model.parameters()).dtype)
        for index, layer in enumerate(model.layers):
            hidden = layer(hidden, cache[index], bias)
        return model.output(model.norm(hidden)).float()


class SoftTextContext:
    """No-grad public TextDecoder-equivalent cache state for a continuous suffix."""

    def __init__(self, model, transformer, token_sets, prompt_ids, suffix, max_steps):
        import torch

        self.model = model
        self.token_sets = token_sets
        self.history = list(prompt_ids)
        prompt_without_eot = prompt_ids[:-1]
        initial_length = len(prompt_ids) + suffix.shape[0]
        self.cache = transformer.make_cache(
            model.args,
            length=initial_length + max_steps + 2,
            device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        self.bias = transformer.AttnBias.from_seqlens(
            q_seqlen=[initial_length], kv_seqlen=[initial_length], kv_padding=initial_length + max_steps + 2
        )
        embeddings = soft_input_embeddings(model, prompt_without_eot, token_sets.eot_id, suffix.detach())
        with torch.no_grad():
            hidden = embeddings
            for index, layer in enumerate(model.layers):
                hidden = layer(hidden, self.cache[index], self.bias)
            self.logits = model.output(model.norm(hidden)).float()[-1]
        self.bias.q_seqinfo.seqstart.copy_(torch.arange(2, dtype=torch.int, device="cuda"))
        self.bias.q_seqinfo.max_seqlen = 1
        self.bias.q_seqinfo.seqstart_py = self.bias.q_seqinfo.seqstart.tolist()
        self.bias.k_seqinfo.seqlen.add_(1)

    def advance(self, token):
        import torch

        tensor = torch.tensor([token], dtype=torch.long, device="cuda")
        with torch.no_grad():
            self.logits = self.model.forward_with_attn_bias(tensor, self.bias, self.cache).float()[0]
        self.bias.k_seqinfo.seqlen.add_(1)
        self.history.append(token)


def route_snapshot(policy, boi_id):
    boi_logprob = float(policy[boi_id])
    return {
        "boi_logprob_pre_warp": boi_logprob,
        "boi_probability_pre_warp": math.exp(boi_logprob),
        "boi_rank_pre_warp": int((policy > policy[boi_id]).sum().item()) + 1,
    }


def student_route_nll(model, transformer, token_sets, prompt_ids, suffix, target_ids, support):
    """Exact segmented route NLL through a fixed bridge trajectory, including BOI."""
    import torch
    from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs

    if not target_ids or target_ids[-1] != token_sets.boi_id:
        raise ValueError("route elite must end in BOI")
    embeddings = soft_input_embeddings(model, prompt_ids[:-1], token_sets.eot_id, suffix, target_ids[:-1])
    logits = full_forward(model, transformer, embeddings)
    first_prediction = len(prompt_ids) + suffix.shape[0] - 1
    history = list(prompt_ids)
    losses = []
    for offset, target in enumerate(target_ids):
        past = torch.tensor(history, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(logits[first_prediction + offset], past, penalty=1.2), support
        )
        # BOI is the route boundary.  It receives a bounded extra weight so a
        # long text preamble does not swamp the actual modality event.
        weight = 4.0 if target == token_sets.boi_id else 1.0
        losses.append(-weight * policy[target])
        history.append(target)
    return torch.stack(losses).sum() / sum(4.0 if token == token_sets.boi_id else 1.0 for token in target_ids)


def mean_route_nll(model, transformer, token_sets, prompt_ids, suffix, flat_elites, support):
    import torch

    with torch.no_grad():
        losses = [
            student_route_nll(model, transformer, token_sets, prompt_ids[prompt_id], suffix, elite["target_ids"], support)
            for prompt_id, elite in flat_elites
        ]
        return torch.stack(losses).mean(), losses


def spsa_gradient(model, transformer, token_sets, prompt_ids, suffix, flat_elites, support, epsilon, samples):
    """Two-sided SPSA gradient of the exact official-forward route objective."""
    import torch

    if epsilon <= 0 or samples <= 0:
        raise ValueError("SPSA epsilon and sample count must be positive")
    gradient = torch.zeros_like(suffix)
    losses_plus, losses_minus = [], []
    with torch.no_grad():
        for _ in range(samples):
            direction = torch.empty_like(suffix).bernoulli_(0.5).mul_(2).sub_(1)
            plus, _ = mean_route_nll(
                model, transformer, token_sets, prompt_ids, suffix + epsilon * direction, flat_elites, support
            )
            minus, _ = mean_route_nll(
                model, transformer, token_sets, prompt_ids, suffix - epsilon * direction, flat_elites, support
            )
            gradient.add_((plus - minus) / (2 * epsilon) * direction)
            losses_plus.append(float(plus))
            losses_minus.append(float(minus))
    return gradient.div_(samples), losses_plus, losses_minus


def select_elites(metrics_path, prompts, preferred_alpha, fallback_alphas, per_prompt):
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    by_prompt = defaultdict(list)
    allowed_alphas = [preferred_alpha, *fallback_alphas]
    for row in rows:
        if row.get("prompt_id") not in prompts or not row.get("boi_sampled"):
            continue
        alpha = float(row["alpha"])
        if alpha not in allowed_alphas:
            continue
        tokens = row.get("sampled_tokens", [])
        if not tokens:
            continue
        by_prompt[row["prompt_id"]].append(row)
    selected = {}
    for prompt_id in prompts:
        candidates = sorted(
            by_prompt[prompt_id],
            key=lambda row: (
                float(row["alpha"]) != preferred_alpha,
                -float(row.get("boi_logprob_max", float("-inf"))),
                len(row["sampled_tokens"]),
                row["seed"],
            ),
        )
        if len(candidates) < per_prompt:
            raise RuntimeError(
                f"need {per_prompt} BOI bridge elites for {prompt_id}; only {len(candidates)} are available in {metrics_path}"
            )
        selected[prompt_id] = [
            {
                "source_alpha": row["alpha"],
                "source_seed": row["seed"],
                "target_ids": row["sampled_tokens"][: row["boi_step"] + 1],
            }
            for row in candidates[:per_prompt]
        ]
    return selected


def evaluate_student_only(model, transformer, token_sets, prompt_ids, suffix, seeds, max_steps):
    """Student-only native route evaluation for a soft suffix."""
    import torch
    from morph.decoding.processed_policy import apply_common_warpers, apply_repetition_penalty, masked_logprobs

    support = legal_mask(token_sets)
    records = []
    for prompt_id, ids in prompt_ids.items():
        for seed in seeds:
            torch.manual_seed(seed)
            context = SoftTextContext(model, transformer, token_sets, ids, suffix, max_steps)
            sampled, boi_step, best = [], None, None
            for step in range(max_steps):
                history = torch.tensor(context.history, dtype=torch.long, device="cuda")
                policy = masked_logprobs(apply_repetition_penalty(context.logits, history, penalty=1.2), support)
                snapshot = route_snapshot(policy, token_sets.boi_id)
                if best is None or snapshot["boi_logprob_pre_warp"] > best["boi_logprob_pre_warp"]:
                    best = snapshot
                token = int(torch.multinomial(apply_common_warpers(policy, top_p=0.9).exp(), 1).item())
                sampled.append(token)
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
                "sampled_tokens": sampled,
                "token_count": len(sampled),
                "boi_sampled": boi_step is not None,
                "boi_step": boi_step,
                "max_route": best,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preferred-alpha", type=float, default=0.5)
    parser.add_argument("--fallback-alphas", type=float, nargs="*", default=[0.75])
    parser.add_argument("--elites-per-prompt", type=int, default=4)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--spsa-epsilon", type=float, default=0.01)
    parser.add_argument("--spsa-samples", type=int, default=1)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    if not Path(args.bridge_metrics).is_file():
        raise SystemExit(f"bridge metrics file does not exist: {args.bridge_metrics}")
    if min(args.elites_per_prompt, args.suffix_length, args.steps, args.spsa_samples) <= 0 or min(
        args.learning_rate, args.spsa_epsilon
    ) <= 0:
        raise SystemExit("elite count, suffix length, steps, SPSA samples, learning rate, and SPSA epsilon must be positive")
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
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in PROMPTS.items()
    }
    elites = select_elites(
        args.bridge_metrics, PROMPTS, args.preferred_alpha, args.fallback_alphas, args.elites_per_prompt
    )
    (output / "selected_elites.json").write_text(json.dumps(elites, indent=2) + "\n")
    support = legal_mask(token_sets)
    torch.manual_seed(args.init_seed)
    mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
    suffix = (
        mean_embedding.repeat(args.suffix_length, 1)
        + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda")
    )
    metrics_path = output / "metrics.jsonl"

    def append_metrics(records):
        with metrics_path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    if args.skip_initial_eval:
        append_metrics([{
            "kind": "run_note",
            "phase": "initial_soft_control",
            "note": "initial student-only evaluation skipped by explicit CLI flag",
        }])
    else:
        initial_records = evaluate_student_only(
            model, transformer, token_sets, prompt_ids, suffix, args.eval_seeds, args.max_steps
        )
        for record in initial_records:
            record["phase"] = "initial_soft_control"
        append_metrics(initial_records)

    flat_elites = [(prompt_id, elite) for prompt_id, rows in elites.items() for elite in rows]
    for step in range(args.steps):
        gradient, spsa_plus, spsa_minus = spsa_gradient(
            model, transformer, token_sets, prompt_ids, suffix, flat_elites, support,
            args.spsa_epsilon, args.spsa_samples,
        )
        if not torch.isfinite(gradient).all():
            raise RuntimeError("SPSA produced a non-finite suffix gradient estimate")
        with torch.no_grad():
            suffix -= args.learning_rate * gradient
            loss, losses = mean_route_nll(model, transformer, token_sets, prompt_ids, suffix, flat_elites, support)
        grad_norm = float(gradient.norm())
        record = {
            "kind": "soft_suffix_route_distillation",
            "step": step,
            "student_route_nll": float(loss.detach()),
            "suffix_grad_norm": grad_norm,
            "per_elite_nll": [float(item.detach()) for item in losses],
            "optimizer": "spsa",
            "spsa_epsilon": args.spsa_epsilon,
            "spsa_samples": args.spsa_samples,
            "spsa_loss_plus": spsa_plus,
            "spsa_loss_minus": spsa_minus,
        }
        append_metrics([record])
        print(json.dumps(record), flush=True)

    trained_records = evaluate_student_only(
        model, transformer, token_sets, prompt_ids, suffix, args.eval_seeds, args.max_steps
    )
    for record in trained_records:
        record["phase"] = "trained_soft_suffix"
        print(json.dumps(record), flush=True)
    append_metrics(trained_records)
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length}, output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
