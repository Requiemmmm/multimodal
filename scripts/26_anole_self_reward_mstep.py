#!/usr/bin/env python
"""N0: self-reward-guided M-step with image->text readback signal (run 26).

The collapse attractor is the content-control bottleneck (run 16: 54/64 flat
images).  MoTiF's fix for modal isolation is transition-level supervision in
BOTH directions; we have only ever trained text->image.  Here the model's own
understanding branch scores every generated image (image->text): the reward
is the teacher-forced likelihood of the authorized instruction given the
generated image (Self-SpectraReward), plus a small CLIP target-vs-distractor
margin.  A collapsed image reads out as garbage, so the reward directly
penalizes collapse.

REINFORCE: rollout the deployed (CFG-warped) trajectory, score it, then
backprop (R - baseline) * sum_t log p(y_t | y_<t) through the exact-forward
graph.  The raw-logit recompute under CFG sampling is a known off-policy
bias, accepted for this first pass.  The route NLL stays in the loss as the
constraint (MoTiF warning: end-to-end reward must not displace routing).
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


p22 = _load("probe22", str(PROJECT_ROOT / "scripts/22_anole_context_homotopy_probe.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))
p24 = _load("compress24", str(PROJECT_ROOT / "scripts/24_anole_context_homotopy_compress.py"))


def readback_reward(model, transformer, token_manager, token_sets, visual, caption_tokens):
    """Mean log p(caption_t | [BOI, image, EOI, EOT], caption_<t) -- exact forward."""
    import torch
    ids = [token_sets.bos_id, token_sets.boi_id, *visual, token_sets.eoi_id,
           token_sets.eot_id, *caption_tokens]
    embeddings = model.tok_embeddings(torch.tensor(ids, dtype=torch.long, device="cuda"))
    with torch.no_grad():
        logits = p22.p0.exact_forward(model, transformer, embeddings)
        logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    caption_start = 2 + len(visual) + 2  # first caption token predicted from here
    total = 0.0
    for offset, token in enumerate(caption_tokens):
        total += float(logp[caption_start + offset - 1][token])
    return total / len(caption_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--start-suffix-path", required=True,
                        help="retry8 route suffix (route already solved; N0 attacks content)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["red_apple"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--clip-weight", type=float, default=0.5)
    parser.add_argument("--baseline-rollouts", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gate-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--clip-dir", default=os.environ.get("CLIP_DIR", "") or None)
    parser.add_argument("--max-text-steps", type=int, default=128)
    parser.add_argument("--image-length", type=int, default=1024)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--caption-steps", type=int, default=48)
    parser.add_argument("--guidance-text", type=float, default=3.0)
    parser.add_argument("--guidance-image", type=float, default=1.2)
    parser.add_argument("--image-temp", type=float, default=0.7)
    parser.add_argument("--image-top-p", type=float, default=0.9)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--text-top-p", type=float, default=0.9)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    (output / "images").mkdir(parents=True)
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

    clip_model, clip_proc = None, None
    if Path(args.clip_dir).is_dir():
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(args.clip_dir).to("cuda").eval()
        clip_proc = CLIPProcessor.from_pretrained(args.clip_dir)

    def clip_similarity(image, caption):
        inputs = clip_proc(text=[caption], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            feat_i = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            feat_t = clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            return float(torch.nn.functional.cosine_similarity(feat_i, feat_t).item())

    masks = (
        p22.p12.build_mask(token_sets, token_sets.text_step_support),
        p22.p12.build_mask(token_sets, token_sets.image_token_ids),
    )
    support = p22.d08.legal_mask(token_sets)
    elites = p22.d08.select_elites(args.bridge_metrics, p22.d08.PROMPTS, 0.5, [0.75], 1)
    state = torch.load(args.start_suffix_path, map_location="cuda")
    start_suffix = state["suffix"].to("cuda").detach()
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    from PIL import Image

    for prompt_id in args.prompt_ids:
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": p22.STUDENT_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        caption_tokens = p24.stage_text_tokens(token_manager, p22.TEACHER_PROMPTS[prompt_id])
        elite = elites[prompt_id][0]

        # Fixed baseline: readback reward of empty-suffix rollouts.
        torch.manual_seed(0)
        mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
        empty_suffix = (mean_embedding.repeat(start_suffix.shape[0], 1)
                        + 0.01 * torch.randn(start_suffix.shape[0], mean_embedding.numel(), device="cuda"))
        baseline_rewards = []
        for seed in range(args.baseline_rollouts):
            torch.manual_seed(40000 + seed)
            record = {"condition": "baseline", "prompt_id": prompt_id, "seed": seed}
            record = p22.rollout_soft(
                model, transformer, token_manager, token_sets, student_ids, empty_suffix,
                masks, args, output, clip_similarity, record)
            if record.get("visual_count", 0) == args.image_length:
                visual = [t for t in record["generated_tokens"]
                          if t in token_sets.image_token_ids][:args.image_length]
                baseline_rewards.append(readback_reward(
                    model, transformer, token_manager, token_sets, visual, caption_tokens))
            append({**record, "kind": "n0_baseline_rollout"})
        baseline = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0
        append({"kind": "n0_baseline", "prompt_id": prompt_id,
                "baseline_reward": baseline, "count": len(baseline_rewards)})

        suffix = start_suffix.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
        for step in range(args.steps):
            torch.manual_seed(30000 + step)
            record = {"condition": "n0_step", "prompt_id": prompt_id, "seed": step}
            record = p22.rollout_soft(
                model, transformer, token_manager, token_sets, student_ids, suffix.detach(),
                masks, args, output, clip_similarity, record)
            record["kind"] = "n0_step_rollout"
            append(record)
            if record.get("visual_count", 0) != args.image_length:
                append({"kind": "n0_step", "prompt_id": prompt_id, "step": step,
                        "note": "rollout did not complete an image; skipping update"})
                continue
            visual = [t for t in record["generated_tokens"]
                      if t in token_sets.image_token_ids][:args.image_length]
            reward = readback_reward(model, transformer, token_manager, token_sets,
                                     visual, caption_tokens)
            clip_target = record.get("clip_target_similarity")
            clip_distractor = record.get("clip_distractor_similarity")
            if clip_target is not None and clip_distractor is not None:
                reward += args.clip_weight * (clip_target - clip_distractor)

            # REINFORCE gradient through the sampled trajectory (up to EOI).
            generated = record["generated_tokens"]
            eoi_index = next(i for i, t in enumerate(generated) if t == token_sets.eoi_id)
            credit_tokens = generated[:eoi_index + 1]
            embeddings = p22.d08.soft_input_embeddings(
                model, student_ids[:-1], token_sets.eot_id, suffix,
                continuation=credit_tokens)
            logits = p22.p0.exact_forward(model, transformer, embeddings)
            logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            first_prediction = len(student_ids) + suffix.shape[0] - 1
            reinforce = 0.0
            for offset, token in enumerate(credit_tokens):
                reinforce = reinforce + logp[first_prediction + offset][token]
            reinforce = reinforce / len(credit_tokens)
            route = p14.differentiable_route_nll(
                model, transformer, token_sets, student_ids, suffix,
                elite["target_ids"], support)
            loss = args.route_weight * route + args.reward_weight * (-(reward - baseline)) * reinforce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(suffix.grad).all():
                append({"kind": "n0_step", "prompt_id": prompt_id, "step": step,
                        "note": "non-finite gradient; skipping update"})
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            append({"kind": "n0_step", "prompt_id": prompt_id, "step": step,
                    "readback_reward": reward, "baseline": baseline,
                    "advantage": reward - baseline, "reinforce_logp": float(reinforce.detach()),
                    "route_nll": float(route.detach()), "loss": float(loss.detach()),
                    "clip_target": clip_target, "clip_distractor": clip_distractor})

            if (step + 1) % args.eval_every == 0:
                summary = p24.gate_eval(
                    model, transformer, token_manager, token_sets, student_ids, suffix.detach(),
                    masks, args, output, clip_similarity, prompt_id, step + 1, args.gate_seeds)
                append({"kind": "n0_gate", "prompt_id": prompt_id, "step": step + 1, **summary})

        torch.save({"suffix": suffix.detach().cpu(),
                    "suffix_length": suffix.shape[0]}, output / f"soft_suffix_{prompt_id}.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
