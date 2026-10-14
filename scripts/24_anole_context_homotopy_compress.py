#!/usr/bin/env python
"""N1: context-side homotopy compression loop (run 24).

Start from the full authorized instruction carried in the 16-slot soft suffix
(stage 0 reference: the plan already works in context -- verified by the run
22 probe), then compress the instruction token-by-token through a per-prompt
ladder.  After each compression step, K Adam M-steps let the suffix absorb the
deleted tokens' function, supervised by:

  L = route NLL (bridge elite target, scripts/14 exact-forward machinery)
    + lambda_state * on-policy gateway matching

The state term matches the student's OWN BOI/vis1-position final residuals
(collected from frozen rollouts of the current suffix -- on-policy, never
teacher-forced, the run 18 lesson) against the authorized donor states.

After each stage a gate runs n free full-image rollouts (run 16 protocol:
completion + collapse stats + CLIP target/distractor + Self-SpectraReward).
Degradation beyond tolerance reverts the step: first retry with 2x M-steps,
then insert a half-token intermediate stage; if the floor persists, the last
stable stage is recorded as the compression floor (a citable negative result
in itself).
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

LADDERS = {
    "red_apple": [
        "Draw a red apple on a plain light background.",
        "Draw a red apple.",
        "red apple",
        "",
    ],
    "golden_retriever": [
        "Draw a golden retriever standing on a plain light background.",
        "Draw a golden retriever.",
        "golden retriever",
        "",
    ],
}


def stage_text_tokens(token_manager, text):
    if not text:
        return []
    ids = token_manager.tokens_from_ui([
        {"type": "text", "value": text},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    return ids[1:-1]


def stage_suffix(model, token_manager, text, length, mean_embedding, prev=None):
    """Suffix for a stage: first slots hold the stage text's real embeddings;
    slots that carried deleted tokens reset to the mean; surviving slots warm
    start from the previous stage's optimized suffix (homotopy continuity)."""
    import torch
    tokens = stage_text_tokens(token_manager, text)
    if len(tokens) > length:
        raise SystemExit(f"stage text has {len(tokens)} tokens > suffix length {length}")
    prev_tokens = None
    if prev is not None:
        prev_tokens = prev["tokens"]
    suffix = mean_embedding.repeat(length, 1).clone()
    for i, token in enumerate(tokens):
        if prev is not None and prev["suffix"] is not None and prev_tokens is not None \
                and i < len(prev_tokens) and prev_tokens[i] == token:
            suffix[i] = prev["suffix"][i].detach()
        else:
            suffix[i] = model.tok_embeddings(
                torch.tensor([token], dtype=torch.long, device="cuda")
            ).detach().float()[0]
    if prev is not None and prev["suffix"] is not None and prev_tokens is not None:
        for i in range(len(tokens), len(prev_tokens)):
            suffix[i] = mean_embedding  # deleted token slots reset
    return suffix, tokens


def collect_on_policy_prefixes(model, transformer, token_sets, prompt_ids, suffix,
                               masks, args, count):
    """Light text-phase rollouts (stop at BOI/EOS); return (generated, boi_step)
    for rollouts that sampled BOI.  The student's own path only."""
    import torch
    text_support, _ = masks
    prefixes = []
    for seed in range(count):
        torch.manual_seed(10000 + seed)
        context = p22.d08.SoftTextContext(
            model, transformer, token_sets, prompt_ids, suffix.detach(),
            args.max_text_steps + 2)
        generated = []
        boi_step = None
        for step in range(args.max_text_steps):
            history = torch.tensor(context.history, dtype=torch.long, device="cuda")
            policy = p22.p12.text_policy(context.logits, history, text_support,
                                         args.text_repetition_penalty, args.text_top_p)
            token = p22.p12.sample_token(policy)
            generated.append(token)
            context.advance(token)
            if token == token_sets.boi_id:
                boi_step = step
                break
            if token == token_sets.eos_id:
                break
        if boi_step is not None:
            prefixes.append((generated[:boi_step + 1], boi_step))
    return prefixes


def mstep_loss(model, transformer, token_sets, prompt_ids, suffix, elite_target,
               support, prefixes, donor_finals, suffix_length, state_weight):
    import torch
    route = p14.differentiable_route_nll(
        model, transformer, token_sets, prompt_ids, suffix, elite_target, support)
    if not prefixes:
        return route
    state_terms = []
    for generated, boi_step in prefixes:
        with torch.no_grad():
            continuation = list(generated) + [token_sets.boi_id]
        embeddings = p22.d08.soft_input_embeddings(
            model, prompt_ids[:-1], token_sets.eot_id, suffix, continuation=continuation)
        _, hidden = p22.p0.exact_forward(model, transformer, embeddings, return_hidden=True)
        boi_pos = len(prompt_ids) + suffix_length + boi_step
        s_boi = hidden[boi_pos].float()
        s_vis1 = hidden[boi_pos + 1].float()
        rel_boi = (s_boi - donor_finals["boi"][0]).square().sum() / donor_finals["boi"][0].square().sum()
        rel_vis1 = (s_vis1 - donor_finals["vis1"][0]).square().sum() / donor_finals["vis1"][0].square().sum()
        state_terms.append(rel_boi + rel_vis1)
    state_loss = torch.stack(state_terms).mean()
    return route + state_weight * state_loss


def gate_eval(model, transformer, token_manager, token_sets, prompt_ids, suffix,
              masks, args, output, clip_similarity, prompt_id, stage_index, seeds):
    """n free full-image rollouts with semantics; returns a summary dict."""
    import torch
    text_support = masks[0]
    records = []
    for seed in seeds:
        torch.manual_seed(20000 + seed)
        record = {"kind": "homotopy_gate_rollout", "evidence_class": "soft-control",
                  "condition": f"stage{stage_index}", "prompt_id": prompt_id, "seed": seed}
        record = p22.rollout_soft(
            model, transformer, token_manager, token_sets, prompt_ids, suffix.detach(),
            masks, args, output, clip_similarity, record)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)
        records.append(record)
    boi_rate = sum(1 for r in records if r.get("boi_sampled")) / len(records)
    completed = [r for r in records if r.get("visual_count", 0) == args.image_length]
    collapse = 0
    for r in completed:
        std = r.get("image_stats", {}).get("stddev", [0, 0, 0])
        if max(std) < 10:
            collapse += 1
    clip_targets = [r["clip_target_similarity"] for r in completed
                    if r.get("clip_target_similarity") is not None]
    clip_distractors = [r["clip_distractor_similarity"] for r in completed
                        if r.get("clip_distractor_similarity") is not None]
    return {
        "boi_rate": boi_rate,
        "completion_rate": len(completed) / len(records),
        "collapse_rate": collapse / len(completed) if completed else None,
        "clip_target_mean": sum(clip_targets) / len(clip_targets) if clip_targets else None,
        "clip_distractor_mean": sum(clip_distractors) / len(clip_distractors) if clip_distractors else None,
    }


def degraded(prev, current):
    """True if the current gate summary degrades beyond tolerance."""
    if prev is None:
        return False
    if current["completion_rate"] < prev["completion_rate"] - 0.25:
        return True
    if prev["clip_target_mean"] is not None and current["clip_target_mean"] is not None:
        if current["clip_target_mean"] < prev["clip_target_mean"] - 0.08:
            return True
    if prev["collapse_rate"] is not None and current["collapse_rate"] is not None:
        if current["collapse_rate"] > prev["collapse_rate"] + 0.25:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--bridge-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["red_apple", "golden_retriever"])
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--m-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--on-policy-rollouts", type=int, default=2)
    parser.add_argument("--gate-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--max-m-retries", type=int, default=2)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                        help="warm-start surviving slots from the previous stage "
                             "(the homotopy); --no-warm-start re-initializes from text "
                             "each stage (cold-start ablation)")
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
    elites = p22.d08.select_elites(
        args.bridge_metrics, p22.d08.PROMPTS, 0.5, [0.75], 1)
    (output / "selected_elites.json").write_text(json.dumps(elites, indent=2) + "\n")
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)

    for prompt_id in args.prompt_ids:
        if prompt_id not in LADDERS:
            raise SystemExit(f"no compression ladder for {prompt_id}")
        teacher_text = p22.TEACHER_PROMPTS[prompt_id]
        student_text = p22.STUDENT_PROMPTS[prompt_id]
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": teacher_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": student_text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])

        # Donor: authorized teacher trajectory + gateway states at BOI/vis1.
        torch.manual_seed(args.donor_seed)
        donor_record = {"condition": "teacher_authorized", "prompt_id": prompt_id,
                        "seed": args.donor_seed}
        donor_record = p22.rollout_hard(
            model, transformer, token_manager, token_sets, teacher_ids, masks,
            args, output, clip_similarity, donor_record)
        append(donor_record)
        if not donor_record.get("boi_sampled"):
            raise SystemExit(f"donor did not sample BOI for {prompt_id}")
        boi_step = donor_record["boi_step"]
        positions = {
            "boi": len(teacher_ids) + boi_step,
            "vis1": len(teacher_ids) + boi_step + 1,
        }
        capture_ids = teacher_ids + donor_record["generated_tokens"][:boi_step + 2]
        _, donor_finals = p22.p12.capture_states(model, transformer, capture_ids, positions, set())

        ladder = LADDERS[prompt_id]
        elite = elites[prompt_id][0]
        prev_stage = None
        prev_summary = None
        stage_index = 0
        while stage_index < len(ladder):
            text = ladder[stage_index]
            tokens = stage_text_tokens(token_manager, text)
            suffix, tokens = stage_suffix(
                model, token_manager, text, args.suffix_length, mean_embedding,
                prev=prev_stage if args.warm_start else None)
            m_steps = args.m_steps
            summary = None
            for retry in range(args.max_m_retries + 1):
                if stage_index > 0:
                    suffix = suffix.detach().requires_grad_(True)
                    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
                    prefixes = collect_on_policy_prefixes(
                        model, transformer, token_sets, student_ids, suffix, masks, args,
                        args.on_policy_rollouts)
                    for step in range(m_steps):
                        loss = mstep_loss(
                            model, transformer, token_sets, student_ids, suffix,
                            elite["target_ids"], support, prefixes, donor_finals,
                            args.suffix_length, args.state_weight)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if not torch.isfinite(suffix.grad).all():
                            raise RuntimeError("non-finite suffix gradient in M-step")
                        optimizer.step()
                        if step % 5 == 0 or step == m_steps - 1:
                            append({"kind": "homotopy_mstep", "prompt_id": prompt_id,
                                    "stage": stage_index, "stage_text": text,
                                    "step": step, "loss": float(loss.detach()),
                                    "on_policy_prefixes": len(prefixes),
                                    "retry": retry})
                    suffix = suffix.detach()
                summary = gate_eval(
                    model, transformer, token_manager, token_sets, student_ids, suffix,
                    masks, args, output, clip_similarity, prompt_id, stage_index,
                    args.gate_seeds)
                append({"kind": "homotopy_stage_summary", "prompt_id": prompt_id,
                        "stage": stage_index, "stage_text": text,
                        "suffix_tokens": len(tokens), "retry": retry, **summary})
                torch.save(
                    {"suffix": suffix.detach().cpu(), "suffix_length": suffix.shape[0],
                     "stage": stage_index, "stage_text": text, "retry": retry},
                    output / f"suffix_{prompt_id}_stage{stage_index}_retry{retry}.pt")
                if degraded(prev_summary, summary) and stage_index > 0:
                    if retry < args.max_m_retries:
                        m_steps *= 2
                        suffix, tokens = stage_suffix(
                            model, token_manager, text, args.suffix_length,
                            mean_embedding, prev=prev_stage if args.warm_start else None)
                        continue
                    append({"kind": "homotopy_floor", "prompt_id": prompt_id,
                            "floor_stage": stage_index, "floor_text": text,
                            "note": "degradation persisted across retries; "
                                    "last stable stage is the compression floor"})
                    stage_index = len(ladder)
                    break
                break
            prev_stage = {"suffix": suffix, "tokens": tokens}
            prev_summary = summary
            stage_index += 1
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
