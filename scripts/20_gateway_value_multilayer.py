#!/usr/bin/env python
"""Gateway-Value M-step slice 3 (config 04), one prompt: red_apple.

Slice 2 (script 19) achieved direct BOI at the first sampled position (8/8 +
12/12 retention), so the training path equals the deployment path, and the
BOI-position final residual matched to ~99% -- yet free rollouts still
collapsed.  Slice 3 escalates along the two P1/P2-validated axes:
- state matching moves from the final residual alone to the inputs of late
  layers 28-31 at the BOI position (P1's strongest condition), so the patched
  K/V at those layers also carry the donor plan into the trajectory;
- the visual plan seed grows from 8 to 32 donor visual tokens (P2: pixel
  commitment needs k~=256-512; 32 is the next affordable rung).

Objective (exact-forward Adam):
  L = -4 * log p(BOI | first sampled position)
    + 0.5 * mean_l ||h_l(w) - h_l_donor||^2 / ||h_l_donor||^2   l in {28..31}
    + 0.5 * cfg_mixed_nll(first 32 donor visual tokens)

Evaluation unchanged: semantics + CLIP on red_apple, route retention on the
other three prompts.  Soft-control evidence only.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
s16 = _load("content16", str(PROJECT_ROOT / "scripts/16_eval_soft_content_control.py"))

from morph.decoding.processed_policy import (
    apply_common_warpers, apply_repetition_penalty, masked_logprobs,
)


def cfg_mixed_visual_nll(model, transformer, token_sets, prompt_ids, suffix,
                         donor_visual, image_mask):
    """Teacher-forced NLL of the first visual tokens under the deployed mixed
    policy.  For ordinary prompts imgcond == uncond, so mixed = 3*full - 2*uncond."""
    import torch
    boi_id = token_sets.boi_id
    prefix = donor_visual[:-1]
    full_emb = d08.soft_input_embeddings(
        model, prompt_ids[:-1], token_sets.eot_id, suffix,
        continuation=[boi_id, *prefix])
    full_logits = p0.exact_forward(model, transformer, full_emb)
    uncond_ids = torch.tensor(
        [token_sets.bos_id, boi_id, *prefix], device="cuda")
    with torch.no_grad():
        uncond_emb = model.tok_embeddings(uncond_ids)
        uncond_logits = p0.exact_forward(model, transformer, uncond_emb)
    first = len(prompt_ids) + suffix.shape[0]  # position predicting vis[0]
    losses = []
    for t, target in enumerate(donor_visual):
        mixed = 3.0 * full_logits[first + t] - 2.0 * uncond_logits[1 + t]
        policy = masked_logprobs(mixed, image_mask)
        losses.append(-policy[target])
    return torch.stack(losses).mean()


def exact_forward_snapshots(model, transformer, embeddings, layers):
    """Exact official forward returning logits, final residual, and the
    hidden states at the inputs of the requested layers (differentiable)."""
    length = embeddings.shape[0]
    cache = transformer.make_cache(
        model.args, length=length, device="cuda", dtype=embeddings.dtype)
    padded, causal = p0.make_biases(transformer, length)
    hidden = embeddings.to(dtype=next(model.parameters()).dtype)
    snapshots = {}
    for index, layer in enumerate(model.layers):
        if index in layers:
            snapshots[index] = hidden
        hidden = p0.differentiable_layer_forward(
            layer, hidden, cache[index], padded, causal)
    logits = model.output(
        p0.RMSNormWithGrad.apply(hidden, model.norm.weight, model.norm.eps)).float()
    return logits, hidden, snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--visual-weight", type=float, default=0.5)
    parser.add_argument("--visual-prefix", type=int, default=32)
    parser.add_argument("--state-layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--retention-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--clip-dir", default=os.environ.get("CLIP_DIR", "") or None)
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
    support = d08.legal_mask(token_sets)
    image_mask = p12.build_mask(token_sets, token_sets.image_token_ids)
    student_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": "Explain in one sentence why apples are a popular fruit."},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    teacher_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": "Draw a red apple on a plain light background."},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])

    # --- donor: authorized trajectory + gateway state + visual prefix ---------
    args12 = SimpleNamespace(
        max_text_steps=128, image_length=1024, post_image_text_steps=32,
        guidance_text=3.0, guidance_image=1.2, image_temp=0.7, image_top_p=0.9,
        text_repetition_penalty=1.2, text_top_p=0.9, patch_layers=[28, 29, 30, 31])
    masks = (support, image_mask)
    (output / "donor_images" / "images").mkdir(parents=True)
    donor_record = p12.run_trajectory(
        model, transformer, token_manager, token_sets, masks, args12,
        "teacher_freeform_control", teacher_ids, None, args.donor_seed,
        output / "donor_images")
    boi_step = donor_record["boi_step"]
    donor_generated = donor_record["generated_tokens"]
    capture_ids = teacher_ids + donor_generated[:boi_step + 1]
    donor_inputs, finals = p12.capture_states(
        model, transformer, capture_ids, {"boi": len(capture_ids) - 1},
        set(args.state_layers))
    s_donor = finals["boi"].detach().float()
    donor_layers = {l: donor_inputs[("boi", l)].detach().float()
                    for l in args.state_layers}
    donor_visual = donor_generated[boi_step + 1: boi_step + 1 + args.visual_prefix]
    (output / "donor.json").write_text(json.dumps({
        "boi_step": boi_step, "donor_visual_prefix": donor_visual}, indent=2) + "\n")

    # --- M-step ---------------------------------------------------------------
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
        vis_emb = d08.soft_input_embeddings(
            model, student_ids[:-1], token_sets.eot_id, suffix,
            continuation=[token_sets.boi_id, *donor_visual[:-1]])
        full_logits, hidden, snapshots = exact_forward_snapshots(
            model, transformer, vis_emb, set(args.state_layers))
        # route: BOI must be the top token at the first sampled position
        last_suffix_pos = len(student_ids) + args.suffix_length - 1
        history = torch.tensor(student_ids, dtype=torch.long, device="cuda")
        boi_policy = masked_logprobs(
            apply_repetition_penalty(full_logits[last_suffix_pos], history, penalty=1.2),
            support)
        route_loss = -4.0 * boi_policy[token_sets.boi_id]
        boi_pos = len(student_ids) + args.suffix_length
        state_terms = [(snapshots[l][boi_pos:boi_pos + 1].float() - donor_layers[l])
                       .square().sum() / donor_layers[l].square().sum()
                       for l in args.state_layers]
        state_terms.append(
            (hidden[boi_pos:boi_pos + 1].float() - s_donor).square().sum()
            / s_donor.square().sum())
        state_loss = torch.stack(state_terms).mean()
        visual_loss = cfg_mixed_visual_nll(
            model, transformer, token_sets, student_ids, suffix, donor_visual, image_mask)
        loss = route_loss + args.state_weight * state_loss + args.visual_weight * visual_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(suffix.grad.norm())
        if not torch.isfinite(suffix.grad).all():
            raise RuntimeError("non-finite suffix gradient; aborting")
        optimizer.step()
        append({
            "kind": "gateway_value_multilayer_mstep", "step": step,
            "total_loss": float(loss.detach()),
            "boi_route_loss": float(route_loss.detach()),
            "relative_state_loss": float(state_loss.detach()),
            "visual_nll": float(visual_loss.detach()),
            "grad_norm": grad_norm,
            "seconds": round(time.time() - t0, 2),
        })

    # --- evaluation: full-image rollouts with semantics on red_apple ----------
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

    for seed in args.eval_seeds:
        torch.manual_seed(seed)
        t0 = time.time()
        record = {"kind": "gateway_value_full_image_rollout", "evidence_class": "soft-control",
                  "prompt_id": "red_apple", "seed": seed}
        context = d08.SoftTextContext(
            model, transformer, token_sets, student_ids, suffix, 128 + 1024 + 32 + 8)
        sampled, boi_step = [], None
        for step in range(128):
            history = torch.tensor(context.history, dtype=torch.long, device="cuda")
            policy = masked_logprobs(
                apply_repetition_penalty(context.logits, history, penalty=1.2), support)
            warped = apply_common_warpers(policy, top_p=0.9)
            token = int(torch.multinomial(warped.exp(), 1).item())
            sampled.append(token)
            if token == token_sets.boi_id:
                boi_step = step
                break
            if token == token_sets.eos_id:
                break
            context.advance(token)
        record["boi_sampled"] = boi_step is not None
        record["boi_step"] = boi_step
        if boi_step is None:
            record.update({"visual_count": 0, "image_path": None,
                           "seconds": round(time.time() - t0, 1)})
            append(record)
            continue
        image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, token_sets.boi_id, token_sets.eoi_id}
        imgcond_ids = [t for t in student_ids if t in image_cond_allowed]
        if not imgcond_ids or imgcond_ids[-1] != token_sets.boi_id:
            imgcond_ids.append(token_sets.boi_id)
        uncond_ids = [token_sets.bos_id, token_sets.boi_id]
        aux_length = 1024 + 32 + 8
        imgcond = p12.Branch(model, transformer, imgcond_ids, aux_length)
        uncond = p12.Branch(model, transformer, uncond_ids, aux_length)
        visual = []
        for _ in range(1024):
            mixed = (uncond.logits
                     + 1.2 * (imgcond.logits - uncond.logits)
                     + 3.0 * (context.logits - imgcond.logits))
            policy = p12.image_policy(mixed, image_mask, 0.7, 0.9)
            token = p12.sample_token(policy)
            visual.append(token)
            context.advance(token)
            imgcond.advance(token)
            uncond.advance(token)
        context.advance(token_sets.eoi_id)
        sampled.extend(visual)
        sampled.append(token_sets.eoi_id)
        from PIL import ImageStat
        image = token_manager.pil_from_bpe_tokens(
            torch.tensor(visual, dtype=torch.long, device="cuda"))
        image_path = f"images/red_apple_seed{seed}.png"
        image.save(output / image_path)
        s = ImageStat.Stat(image.convert("RGB"))
        caption = s16.reverse_understanding(
            token_manager, model, transformer, token_sets, visual, s16.QUESTION,
            48, 1.2, support)
        record.update({
            "sampled_tokens": sampled,
            "visual_count": len(visual),
            "image_path": image_path,
            "image_stats": {"mean": [round(v, 2) for v in s.mean],
                            "stddev": [round(v, 2) for v in s.stddev]},
            "reverse_understanding": caption,
            "clip_target_similarity": clip_similarity(image, s16.TARGET_CAPTIONS["red_apple"]) if clip_model else None,
            "clip_distractor_similarity": clip_similarity(image, s16.DISTRACTOR_CAPTION) if clip_model else None,
            "seconds": round(time.time() - t0, 1),
        })
        append(record)

    # --- route retention on the other three prompts ---------------------------
    retention_prompt_ids = {
        prompt_id: token_manager.tokens_from_ui([
            {"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}
        ])
        for prompt_id, prompt in d08.PROMPTS.items() if prompt_id != "red_apple"
    }
    retention_records = d08.evaluate_student_only(
        model, transformer, token_sets, retention_prompt_ids,
        suffix, args.retention_seeds, 128)
    for record in retention_records:
        record["phase"] = "gateway_value_retention_check"
        append(record)
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length},
               output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
