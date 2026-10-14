#!/usr/bin/env python
"""P1: Anole SOI/structure gateway causal patch ladder (T3 diagnostics).

One vertical slice: one ordinary student prompt, one authorized teacher
trajectory as the state donor, a few fixed seeds.  Intervention ladder:

1. student_none                    ordinary free decoding (control)
2. boundary_patch_final_resid      patch the last prompt position's final
                                   residual with the authorized EOT-position
                                   residual, then free decoding (no forcing)
3. force_boi_only                  force BOI as the first generated token
4. force_boi_patch_final_resid     force BOI + patch final residual at the
                                   BOI position (readout patch)
5. force_boi_patch_late_layers     force BOI + patch inputs to late layers
                                   at the BOI position (causal-tracing patch)
6. force_boi_patch_soi_vis1        force BOI + late-layer patches at the BOI
                                   and first-visual positions

Teacher free-form trajectories are also recorded as the loop-validation
control (native BOI sampling is 64/64 under the official decoder).

All decoding uses one replication of the official public decoder: text mode
(repetition penalty 1.2, temp 1.0, top-p 0.9 over text+EOS+BOI), image mode
(three-branch InstructCFG 3.0/1.2, temp 0.7, top-p 0.9 over image tokens,
forced EOI after 1024 visual IDs).  Every activation/state injection here is
T3 mechanism diagnosis and must never be reported as a T1 native claim.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEACHER_PROMPT = "Draw a red apple on a plain light background."
STUDENT_PROMPT = "Explain in one sentence why apples are a popular fruit."

FORCED_BOI_KINDS = {
    "force_boi_only",
    "force_boi_patch_final_resid",
    "force_boi_patch_late_layers",
    "force_boi_patch_soi_vis1",
}


def build_mask(token_sets, ids):
    import torch
    mask = torch.zeros(token_sets.vocab_size, dtype=torch.bool, device="cuda")
    mask[list(ids)] = True
    return mask


class Branch:
    """No-grad cache state for one decoder branch (official kernels)."""

    def __init__(self, model, transformer, token_ids, max_length,
                 final_resid_pos=None, final_resid=None):
        import torch
        self.model = model
        self.history = list(token_ids)
        self.cache = transformer.make_cache(
            model.args, length=max_length, device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        self.bias = transformer.AttnBias.from_seqlens(
            q_seqlen=[len(token_ids)], kv_seqlen=[len(token_ids)],
            kv_padding=max_length,
        )
        hidden = model.tok_embeddings(
            torch.tensor(token_ids, dtype=torch.long, device="cuda")
        )
        with torch.no_grad():
            for index, layer in enumerate(model.layers):
                hidden = layer(hidden, self.cache[index], self.bias)
            if final_resid is not None:
                hidden = hidden.clone()
                hidden[final_resid_pos:final_resid_pos + 1] = final_resid
            self.logits = model.output(model.norm(hidden)).float()[-1]
        self.bias.q_seqinfo.seqstart.copy_(
            torch.arange(2, dtype=torch.int, device="cuda")
        )
        self.bias.q_seqinfo.max_seqlen = 1
        self.bias.q_seqinfo.seqstart_py = self.bias.q_seqinfo.seqstart.tolist()
        self.bias.k_seqinfo.seqlen.add_(1)

    def advance(self, token, patches=None, final_resid=None):
        """Single-token forward.  patches: {layer_index: (1, dim) donor state}
        applied at the layer input; final_resid replaces the post-layer state
        before the final norm (readout-only patch)."""
        import torch
        hidden = self.model.tok_embeddings(
            torch.tensor([token], dtype=torch.long, device="cuda")
        )
        with torch.no_grad():
            for index, layer in enumerate(self.model.layers):
                if patches and index in patches:
                    hidden = patches[index].clone()
                hidden = layer(hidden, self.cache[index], self.bias)
            if final_resid is not None:
                hidden = final_resid.clone()
            self.logits = self.model.output(self.model.norm(hidden)).float()[0]
        self.bias.k_seqinfo.seqlen.add_(1)
        self.history.append(token)


def capture_states(model, transformer, token_ids, positions, layers):
    """Layer-input states at given layers and final residuals at positions."""
    import torch
    length = len(token_ids)
    cache = transformer.make_cache(
        model.args, length=length, device="cuda",
        dtype=next(model.parameters()).dtype,
    )
    bias = transformer.AttnBias.from_seqlens(
        q_seqlen=[length], kv_seqlen=[length], kv_padding=length
    )
    hidden = model.tok_embeddings(
        torch.tensor(token_ids, dtype=torch.long, device="cuda")
    )
    inputs, finals = {}, {}
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            if index in layers:
                for name, pos in positions.items():
                    inputs[(name, index)] = hidden[pos:pos + 1].clone()
            hidden = layer(hidden, cache[index], bias)
        for name, pos in positions.items():
            finals[name] = hidden[pos:pos + 1].clone()
    return inputs, finals


def text_policy(raw_logits, history, support_mask, penalty, top_p):
    from morph.decoding.processed_policy import (
        apply_common_warpers, apply_repetition_penalty, masked_logprobs,
    )
    policy = masked_logprobs(
        apply_repetition_penalty(raw_logits, history, penalty=penalty),
        support_mask,
    )
    return apply_common_warpers(policy, temperature=1.0, top_p=top_p)


def image_policy(mixed_logits, image_mask, temperature, top_p):
    from morph.decoding.processed_policy import apply_common_warpers, masked_logprobs
    policy = masked_logprobs(mixed_logits, image_mask)
    return apply_common_warpers(policy, temperature=temperature, top_p=top_p)


def sample_token(warped_logprobs):
    import torch
    return int(torch.multinomial(warped_logprobs.exp(), 1).item())


def run_trajectory(model, transformer, token_manager, token_sets, masks, args,
                   kind, prompt_ids, donor, seed, output_dir):
    """One trajectory under one intervention; donor supplies patch states."""
    import torch
    torch.manual_seed(seed)
    t0 = time.time()
    text_support, image_mask = masks
    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    record = {
        "kind": "gateway_patch_ladder",
        "intervention": kind,
        "evidence_class": "T3" if kind in FORCED_BOI_KINDS | {"boundary_patch_final_resid"} else "control",
        "prompt_id": "red_apple",
        "seed": seed,
        "donor": donor["meta"] if donor else None,
        "patched_positions": donor["patched_positions"] if donor else [],
        "patched_layers": args.patch_layers,
        "boi_sampled": False,
        "boi_forced": False,
        "boi_step": None,
    }

    # --- prompt prefill ------------------------------------------------------
    max_length = len(prompt_ids) + args.max_text_steps + args.image_length + args.post_image_text_steps + 8
    boundary = (kind == "boundary_patch_final_resid")
    branch = Branch(
        model, transformer, prompt_ids, max_length,
        final_resid_pos=len(prompt_ids) - 1,
        final_resid=donor["finals"]["eot"] if boundary else None,
    )

    # --- text phase ----------------------------------------------------------
    generated = []
    for step in range(args.max_text_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        if step == 0 and kind in FORCED_BOI_KINDS:
            token, boi_forced = boi_id, True
            if kind == "force_boi_only":
                branch.advance(boi_id)
            elif kind == "force_boi_patch_final_resid":
                branch.advance(boi_id, final_resid=donor["finals"]["boi"])
            else:  # force_boi_patch_late_layers / force_boi_patch_soi_vis1
                branch.advance(boi_id, patches={i: donor["inputs"][("boi", i)] for i in args.patch_layers})
        else:
            policy = text_policy(branch.logits, history, text_support, args.text_repetition_penalty, args.text_top_p)
            token = sample_token(policy)
            boi_forced = False
            branch.advance(token)
        generated.append(token)
        if token == boi_id:
            record.update({"boi_sampled": not boi_forced, "boi_forced": boi_forced, "boi_step": step})
            break
        if token == eos_id:
            break
    if not generated or generated[-1] != boi_id:
        from morph.decoding.anole_segments import analyze_anole_sequence
        structure = analyze_anole_sequence(generated, token_sets)
        record.update({"generated_tokens": generated, "token_count": len(generated),
                       "visual_count": 0, "structure_valid": structure.valid,
                       "structure_error": structure.error,
                       "completed_images": structure.completed_images,
                       "image_path": None, "seconds": round(time.time() - t0, 1)})
        return record

    # --- image phase (three-branch InstructCFG, official replication) --------
    image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
    imgcond_ids = [t for t in prompt_ids if t in image_cond_allowed]
    if not imgcond_ids or imgcond_ids[-1] != boi_id:
        imgcond_ids.append(boi_id)
    uncond_ids = [token_sets.bos_id, boi_id]
    aux_length = args.image_length + args.post_image_text_steps + 8
    imgcond = Branch(model, transformer, imgcond_ids, aux_length)
    uncond = Branch(model, transformer, uncond_ids, aux_length)

    visual = []
    for vstep in range(args.image_length):
        mixed = (uncond.logits
                 + args.guidance_image * (imgcond.logits - uncond.logits)
                 + args.guidance_text * (branch.logits - imgcond.logits))
        policy = image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
        token = sample_token(policy)
        visual.append(token)
        patches = None
        if vstep == 0 and kind == "force_boi_patch_soi_vis1":
            patches = {i: donor["inputs"][("vis1", i)] for i in args.patch_layers}
        branch.advance(token, patches=patches)
        imgcond.advance(token)
        uncond.advance(token)
    branch.advance(eoi_id)  # the public decoder's forced EOI, processed as a real token
    generated.extend(visual)
    generated.append(eoi_id)

    # --- post-image text -----------------------------------------------------
    post = []
    for _ in range(args.post_image_text_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = text_policy(branch.logits, history, text_support, args.text_repetition_penalty, args.text_top_p)
        token = sample_token(policy)
        post.append(token)
        generated.append(token)
        if token == eos_id:
            break
        branch.advance(token)

    # --- artifacts -----------------------------------------------------------
    image_path = None
    image_stats = None
    if visual:
        from PIL import ImageStat
        image = token_manager.pil_from_bpe_tokens(
            torch.tensor(visual, dtype=torch.long, device="cuda")
        )
        image_path = f"images/{kind}_seed{seed}.png"
        image.save(output_dir / image_path)
        rgb = image.convert("RGB")
        s = ImageStat.Stat(rgb)
        image_stats = {"mean": [round(v, 2) for v in s.mean],
                       "stddev": [round(v, 2) for v in s.stddev],
                       "extrema": rgb.getextrema()}
    post_text = ""
    if post:
        post_text = token_manager.decode_text(torch.tensor([post], dtype=torch.long))[0]
    from morph.decoding.anole_segments import analyze_anole_sequence
    structure = analyze_anole_sequence(generated, token_sets)
    record.update({
        "generated_tokens": generated,
        "token_count": len(generated),
        "visual_count": len(visual),
        "eoi_forced": bool(visual),
        "structure_valid": structure.valid,
        "structure_error": structure.error,
        "completed_images": structure.completed_images,
        "post_image_text": post_text,
        "first_visual_ids": visual[:4],
        "image_path": image_path,
        "image_stats": image_stats,
        "semantics": "placeholder_manual_inspection",
        "seconds": round(time.time() - t0, 1),
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--patch-layers", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--max-text-steps", type=int, default=128)
    parser.add_argument("--post-image-text-steps", type=int, default=32)
    parser.add_argument("--image-length", type=int, default=1024)
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
    tokenizer_path = Path(args.model_path) / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    token_manager = TokenManager(
        str(tokenizer_path), str(Path(args.model_path) / "tokenizer/vqgan.yaml"),
        str(Path(args.model_path) / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    model = load_model(str(Path(args.model_path) / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    teacher_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": TEACHER_PROMPT},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    student_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": STUDENT_PROMPT},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    masks = (
        build_mask(token_sets, token_sets.text_step_support),
        build_mask(token_sets, token_sets.image_token_ids),
    )
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    # --- donor: authorized teacher trajectory + captured gateway states ------
    donor_record = run_trajectory(
        model, transformer, token_manager, token_sets, masks, args,
        "teacher_freeform_control", teacher_ids, None, args.donor_seed, output,
    )
    append(donor_record)
    if not donor_record.get("boi_sampled"):
        raise SystemExit("donor teacher trajectory did not sample BOI natively")
    boi_step = donor_record["boi_step"]
    positions = {
        "eot": len(teacher_ids) - 1,
        "boi": len(teacher_ids) + boi_step,
        "vis1": len(teacher_ids) + boi_step + 1,
    }
    capture_ids = teacher_ids + donor_record["generated_tokens"][:boi_step + 2]
    donor_inputs, donor_finals = capture_states(
        model, transformer, capture_ids, positions, set(args.patch_layers)
    )
    donor = {
        "meta": {"prompt_id": "red_apple_teacher", "seed": args.donor_seed,
                 "boi_step": boi_step, "capture_positions": positions},
        "inputs": donor_inputs,
        "finals": donor_finals,
        "patched_positions": [],
    }

    # --- intervention ladder ---------------------------------------------------
    for kind in ["student_none", "boundary_patch_final_resid",
                 "force_boi_only", "force_boi_patch_final_resid",
                 "force_boi_patch_late_layers", "force_boi_patch_soi_vis1"]:
        if kind in FORCED_BOI_KINDS:
            donor["patched_positions"] = ["boi"] if kind != "force_boi_patch_soi_vis1" else ["boi", "vis1"]
        else:
            donor["patched_positions"] = ["eot"] if kind == "boundary_patch_final_resid" else []
        for seed in args.seeds:
            record = run_trajectory(
                model, transformer, token_manager, token_sets, masks, args,
                kind, student_ids, donor, seed, output,
            )
            append(record)

    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
