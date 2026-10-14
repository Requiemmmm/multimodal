#!/usr/bin/env python
"""N1 stage-0 probe: does the authorized instruction still work when it is
appended to the ordinary student prompt? (run 22)

The context-side homotopy compresses the authorized image instruction into a
suffix, starting from "the instruction IS the suffix" and never cold-starting
the visual plan.  Stage 0 asks whether that starting point is valid: under
the ordinary prompt "Explain in one sentence why apples are popular.", a
suffix carrying the full instruction "Draw a red apple on a plain light
background." must still produce completed, semantically correct images.

Four conditions per prompt (n seeds each):
1. teacher_authorized     the authorized prompt alone (reference images);
2. student_hard_instruction  ordinary prompt + instruction as real text;
3. student_soft_instruction  ordinary prompt + soft suffix initialized from
                          the instruction's real token embeddings (16 slots,
                          mean-embedding padding) -- the homotopy's start;
4. student_soft_empty     ordinary prompt + empty soft suffix (mean + noise)
                          -- the soft-control baseline (expect ~0 BOI).

Every completed rollout is evaluated with the run 16 protocol: image stats
(collapse detector), CLIP target vs distractor, Self-SpectraReward caption,
plus gateway-state distances (BOI and first-visual positions) of the
student's OWN path against the authorized donor trajectory.

Everything here is soft-control / T2 evidence; no hard-text claim.
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

from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs

TEACHER_PROMPTS = {
    "red_apple": "Draw a red apple on a plain light background.",
    "golden_retriever": "Draw a golden retriever standing on a plain light background.",
}
STUDENT_PROMPTS = {
    "red_apple": "Explain in one sentence why apples are a popular fruit.",
    "golden_retriever": "Explain in one sentence why golden retrievers are suitable family pets.",
    "green_frog": "Explain in one sentence why frogs are green.",
    "strawberry": "Explain in one sentence why strawberries are red.",
    "sky_blue": "Explain in one sentence why the sky is blue.",
    "tomato": "Explain in one sentence why tomatoes are red.",
    "cherry": "Explain in one sentence why cherries are small and sweet.",
    "labrador": "Explain in one sentence why Labrador retrievers are popular guide dogs.",
    "husky": "Explain in one sentence why huskies have thick fur.",
    "truck": "Explain in one sentence why trucks are used for transporting goods.",
    "mug": "Explain in one sentence why ceramic mugs are common drinkware.",
    "sushi": "Explain in one sentence why sushi is a popular food.",
    "kimchi": "Explain in one sentence why kimchi is a traditional Korean food.",
    "ethiopia_coffee": "Explain in one sentence why Ethiopia is known for coffee.",
    "kenya_marathon": "Explain in one sentence why Kenya produces many marathon runners.",
}
TARGET_CAPTIONS = {
    "red_apple": "a photo of a red apple",
    "golden_retriever": "a photo of a golden retriever dog",
    "green_frog": "a photo of a green frog",
    "strawberry": "a photo of a red strawberry",
    "sky_blue": "a photo of a blue sky",
    "tomato": "a photo of a red tomato",
    "cherry": "a photo of a cherry",
    "labrador": "a photo of a labrador retriever",
    "husky": "a photo of a husky dog",
    "truck": "a photo of a truck",
    "mug": "a photo of a ceramic mug",
    "sushi": "a photo of sushi",
    "kimchi": "a photo of kimchi",
    "ethiopia_coffee": "a photo of coffee beans",
    "kenya_marathon": "a photo of a marathon runner",
}
DISTRACTOR_CAPTION = "a photo of a car"
QUESTION = "Answer in one short phrase: what is in this image?"


def instruction_suffix(model, token_manager, text, length):
    """Soft suffix initialized from the instruction's real token embeddings,
    padded to `length` slots with the embedding mean."""
    import torch
    ids = token_manager.tokens_from_ui([
        {"type": "text", "value": text},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    text_ids = ids[1:-1]
    if len(text_ids) > length:
        raise SystemExit(f"instruction has {len(text_ids)} tokens > suffix length {length}")
    embedded = model.tok_embeddings(
        torch.tensor(text_ids, dtype=torch.long, device="cuda")).detach().float()
    mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
    padding = mean_embedding.repeat(length - len(text_ids), 1)
    return torch.cat([embedded, padding], dim=0)


def reverse_understanding(token_manager, model, transformer, token_sets, visual_ids,
                          question, max_steps, penalty, support_mask):
    import torch
    boi_id, eoi_id, eos_id = token_sets.boi_id, token_sets.eoi_id, token_sets.eos_id
    ids = token_manager.tokens_from_ui([
        {"type": "text", "value": question},
        {"type": "ids", "value": [boi_id, *visual_ids, eoi_id]},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    branch = p12.Branch(model, transformer, ids, len(ids) + max_steps + 8)
    out = []
    for _ in range(max_steps):
        history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
        policy = masked_logprobs(
            apply_repetition_penalty(branch.logits, history, penalty=penalty),
            support_mask)
        token = int(torch.argmax(policy))
        out.append(token)
        if token == eos_id:
            break
        branch.advance(token)
    return token_manager.decode_text(torch.tensor([out], dtype=torch.long))[0]


def state_distances(a, b):
    import torch
    relative = float((a - b).square().sum() / b.square().sum())
    cosine = float(
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return relative, cosine


def rollout_hard(model, transformer, token_manager, token_sets, prompt_ids, masks,
                 args, output, clip_similarity, record):
    """Hard-token rollout through the Branch replication (run 12 protocol)."""
    import torch
    from morph.decoding.anole_segments import analyze_anole_sequence
    text_support, image_mask = masks
    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    max_length = len(prompt_ids) + args.max_text_steps + args.image_length + args.post_image_text_steps + 8
    context = p12.Branch(model, transformer, prompt_ids, max_length)
    generated = []
    boi_step = None
    for step in range(args.max_text_steps):
        history = torch.tensor(context.history, dtype=torch.long, device="cuda")
        policy = p12.text_policy(context.logits, history, text_support,
                                 args.text_repetition_penalty, args.text_top_p)
        token = p12.sample_token(policy)
        generated.append(token)
        context.advance(token)
        if token == boi_id:
            boi_step = step
            break
        if token == eos_id:
            break
    record.update({"boi_sampled": boi_step is not None, "boi_step": boi_step})
    if boi_step is None:
        structure = analyze_anole_sequence(generated, token_sets)
        record.update({"visual_count": 0, "structure_valid": structure.valid,
                       "image_path": None})
        return record

    image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
    imgcond_ids = [t for t in prompt_ids if t in image_cond_allowed]
    if not imgcond_ids or imgcond_ids[-1] != boi_id:
        imgcond_ids.append(boi_id)
    uncond_ids = [token_sets.bos_id, boi_id]
    aux_length = args.image_length + args.post_image_text_steps + 8
    imgcond = p12.Branch(model, transformer, imgcond_ids, aux_length)
    uncond = p12.Branch(model, transformer, uncond_ids, aux_length)
    visual = []
    for _ in range(args.image_length):
        mixed = (uncond.logits
                 + args.guidance_image * (imgcond.logits - uncond.logits)
                 + args.guidance_text * (context.logits - imgcond.logits))
        policy = p12.image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
        token = p12.sample_token(policy)
        visual.append(token)
        context.advance(token)
        imgcond.advance(token)
        uncond.advance(token)
    context.advance(eoi_id)
    generated.extend(visual)
    generated.append(eoi_id)
    post = []
    for _ in range(args.post_image_text_steps):
        history = torch.tensor(context.history, dtype=torch.long, device="cuda")
        policy = p12.text_policy(context.logits, history, text_support,
                                 args.text_repetition_penalty, args.text_top_p)
        token = p12.sample_token(policy)
        post.append(token)
        generated.append(token)
        if token == eos_id:
            break
        context.advance(token)
    return finish_rollout(model, transformer, token_manager, token_sets, generated, visual, post,
                          args, output, clip_similarity, text_support, record)


def rollout_soft(model, transformer, token_manager, token_sets, prompt_ids, suffix,
                 masks, args, output, clip_similarity, record):
    """Soft-suffix rollout through SoftTextContext (run 16 protocol)."""
    import torch
    from morph.decoding.anole_segments import analyze_anole_sequence
    text_support, image_mask = masks
    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    capacity = args.max_text_steps + args.image_length + args.post_image_text_steps + 8
    context = d08.SoftTextContext(model, transformer, token_sets, prompt_ids, suffix, capacity)
    generated = []
    boi_step = None
    for step in range(args.max_text_steps):
        history = torch.tensor(context.history, dtype=torch.long, device="cuda")
        policy = p12.text_policy(context.logits, history, text_support,
                                 args.text_repetition_penalty, args.text_top_p)
        token = p12.sample_token(policy)
        generated.append(token)
        context.advance(token)
        if token == boi_id:
            boi_step = step
            break
        if token == eos_id:
            break
    record.update({"boi_sampled": boi_step is not None, "boi_step": boi_step})
    if boi_step is None:
        structure = analyze_anole_sequence(generated, token_sets)
        record.update({"visual_count": 0, "structure_valid": structure.valid,
                       "image_path": None})
        return record

    image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
    imgcond_ids = [t for t in prompt_ids if t in image_cond_allowed]
    if not imgcond_ids or imgcond_ids[-1] != boi_id:
        imgcond_ids.append(boi_id)
    uncond_ids = [token_sets.bos_id, boi_id]
    aux_length = args.image_length + args.post_image_text_steps + 8
    imgcond = p12.Branch(model, transformer, imgcond_ids, aux_length)
    uncond = p12.Branch(model, transformer, uncond_ids, aux_length)
    visual = []
    for _ in range(args.image_length):
        mixed = (uncond.logits
                 + args.guidance_image * (imgcond.logits - uncond.logits)
                 + args.guidance_text * (context.logits - imgcond.logits))
        policy = p12.image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
        token = p12.sample_token(policy)
        visual.append(token)
        context.advance(token)
        imgcond.advance(token)
        uncond.advance(token)
    context.advance(eoi_id)
    generated.extend(visual)
    generated.append(eoi_id)
    post = []
    for _ in range(args.post_image_text_steps):
        history = torch.tensor(context.history, dtype=torch.long, device="cuda")
        policy = p12.text_policy(context.logits, history, text_support,
                                 args.text_repetition_penalty, args.text_top_p)
        token = p12.sample_token(policy)
        post.append(token)
        generated.append(token)
        if token == eos_id:
            break
        context.advance(token)
    return finish_rollout(model, transformer, token_manager, token_sets, generated, visual, post,
                          args, output, clip_similarity, text_support, record)


def finish_rollout(model, transformer, token_manager, token_sets, generated, visual, post,
                   args, output, clip_similarity, text_support, record):
    import torch
    from PIL import ImageStat
    from morph.decoding.anole_segments import analyze_anole_sequence
    image_path = None
    image_stats = None
    caption = None
    clip_target = None
    clip_distractor = None
    if visual:
        image = token_manager.pil_from_bpe_tokens(
            torch.tensor(visual, dtype=torch.long, device="cuda"))
        image_path = f"images/{record['condition']}_{record['prompt_id']}_seed{record['seed']}.png"
        image.save(output / image_path)
        s = ImageStat.Stat(image.convert("RGB"))
        image_stats = {"mean": [round(v, 2) for v in s.mean],
                       "stddev": [round(v, 2) for v in s.stddev]}
        caption = reverse_understanding(
            token_manager, model, transformer, token_sets, visual, QUESTION,
            args.caption_steps, args.text_repetition_penalty, text_support)
        if clip_similarity is not None:
            clip_target = clip_similarity(image, TARGET_CAPTIONS[record["prompt_id"]])
            clip_distractor = clip_similarity(image, DISTRACTOR_CAPTION)
    post_text = ""
    if post:
        post_text = token_manager.decode_text(torch.tensor([post], dtype=torch.long))[0]
    structure = analyze_anole_sequence(generated, token_sets)
    record.update({
        "generated_tokens": generated,
        "visual_count": len(visual),
        "structure_valid": structure.valid,
        "structure_error": structure.error,
        "completed_images": structure.completed_images,
        "image_path": image_path,
        "image_stats": image_stats,
        "reverse_understanding": caption,
        "clip_target_similarity": clip_target,
        "clip_distractor_similarity": clip_distractor,
        "post_image_text": post_text,
        "first_visual_ids": visual[:4],
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-ids", nargs="+", default=["red_apple", "golden_retriever"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--suffix-length", type=int, default=16)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=0)
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
        p12.build_mask(token_sets, token_sets.text_step_support),
        p12.build_mask(token_sets, token_sets.image_token_ids),
    )
    text_support = masks[0]
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    for prompt_id in args.prompt_ids:
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": TEACHER_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        student_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": STUDENT_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        hard_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": STUDENT_PROMPTS[prompt_id] + " " + TEACHER_PROMPTS[prompt_id]},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        soft_instruction = instruction_suffix(
            model, token_manager, TEACHER_PROMPTS[prompt_id], args.suffix_length)
        torch.manual_seed(args.init_seed)
        mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
        soft_empty = (mean_embedding.repeat(args.suffix_length, 1)
                      + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda"))

        # Donor: authorized teacher trajectory + gateway states at BOI/vis1.
        torch.manual_seed(args.donor_seed)
        donor_record = {"condition": "teacher_authorized", "prompt_id": prompt_id,
                        "seed": args.donor_seed}
        donor_record = rollout_hard(
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
        _, donor_finals = p12.capture_states(model, transformer, capture_ids, positions, set())

        conditions = [
            ("teacher_authorized", "hard", teacher_ids, None),
            ("student_hard_instruction", "hard", hard_ids, None),
            ("student_soft_instruction", "soft", student_ids, soft_instruction),
            ("student_soft_empty", "soft", student_ids, soft_empty),
        ]
        for condition, kind, ids, suffix in conditions:
            if condition == "teacher_authorized":
                seeds = [seed for seed in args.seeds if seed != args.donor_seed]
            else:
                seeds = args.seeds
            for seed in seeds:
                torch.manual_seed(seed)
                t0 = time.time()
                record = {"kind": "homotopy_stage0_probe", "evidence_class": "soft-control",
                          "condition": condition, "prompt_id": prompt_id, "seed": seed}
                if kind == "hard":
                    record = rollout_hard(
                        model, transformer, token_manager, token_sets, ids, masks,
                        args, output, clip_similarity, record)
                    if record.get("boi_sampled"):
                        full_ids = ids + record["generated_tokens"][:record["boi_step"] + 2]
                        positions = {"boi": len(ids) + record["boi_step"],
                                     "vis1": len(ids) + record["boi_step"] + 1}
                        _, finals = p12.capture_states(model, transformer, full_ids, positions, set())
                        rel_boi, cos_boi = state_distances(finals["boi"][0], donor_finals["boi"][0])
                        rel_vis1, cos_vis1 = state_distances(finals["vis1"][0], donor_finals["vis1"][0])
                        record.update({"gateway_boi_rel_l2": rel_boi, "gateway_boi_cosine": cos_boi,
                                       "gateway_vis1_rel_l2": rel_vis1, "gateway_vis1_cosine": cos_vis1})
                else:
                    record = rollout_soft(
                        model, transformer, token_manager, token_sets, ids, suffix, masks,
                        args, output, clip_similarity, record)
                    if record.get("boi_sampled"):
                        with torch.no_grad():
                            embeddings = d08.soft_input_embeddings(
                                model, ids[:-1], token_sets.eot_id, suffix.detach(),
                                continuation=record["generated_tokens"][:record["boi_step"] + 2])
                            _, hidden = p0.exact_forward(model, transformer, embeddings, return_hidden=True)
                            boi_pos = len(ids) + args.suffix_length + record["boi_step"]
                            vis1_pos = boi_pos + 1
                            s_boi = hidden[boi_pos].float()
                            s_vis1 = hidden[vis1_pos].float()
                        rel_boi, cos_boi = state_distances(s_boi, donor_finals["boi"][0])
                        rel_vis1, cos_vis1 = state_distances(s_vis1, donor_finals["vis1"][0])
                        record.update({"gateway_boi_rel_l2": rel_boi, "gateway_boi_cosine": cos_boi,
                                       "gateway_vis1_rel_l2": rel_vis1, "gateway_vis1_cosine": cos_vis1})
                record["seconds"] = round(time.time() - t0, 1)
                append(record)
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
