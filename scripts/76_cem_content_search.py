#!/usr/bin/env python
"""Behaviour-level (CEM) search for hard-suffix content control.

Every other content arm in this project optimised a *token-level proxy*: the
teacher's visual tokens, scored teacher-forced or on-policy.  Run 117 showed why
that is the wrong shape of target -- 14 CLIP-verified apple images start with
five different visual tokens, so no single token is the apple token -- and run
118 confirmed it by improving that proxy more than any other arm and still
producing 0 apples in 8 images.

This script never mentions a teacher token.  It evolves a population of discrete
suffixes against a reward computed from the *rendered image*:

    reward = w_boi * P(model samples BOI)
           + w_valid * P(completed a structurally valid image)
           + w_clip * mean(CLIP(image, "a photo of a red apple")
                         - CLIP(image, "a photo of a strawberry"))

CLIP supplies the dense shaping signal (readback alone is flat at 0/70, which
would make the search blind), while readback stays the *independent* verdict --
per the project rule that a model's own readback may not be both the only
training reward and the only final evidence.  Readback is therefore computed
after the search, not inside the loop.

Selection is elitist CEM: keep the top `--elite-frac`, mutate the survivors.
Every candidate of every generation is logged, not just the elites, so the
search cannot be reported selectively.
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--init-projection", required=True)
    parser.add_argument("--prompt-map", required=True)
    parser.add_argument("--clip-dir",
                        default=os.environ.get("CLIP_DIR", "") or None)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seeds-per-candidate", type=int, default=4)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--w-boi", type=float, default=1.0)
    parser.add_argument("--w-valid", type=float, default=2.0)
    parser.add_argument("--w-clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    (output / "images").mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from chameleon.inference.chameleon import ChameleonInferenceModel, Options, TokenManager
    from morph.adapters.anole import load_anole_token_sets

    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    model = ChameleonInferenceModel(
        str(model_path / "models/7b"), str(tokenizer_path),
        str(model_path / "tokenizer/vqgan.yaml"), str(model_path / "tokenizer/vqgan.ckpt"),
    )
    text_ids = sorted(int(t) for t in token_sets.text_token_ids)

    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(args.clip_dir).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_dir)
    clip_captions = ["a photo of a red apple", "a photo of a strawberry"]

    prompts = json.loads(Path(args.prompt_map).read_text())
    prompt_id = next(iter(prompts))
    prompt_text = prompts[prompt_id]
    init_ids = [int(t) for t in
                json.loads(Path(args.init_projection).read_text())["token_ids"]]
    suffix_len = len(init_ids)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    def clip_margin(image):
        inputs = clip_proc(text=clip_captions, images=image, return_tensors="pt",
                           padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            fi = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            ft = clip_model.get_text_features(input_ids=inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])
        sims = torch.nn.functional.cosine_similarity(fi, ft).tolist()
        return sims[0] - sims[1]

    def evaluate(suffix_ids, tag):
        """Roll out the official decoder and score the rendered images."""
        boi_hits, valid, margins, images = 0, 0, [], []
        for seed in range(args.seeds_per_candidate):
            options = Options(max_gen_len=128)
            options.seed = seed
            generated = model.generate(
                prompt_ui=[
                    {"type": "text", "value": prompt_text},
                    {"type": "ids", "value": suffix_ids},
                    {"type": "sentinel", "value": "<END-OF-TURN>"},
                ],
                options=options,
            )
            raw = generated[0].tolist() if hasattr(generated[0], "tolist") \
                else list(generated[0])
            if token_sets.boi_id not in raw:
                continue
            boi_hits += 1
            start = raw.index(token_sets.boi_id) + 1
            visual = raw[start:start + 1024]
            if len(visual) < 1024 or token_sets.eoi_id in visual:
                continue
            valid += 1
            image = model.decode_image(
                torch.tensor([visual], dtype=torch.long, device="cuda"))[0]
            path = output / "images" / f"{tag}_seed{seed}.png"
            image.save(path)
            margin = clip_margin(image)
            margins.append(margin)
            images.append({"seed": seed, "clip_margin": round(margin, 4),
                           "image": path.name})
        n = args.seeds_per_candidate
        boi_rate, valid_rate = boi_hits / n, valid / n
        clip_mean = sum(margins) / len(margins) if margins else 0.0
        reward = (args.w_boi * boi_rate + args.w_valid * valid_rate
                  + args.w_clip * clip_mean)
        return {
            "reward": round(reward, 4), "boi_rate": round(boi_rate, 4),
            "valid_rate": round(valid_rate, 4),
            "clip_margin_mean": round(clip_mean, 4),
            "n_images": len(images), "images": images,
        }

    # Population: the strongest known route suffix plus mutations of it.  A
    # purely random population would waste every generation rediscovering the
    # route, which CE-GCG already solved.
    population = [list(init_ids)]
    while len(population) < args.population:
        mutant = list(init_ids)
        for i in range(suffix_len):
            if rng.random() < args.mutation_rate:
                mutant[i] = rng.choice(text_ids)
        population.append(mutant)

    metrics_path = output / "metrics.jsonl"
    history = []
    for generation in range(args.generations):
        records = []
        for index, candidate in enumerate(population):
            t0 = time.time()
            record = evaluate(candidate, f"g{generation}_c{index}")
            record.update({"kind": "cem_candidate", "generation": generation,
                           "candidate": index, "token_ids": candidate,
                           "seconds": round(time.time() - t0, 1)})
            records.append(record)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            print(json.dumps({k: v for k, v in record.items()
                              if k != "token_ids" and k != "images"}),
                  flush=True)
        records.sort(key=lambda r: -r["reward"])
        n_elite = max(2, int(len(records) * args.elite_frac))
        elites = records[:n_elite]
        history.append({
            "generation": generation,
            "best_reward": elites[0]["reward"],
            "best_boi_rate": elites[0]["boi_rate"],
            "best_clip_margin": elites[0]["clip_margin_mean"],
            "mean_reward": round(sum(r["reward"] for r in records) / len(records), 4),
            "n_elite": n_elite,
            "elite_token_ids": [e["token_ids"] for e in elites],
        })
        print(json.dumps({"kind": "cem_generation",
                          **{k: v for k, v in history[-1].items()
                             if k != "elite_token_ids"}}), flush=True)
        if generation == args.generations - 1:
            break
        population = []
        while len(population) < args.population:
            parent = rng.choice(elites)["token_ids"]
            child = list(parent)
            for i in range(suffix_len):
                if rng.random() < args.mutation_rate:
                    child[i] = rng.choice(text_ids)
            population.append(child)

    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    best = history[-1]
    (output / "best_projection.json").write_text(json.dumps({
        "kind": "soft_to_hard_projection",
        "projection": "behaviour-level CEM over discrete suffixes",
        "token_ids": best["elite_token_ids"][0],
        "reward": best["best_reward"],
        "note": "re-evaluate with script 10 and readback with script 66 before any claim",
    }, indent=2) + "\n")
    print(json.dumps({"kind": "cem_done", "best_reward": best["best_reward"],
                      "best_boi_rate": best["best_boi_rate"],
                      "best_clip_margin": best["best_clip_margin"]}, indent=2))


if __name__ == "__main__":
    main()
