#!/usr/bin/env python
"""Run the official Lumina authorized-image path and save Stage-0 artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_PROMPT = "Generate an image of 768x768 according to the following prompt:\nA golden retriever standing on a plain light background."


def require_vq_files(code_root: Path) -> None:
    tokenizer_dir = code_root / "lumina_mgpt" / "ckpts" / "chameleon" / "tokenizer"
    required = [tokenizer_dir / name for name in ("text_tokenizer.json", "vqgan.yaml", "vqgan.ckpt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing licensed Meta Chameleon VQ files:\n  "
            + "\n  ".join(missing)
            + "\nObtain them through Meta's official Chameleon access flow; do not substitute an unverified mirror."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-gen-len", type=int, default=8192)
    parser.add_argument("--target-size", type=int, default=768)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--image-top-k", type=int, default=2000)
    parser.add_argument("--text-top-k", type=int, default=5)
    args = parser.parse_args()
    code_root, output_dir = Path(args.code_root), Path(args.output_dir)
    require_vq_files(code_root)
    sys.path.insert(0, str(code_root / "lumina_mgpt"))
    import torch
    from inference_solver import FlexARInferenceSolver
    from xllmx.util.misc import random_seed

    output_dir.mkdir(parents=True, exist_ok=False)
    random_seed(seed=args.seed)
    solver = FlexARInferenceSolver(args.model_path, precision="bf16", target_size=args.target_size)
    generated_text, images = solver.generate(
        images=[],
        qas=[[args.prompt, None]],
        max_gen_len=args.max_gen_len,
        temperature=1.0,
        logits_processor=solver.create_logits_processor(
            cfg=args.cfg, image_top_k=args.image_top_k, text_top_k=args.text_top_k
        ),
    )
    for index, image in enumerate(images):
        image.save(output_dir / f"image_{index:02d}.png")
    record = {
        "kind": "teacher-only",
        "alpha": 1.0,
        "prompt": args.prompt,
        "seed": args.seed,
        "max_gen_len": args.max_gen_len,
        "cfg": args.cfg,
        "image_top_k": args.image_top_k,
        "text_top_k": args.text_top_k,
        "generated_text": generated_text,
        "image_count": len(images),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (output_dir / "metrics.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
