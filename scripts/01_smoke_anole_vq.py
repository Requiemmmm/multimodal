#!/usr/bin/env python
"""Decode a deterministic Anole VQ code sequence without loading the 7B model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model_path, code_root, output_dir = map(Path, (args.model_path, args.code_root, args.output_dir))
    tokenizer_dir = model_path / "tokenizer"
    required = [tokenizer_dir / name for name in ("vqgan.yaml", "vqgan.ckpt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing Anole VQ assets: " + ", ".join(missing))
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    sys.path.insert(0, str(code_root))
    import torch
    from chameleon.inference.image_tokenizer import ImageTokenizer

    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats()
    image_tokenizer = ImageTokenizer(
        cfg_path=str(tokenizer_dir / "vqgan.yaml"),
        ckpt_path=str(tokenizer_dir / "vqgan.ckpt"),
        device=device,
    )
    # A deterministic 32x32 codebook grid confirms the decoder contract but
    # deliberately carries no claim of semantic or model-generated content.
    codebook_ids = torch.arange(1024, dtype=torch.long, device=device)
    image = image_tokenizer.pil_from_img_toks(codebook_ids)
    image_path = output_dir / "vq_decoder_only.png"
    image.save(image_path)
    pixels = torch.from_numpy(__import__("numpy").array(image, copy=True)).float()
    record = {
        "kind": "vq_decoder_only",
        "classification": "decoder integration only; not teacher, bridge, or student rollout",
        "input": {"codebook_token_count": int(codebook_ids.numel()), "codebook_id_min": int(codebook_ids.min()), "codebook_id_max": int(codebook_ids.max())},
        "output": {"image": image_path.name, "size": list(image.size), "pixel_std": float(pixels.std())},
        "device": str(device),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
