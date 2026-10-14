#!/usr/bin/env python
"""Boundary-story variant generator (CPU).

From one dual-function soft suffix (default: run 72), emit:
  1. angle-matched noise variants at target cosines (precision sweep), and
  2. partial-slot quantization variants (k of L slots NN-quantized, rest soft),
saved as soft_suffix.pt files + a candidates JSON for the screening driver
(script 68). The screening driver then measures soft R/C (and hard R on the
full-NN variants) per variant -- the discretization boundary as a point on
the precision axis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suffix", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cosines", type=float, nargs="+",
                        default=[0.99, 0.98, 0.95, 0.90, 0.85, 0.80])
    parser.add_argument("--quant-k", type=int, nargs="+",
                        default=[16, 32, 48, 64])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    payload = torch.load(args.source_suffix, map_location="cpu", weights_only=True)
    suffix = payload["suffix"].float()
    length = suffix.shape[0]

    # Vocabulary matrix for quantization (CPU, from consolidated checkpoint).
    checkpoint = Path(args.model_path) / "models/7b/consolidated.pth"
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = next(k for k in weights if k.endswith("tok_embeddings.weight"))
    vocab = weights[key].float()

    def quantized(suffix_rows):
        rows_n = torch.nn.functional.normalize(suffix_rows, dim=-1)
        vocab_n = torch.nn.functional.normalize(vocab, dim=-1)
        ids = (rows_n @ vocab_n.T).argmax(dim=-1)
        return vocab[ids]

    candidates = []

    # 1. Angle-matched noise variants (continuous precision axis).
    suffix_norm = suffix.norm(dim=-1, keepdim=True)
    for cosine in args.cosines:
        noise = torch.randn_like(suffix)
        projection = (noise * suffix).sum(dim=-1, keepdim=True) / suffix.square().sum(dim=-1, keepdim=True)
        noise = noise - projection * suffix
        noise = torch.nn.functional.normalize(noise, dim=-1) * suffix_norm
        variant = cosine * suffix + (1.0 - cosine * cosine) ** 0.5 * noise
        name = f"noise_cos{int(cosine * 100):03d}"
        torch.save({"suffix": variant, "suffix_length": length}, out / f"{name}.pt")
        candidates.append({"name": name, "suffix_path": str(out / f"{name}.pt")})

    # 2. Partial-slot quantization (locate route vs content slots).
    quant_all = quantized(suffix)
    for k in args.quant_k:
        variant = suffix.clone()
        variant[:k] = quant_all[:k]
        name = f"mixq_{k:02d}"
        torch.save({"suffix": variant, "suffix_length": length}, out / f"{name}.pt")
        candidates.append({"name": name, "suffix_path": str(out / f"{name}.pt")})

    (out / "candidates.json").write_text(json.dumps(candidates, indent=1) + "\n")
    print(json.dumps({"variants": [c["name"] for c in candidates]}))
    print("saved to", out)


if __name__ == "__main__":
    main()
