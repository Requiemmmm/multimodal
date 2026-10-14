#!/usr/bin/env python
"""Convert an HF-format Chameleon checkpoint into Meta's raw layout so the
official inference code (and therefore the whole MORPH attack pipeline) can run it.

Why this exists: several Chameleon-family checkpoints that pass the attack's
gateway screen -- notably `chameleon_gen_model`, which unlike the Chameleon base
model was actually fine-tuned to generate images -- are published in HF
safetensors format.  The official decoder (`chameleon/inference/loader.py`) loads
`models/7b/consolidated.pth` with the *official* parameter names and asserts

    missing_keys == []  and  unexpected_keys == ["rope.freqs"]

so the conversion is an exact key renaming plus the rope buffer.  The mapping
below was read off Anole's own raw checkpoint (same architecture), not guessed.

The tokenizer and VQ-GAN are bit-identical across the Chameleon family, so they
are copied from an existing raw checkpoint rather than converted.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open


def build_key_map(n_layers: int) -> dict[str, str]:
    m = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    for i in range(n_layers):
        p_hf = f"model.layers.{i}."
        p_raw = f"layers.{i}."
        m.update({
            p_hf + "input_layernorm.weight": p_raw + "attention_norm.weight",
            p_hf + "post_attention_layernorm.weight": p_raw + "ffn_norm.weight",
            p_hf + "self_attn.q_proj.weight": p_raw + "attention.wq.weight",
            p_hf + "self_attn.k_proj.weight": p_raw + "attention.wk.weight",
            p_hf + "self_attn.v_proj.weight": p_raw + "attention.wv.weight",
            p_hf + "self_attn.o_proj.weight": p_raw + "attention.wo.weight",
            p_hf + "self_attn.q_norm.weight": p_raw + "attention.q_normalization.weight",
            p_hf + "self_attn.q_norm.bias": p_raw + "attention.q_normalization.bias",
            p_hf + "self_attn.k_norm.weight": p_raw + "attention.k_normalization.weight",
            p_hf + "self_attn.k_norm.bias": p_raw + "attention.k_normalization.bias",
            p_hf + "mlp.gate_proj.weight": p_raw + "feed_forward.w1.weight",
            p_hf + "mlp.down_proj.weight": p_raw + "feed_forward.w2.weight",
            p_hf + "mlp.up_proj.weight": p_raw + "feed_forward.w3.weight",
        })
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True, help="HF-format source dir")
    ap.add_argument("--reference-raw", required=True,
                    help="an existing raw Chameleon checkpoint, for rope.freqs "
                         "and for the shared tokenizer/VQ assets")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_model)
    ref = Path(args.reference_raw)
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"output already exists: {out}")

    cfg = json.loads((hf_dir / "config.json").read_text())
    n_layers = int(cfg["num_hidden_layers"])
    key_map = build_key_map(n_layers)

    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    def normalise(key_hf: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reconcile the q/k-norm layout difference between the two formats.

        HF stores the per-head q/k normalisation as [n_heads, head_dim] while the
        official Transformer holds a single [head_dim] vector.  The HF rows are
        byte-identical (verified: row-difference std is exactly 0), so selecting
        the first row is lossless -- without this the loader fails with
        "size mismatch ... shape [32, 128] vs [128]".
        """
        if key_hf.endswith(("self_attn.q_norm.weight", "self_attn.q_norm.bias",
                            "self_attn.k_norm.weight", "self_attn.k_norm.bias")):
            if tensor.dim() == 2:
                if not torch.equal(tensor[0].expand_as(tensor), tensor):
                    raise SystemExit(
                        f"{key_hf}: per-head rows differ, cannot collapse to a "
                        "single vector losslessly -- inspect before converting")
                return tensor[0].contiguous()
        return tensor

    converted, skipped = {}, []
    for shard in shard_names:
        with safe_open(hf_dir / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in key_map:
                    skipped.append(name)          # e.g. the VQ model weights
                    continue
                converted[key_map[name]] = normalise(name, f.get_tensor(name))
    print(json.dumps({"kind": "converted_tensors", "n": len(converted),
                      "n_skipped": len(skipped),
                      "skipped_head": skipped[:6]}, indent=2), flush=True)

    # rope.freqs is a buffer the official loader expects to find in the file.
    ref_sd = torch.load(ref / "models/7b/consolidated.pth", map_location="cpu")
    converted["rope.freqs"] = ref_sd["rope.freqs"]

    missing = [k for k in ref_sd.keys() if k not in converted]
    extra = [k for k in converted if k not in ref_sd]
    # Names matching is not enough: the first attempt at this conversion passed
    # the key check while carrying [n_heads, head_dim] q/k norms that the
    # official model could not load.  Compare shapes too.
    shape_mismatch = [f"{k}: {tuple(converted[k].shape)} vs ref "
                      f"{tuple(ref_sd[k].shape)}"
                      for k in converted
                      if k in ref_sd and converted[k].shape != ref_sd[k].shape]
    print(json.dumps({"kind": "key_check", "missing_vs_reference": missing,
                      "extra_vs_reference": extra,
                      "shape_mismatch": shape_mismatch}, indent=2), flush=True)
    if missing:
        raise SystemExit(f"conversion incomplete, missing {len(missing)} keys")
    if shape_mismatch:
        raise SystemExit(
            f"shape mismatch on {len(shape_mismatch)} tensors; first: "
            f"{shape_mismatch[0]}")

    (out / "models" / "7b").mkdir(parents=True)
    torch.save(converted, out / "models" / "7b" / "consolidated.pth")

    # Tokenizer and VQ are shared across the family -- copy them.
    (out / "tokenizer").mkdir(parents=True)
    for name in ("text_tokenizer.json", "vqgan.yaml", "vqgan.ckpt"):
        shutil.copy2(ref / "tokenizer" / name, out / "tokenizer" / name)
    # The official loader also reads these two files alongside the weights.
    for name in ("params.json", "consolidate_params.json", "config.json", "checklist.chk"):
        src = ref / "models" / "7b" / name
        if src.is_file():
            shutil.copy2(src, out / "models" / "7b" / name)

    print(json.dumps({"kind": "raw_checkpoint_written", "output": str(out),
                      "n_layers": n_layers, "n_tensors": len(converted)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
