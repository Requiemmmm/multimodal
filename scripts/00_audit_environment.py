#!/usr/bin/env python
"""Record one-time environment and Lumina checkpoint provenance/load audit."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_ROOT", "/srv/morph/models") + "/lumina-mgpt-7b-768-omni")
    parser.add_argument("--code-root", default=os.environ.get("EXTERNAL_ROOT", "/srv/morph/external") + "/Lumina-mGPT")
    parser.add_argument("--output", default=os.environ.get("RUN_ROOT", "/srv/morph/runs") + "/bootstrap/bootstrap_manifest.json")
    args = parser.parse_args()
    model_path, code_root, output = map(Path, (args.model_path, args.code_root, args.output))
    required = ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json"]
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise SystemExit(f"checkpoint missing required files: {', '.join(missing)}")

    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    sys.path.insert(0, str(code_root / "lumina_mgpt"))
    from model.chameleon import ChameleonForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Loading on meta/device_map is intentionally separate from image decode:
    # the latter requires the licensed Meta VQ files and is audited in Stage 0.
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    input_ids = tokenizer("MORPH load audit", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    if out.logits.shape[:2] != input_ids.shape:
        raise RuntimeError(f"unexpected logits shape: {tuple(out.logits.shape)}")

    def token_id(token: str) -> int:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"token is not a single vocabulary entry: {token!r} -> {ids}")
        return ids[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": command_output("date", "-Is"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "transformers": transformers.__version__,
        "hub_id": "Alpha-VLLM/Lumina-mGPT-7B-768-Omni",
        "revision": "c24a700e56cd319917598782ede8b126a4eb1b5d",
        "model_path": str(model_path),
        "code_root": str(code_root),
        "code_commit": command_output("git", "-C", str(code_root), "rev-parse", "HEAD"),
        "files": {name: (model_path / name).stat().st_size for name in required},
        "config": {"model_type": config.model_type, "vocab_size": config.vocab_size, "hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers, "max_position_embeddings": config.max_position_embeddings},
        "token_ids": {token: token_id(token) for token in ["<racm3:break>", "<eoss>", "<reserved08796>", "<reserved08797>", "<reserved08798>", "<reserved08799>", "<reserved08706>"]},
        "load_forward": {"success": True, "logits_shape": list(out.logits.shape)},
        "image_decode": {"success": False, "reason": "Meta Chameleon VQ files not installed yet"},
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
