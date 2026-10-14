#!/usr/bin/env python
"""CPU-only audit of Anole's official text/image legality support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from morph.adapters.anole import load_anole_token_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokens = load_anole_token_sets(args.tokenizer_path)
    if tokens.boi_id not in tokens.text_step_support:
        raise RuntimeError("official Anole text support unexpectedly excludes BOI")
    if tokens.boi_id in tokens.image_step_support:
        raise RuntimeError("official Anole image support unexpectedly includes BOI")
    if tokens.eoi_id in tokens.image_step_support:
        raise RuntimeError("official Anole image support should force EOI, not sample it")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "official_support_audit",
        "tokenizer_path": str(Path(args.tokenizer_path)),
        "vocab_size": tokens.vocab_size,
        "token_ids": {"bos": tokens.bos_id, "eos": tokens.eos_id, "pad": tokens.pad_id, "boi": tokens.boi_id, "eoi": tokens.eoi_id, "eot": tokens.eot_id},
        "support": {
            "text_step_count": len(tokens.text_step_support),
            "image_step_count": len(tokens.image_step_support),
            "boi_finite_in_text": tokens.boi_id in tokens.text_step_support,
            "boi_finite_in_image": tokens.boi_id in tokens.image_step_support,
            "eoi_sampled_in_image": tokens.eoi_id in tokens.image_step_support,
            "text_image_overlap": len(tokens.text_step_support & tokens.image_step_support),
        },
        "bridge_implication": "At a synchronized text prefix, authorized teacher and normal student have the same official text support and BOI is legal; after a shared BOI, both have the same 8192-token visual support. This is a static decoder fact, not a rollout result.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
