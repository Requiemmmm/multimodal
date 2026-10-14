#!/usr/bin/env python
"""Audit the ordinary user-suffix prompt boundary without loading Anole weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from morph.adapters.anole import build_anole_user_turn, load_anole_token_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suffix", default=" [MORPH route probe]")
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer_path)
    prompts = yaml.safe_load(Path(args.prompts).read_text())
    tokens = load_anole_token_sets(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records = {}
    for prompt_id, prompt_set in prompts.items():
        user_ids = build_anole_user_turn(prompt_set["student"], args.suffix, tokenizer, tokens)
        teacher_ids = build_anole_user_turn(prompt_set["teacher"], "", tokenizer, tokens)
        records[prompt_id] = {
            "teacher_token_count": len(teacher_ids),
            "student_suffix_token_count": len(user_ids),
            "student_last_token": user_ids[-1],
            "expected_eot": tokens.eot_id,
            "student_suffix_before_eot": user_ids[-2] != tokens.eot_id,
            "teacher_bos": teacher_ids[0],
            "expected_bos": tokens.bos_id,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"kind": "prompt_boundary_audit", "suffix": args.suffix, "prompts": records}, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
