#!/usr/bin/env python
"""Validate a saved Anole token trajectory without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from morph.adapters.anole import load_anole_token_sets
from morph.decoding.anole_segments import analyze_anole_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--trajectory", required=True, help="JSON list of generated token IDs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trajectory_path = Path(args.trajectory)
    token_ids = json.loads(trajectory_path.read_text())
    if not isinstance(token_ids, list) or not all(isinstance(token_id, int) for token_id in token_ids):
        raise SystemExit("trajectory must be a JSON list of integer token IDs")
    segments = analyze_anole_sequence(token_ids, load_anole_token_sets(args.tokenizer_path))
    record = {
        "kind": "trajectory_structure_audit",
        "token_count": len(token_ids),
        "route_positions": [index for index, value in enumerate(segments.route_mask) if value],
        "visual_token_count": sum(segments.visual_mask),
        "forced_positions": [index for index, value in enumerate(segments.forced_mask) if value],
        "completed_images": segments.completed_images,
        "valid": segments.valid,
        "error": segments.error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
