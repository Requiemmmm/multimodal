#!/usr/bin/env python
"""Project a trained Anole soft suffix onto ordinary text vocabulary IDs.

Projection is a diagnostic: the resulting hard suffix is re-evaluated by the
official tokenizer path in a separate run and is never treated as equivalent
to the continuous soft result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-state", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    state = Path(args.soft_state)
    model_path = Path(args.model_path)
    if not state.is_file():
        raise SystemExit(f"soft state missing: {state}")
    if not (model_path / "tokenizer/text_tokenizer.json").is_file():
        raise SystemExit(f"tokenizer missing under {model_path}")
    import torch
    from tokenizers import Tokenizer
    from morph.adapters.anole import load_anole_token_sets

    suffix = torch.load(state, map_location="cpu", weights_only=True)["suffix"].float()
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_sets = load_anole_token_sets(tokenizer_path)
    ordered_ids = sorted(token_sets.text_token_ids)
    # Load the embedding matrix without instantiating a second GPU model: the
    # Anole checkpoint is a consolidated CPU-readable file.
    checkpoint = model_path / "models/7b/consolidated.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"consolidated checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = next((k for k in payload if k.endswith("tok_embeddings.weight")), None)
    if key is None:
        raise SystemExit("tok_embeddings.weight not found in consolidated checkpoint")
    embedding = payload[key].float()
    candidates = torch.nn.functional.normalize(embedding[ordered_ids], dim=-1)
    normalized = torch.nn.functional.normalize(suffix, dim=-1)
    scores, positions = torch.topk(normalized @ candidates.T, k=args.top_k, dim=-1)
    # Greedy projection is the only hard suffix emitted by this diagnostic.
    ids = [ordered_ids[int(position[0])] for position in positions]
    result = {
        "kind": "soft_to_hard_projection",
        "source_soft_state": str(state),
        "projection": "nearest_text_embedding",
        "top_k": args.top_k,
        "token_ids": ids,
        "token_pieces": [tokenizer.decode([token_id]) for token_id in ids],
        "nearest_cosine": [float(score[0]) for score in scores],
        # Full per-slot candidate sets for top-k>1 (used by rerank/greedy scoring).
        "topk_token_ids": [[ordered_ids[int(p)] for p in slot] for slot in positions],
        "topk_nearest_cosine": [[float(s) for s in slot] for slot in scores],
        "ordinary_text_only": True,
        "boi_or_visual_ids_in_suffix": bool(set(ids) & ({token_sets.boi_id, token_sets.eoi_id, token_sets.eot_id} | set(token_sets.image_token_ids))),
        "note": "Projection gap diagnostic; hard suffix must be re-evaluated separately and is not a soft/native result.",
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
