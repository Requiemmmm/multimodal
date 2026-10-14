#!/usr/bin/env python
"""Extract a model-sampled visual prefix from a script-10 eval into a JSON file
usable by script 65's --sampled-prefix-json.

The exposure-bias diagnostic (script 72) showed the deployed rollout agrees with
the teacher's apple trajectory at *zero* of the first eight visual positions, and
that the target margin collapses from 2.31 to 0.41 nats once the model's own
tokens are the context.  Searching against a prefix the model actually visits
requires such a prefix; this writes one out.

Prefixes are concatenated across rollouts so a search can score several of them:
``{"prefixes": [[...], [...]], "seed_sources": [...]}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--content-steps", type=int, default=8)
    parser.add_argument("--max-prefixes", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer_path = Path(args.model_path) / "tokenizer/text_tokenizer.json"
    print(json.dumps({"kind": "loading_token_sets", "path": str(tokenizer_path)}),
          flush=True)
    from morph.adapters.anole import load_anole_token_sets
    token_sets = load_anole_token_sets(str(tokenizer_path))
    boi, eoi = token_sets.boi_id, token_sets.eoi_id

    prefixes, sources = [], []
    for sample_dir in sorted(Path(args.eval_dir).glob("*_seed*")):
        path = sample_dir / "generated_tokens.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if boi not in raw:
            continue
        start = raw.index(boi) + 1
        visual = raw[start:start + 1024]
        if eoi in visual:
            visual = visual[:visual.index(eoi)]
        if len(visual) < args.content_steps:
            continue
        prefixes.append([int(t) for t in visual[:args.content_steps]])
        sources.append(sample_dir.name)
        if len(prefixes) >= args.max_prefixes:
            break

    payload = {
        "kind": "sampled_visual_prefixes",
        "eval_dir": args.eval_dir,
        "content_steps": args.content_steps,
        "n_prefixes": len(prefixes),
        "seed_sources": sources,
        "prefixes": prefixes,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"kind": "sampled_prefix_written",
                      "output": args.output, "n_prefixes": len(prefixes),
                      "sources": sources}, indent=2), flush=True)
    if not prefixes:
        raise SystemExit("no rollout in the eval dir produced a usable prefix")


if __name__ == "__main__":
    main()
