#!/usr/bin/env python
"""Summarize route-only bridge JSONL with per-prompt/alpha Wilson intervals."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [center - half, center + half]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["prompt_id"], float(row["alpha"]))].append(row)
    result = {"total": len(rows), "groups": {}, "by_alpha": {}}
    for (prompt_id, alpha), items in sorted(groups.items()):
        successes = sum(bool(item["boi_sampled"]) for item in items)
        result["groups"][f"{prompt_id}/alpha={alpha:g}"] = {
            "n": len(items),
            "boi_successes": successes,
            "boi_rate": successes / len(items),
            "wilson95": wilson(successes, len(items)),
            "mean_boi_step": (
                sum(item["boi_step"] for item in items if item["boi_step"] is not None) / successes
                if successes else None
            ),
            "mean_top50_overlap": sum(item["mean_top50_support_overlap"] for item in items) / len(items),
        }
    alpha_groups = defaultdict(list)
    for row in rows:
        alpha_groups[float(row["alpha"])].append(row)
    for alpha, items in sorted(alpha_groups.items()):
        successes = sum(bool(item["boi_sampled"]) for item in items)
        result["by_alpha"][str(alpha)] = {
            "n": len(items),
            "boi_successes": successes,
            "boi_rate": successes / len(items),
            "wilson95": wilson(successes, len(items)),
            "mean_top50_overlap": sum(item["mean_top50_support_overlap"] for item in items) / len(items),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    output = summarize(rows)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
