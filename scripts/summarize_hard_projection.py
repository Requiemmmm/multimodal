#!/usr/bin/env python
"""Compare a projected hard suffix against random text controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [c - h, c + h]


def stat(rows):
    k = sum(bool(row["model_sampled_boi"]) for row in rows)
    images = sum(row["image_count"] > 0 for row in rows)
    return {"n": len(rows), "route_successes": k, "route_rate": k / len(rows), "route_wilson95": wilson(k, len(rows)), "image_rollouts": images}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-metrics", required=True)
    parser.add_argument("--control-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    projection = [json.loads(line) for line in Path(args.projection_metrics).read_text().splitlines() if line.strip()]
    controls = [json.loads(line) for line in Path(args.control_metrics).read_text().splitlines() if line.strip()]
    out = {"hard_projection": stat(projection), "random_controls_pooled": stat(controls), "random_controls_by_suffix": {}}
    for control_seed in sorted({row["control_seed"] for row in controls}):
        out["random_controls_by_suffix"][str(control_seed)] = stat([row for row in controls if row["control_seed"] == control_seed])
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
