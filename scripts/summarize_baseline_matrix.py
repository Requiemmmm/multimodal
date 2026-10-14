#!/usr/bin/env python
"""Summarize labeled baseline rollouts with Wilson intervals."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return center - half, center + half


def summarize_records(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["mode"]].append(record)
    summary = {"total_rollouts": len(records), "groups": {}}
    for mode, items in sorted(groups.items()):
        native_route = sum(bool(item["model_sampled_route_positions"]) for item in items)
        valid_images = sum(item["image_count"] > 0 and item["structure_valid"] for item in items)
        image_rollouts = sum(item["image_count"] > 0 for item in items)
        route_ci = wilson(native_route, len(items))
        valid_ci = wilson(valid_images, len(items))
        summary["groups"][mode] = {
            "n": len(items),
            "native_route_successes": native_route,
            "native_route_rate": native_route / len(items),
            "native_route_wilson_95": route_ci,
            "valid_image_rollouts": valid_images,
            "valid_image_rate_per_rollout": valid_images / len(items),
            "valid_image_wilson_95": valid_ci,
            "image_count_total": image_rollouts,
            "forced_boi_rollouts": sum(bool(item["forced_boi_positions"]) for item in items),
            "prompt_ids": sorted({item["prompt_id"] for item in items}),
            "seeds": sorted({item["seed"] for item in items}),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    output = Path(args.output)
    output.write_text(json.dumps(summarize_records(records), indent=2, ensure_ascii=False) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
