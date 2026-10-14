#!/usr/bin/env python
"""Summarize labeled Anole baseline rollouts with Wilson intervals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if not n:
        return [None, None]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def summarize(rows: list[dict]) -> dict:
    result = {"total_rows": len(rows), "by_mode": {}}
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        route = sum(bool(row["model_sampled_route_positions"]) for row in group)
        image = sum(row["image_count"] > 0 for row in group)
        valid = sum(row["structure_valid"] for row in group)
        result["by_mode"][mode] = {
            "n": len(group),
            "route_successes": route,
            "route_rate": route / len(group),
            "route_wilson_95": wilson(route, len(group)),
            "image_successes": image,
            "image_rate": image / len(group),
            "image_wilson_95": wilson(image, len(group)),
            "valid_structure_successes": valid,
            "mean_images_per_row": sum(row["image_count"] for row in group) / len(group),
            "by_prompt": {},
        }
        for prompt_id in sorted({row["prompt_id"] for row in group}):
            subset = [row for row in group if row["prompt_id"] == prompt_id]
            prompt_route = sum(bool(row["model_sampled_route_positions"]) for row in subset)
            prompt_image = sum(row["image_count"] > 0 for row in subset)
            result["by_mode"][mode]["by_prompt"][prompt_id] = {
                "n": len(subset),
                "route_successes": prompt_route,
                "route_rate": prompt_route / len(subset),
                "route_wilson_95": wilson(prompt_route, len(subset)),
                "image_successes": prompt_image,
                "image_rate": prompt_image / len(subset),
                "image_wilson_95": wilson(prompt_image, len(subset)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize(rows), indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
