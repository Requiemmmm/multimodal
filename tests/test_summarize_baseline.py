import json
from pathlib import Path

from scripts.summarize_baseline_matrix import summarize_records


def test_zero_native_route_has_nonzero_upper_interval():
    records = [
        {"mode": "student_only", "model_sampled_route_positions": [], "image_count": 0, "structure_valid": True, "forced_boi_positions": [], "prompt_id": "p", "seed": i}
        for i in range(16)
    ]
    summary = summarize_records(records)["groups"]["student_only"]
    assert summary["native_route_successes"] == 0
    assert summary["native_route_wilson_95"][1] > 0

