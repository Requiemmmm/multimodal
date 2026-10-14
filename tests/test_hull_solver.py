"""CPU sanity tests for the cone/hull distance solvers in script 21."""
import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "hull21", str(PROJECT_ROOT / "scripts/21_anole_hull_distance.py"))
hull = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hull)


def make_rays(count, dim, seed):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, dim, generator=generator)
    center = raw.mean(dim=0)
    centered = raw - center
    return torch.nn.functional.normalize(centered, dim=-1)


def test_single_ray_is_reachable():
    rays = make_rays(2000, 128, seed=0)
    u = rays[42]
    distance, _, _ = hull.solve_cone(rays, u, iters=3000)
    assert distance < 0.01, f"single ray should be reachable, got {distance}"


def test_ray_mixture_is_reachable():
    rays = make_rays(2000, 128, seed=1)
    u = torch.nn.functional.normalize(rays[1] + rays[7] + rays[13], dim=-1)
    distance, _, _ = hull.solve_cone(rays, u, iters=3000)
    assert distance < 0.02, f"three-ray mixture should be reachable, got {distance}"


def test_hull_solver_single_ray():
    rays = make_rays(2000, 128, seed=2)
    u = rays[99]
    distance, _ = hull.solve_hull(rays, u, iters=2000)
    assert distance < 0.01, f"FW on simplex should hit the single ray, got {distance}"


def test_hull_upper_bounds_cone():
    # The hull (sum alpha = 1) is a subset of the cone (alpha >= 0), so its
    # distance can never be smaller than the cone distance.
    rays = make_rays(2000, 128, seed=3)
    generator = torch.Generator().manual_seed(4)
    u = torch.nn.functional.normalize(torch.randn(128, generator=generator), dim=-1)
    cone_distance, _, _ = hull.solve_cone(rays, u, iters=3000)
    hull_distance, _ = hull.solve_hull(rays, u, iters=2000)
    assert hull_distance >= cone_distance - 1e-4, (
        f"hull {hull_distance} should be >= cone {cone_distance}")
    # With n >> d isotropic rays the cone spans the whole space, so the
    # informative quantities are the hull distance and the alpha mass, which
    # the real-data calibration in the run script reports.
    assert 0.0 <= cone_distance <= 2.0


if __name__ == "__main__":
    test_single_ray_is_reachable()
    test_ray_mixture_is_reachable()
    test_hull_solver_single_ray()
    test_hull_upper_bounds_cone()
    print("hull solver tests passed")
