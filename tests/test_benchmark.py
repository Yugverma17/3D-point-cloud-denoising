"""Benchmark harness, baselines and the calibration guard."""

import numpy as np
import pytest

from pointdenoise.baselines import bilateral_filter, estimate_normals, laplacian_smooth
from pointdenoise.benchmark import (
    NOISE_LEVELS,
    RESOLUTIONS,
    BenchmarkCase,
    add_benchmark_noise,
    calibrate,
    comparison_table,
    load_released_set,
    run_case,
)
from pointdenoise.data import Shape
from pointdenoise.metrics import chamfer_distance


def sphere(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    c = BenchmarkCase(dataset="PUNet", resolution="sparse", noise=0.01)
    for s in range(3):
        clean = sphere(seed=s)
        c.shapes.append(Shape(clean, noisy=add_benchmark_noise(clean, 0.01, rng)))
        c.names.append(f"shape{s}")
        c.meshes.append(None)
    return c


def test_noise_is_relative_to_the_unit_sphere():
    """1% must mean the same displacement regardless of the object's size."""
    clean = sphere()
    rng = np.random.default_rng(0)
    small = add_benchmark_noise(clean, 0.02, rng)
    big = add_benchmark_noise(clean * 100.0, 0.02, rng) / 100.0
    d_small = np.linalg.norm(small - clean, axis=1).mean()
    d_big = np.linalg.norm(big - clean, axis=1).mean()
    assert d_big == pytest.approx(d_small, rel=0.15)


def test_bilateral_reduces_noise():
    clean = sphere()
    noisy = add_benchmark_noise(clean, 0.02, np.random.default_rng(0))
    assert chamfer_distance(bilateral_filter(noisy), clean) < chamfer_distance(noisy, clean)


def test_laplacian_helps_once_then_over_smooths():
    """
    The naive baseline is only useful for a single pass.

    Measured on a noisy sphere: one iteration is 52% better than the input,
    two is 26% better, three is 11% *worse* and five is 91% worse. Every pass
    drags points toward their neighbourhood centroid, so past the first the
    filter is destroying real structure rather than noise. Bilateral does not
    do this (see below), which is the whole reason it is the reference
    baseline and Laplacian is not.
    """
    clean = sphere()
    noisy = add_benchmark_noise(clean, 0.02, np.random.default_rng(0))
    baseline = chamfer_distance(noisy, clean)

    assert chamfer_distance(laplacian_smooth(noisy, iterations=1), clean) < baseline
    assert chamfer_distance(laplacian_smooth(noisy, iterations=5), clean) > baseline


def test_bilateral_is_stable_across_iterations():
    """Unlike Laplacian, extra passes do not make bilateral worse."""
    clean = sphere()
    noisy = add_benchmark_noise(clean, 0.02, np.random.default_rng(0))
    baseline = chamfer_distance(noisy, clean)
    scores = [chamfer_distance(bilateral_filter(noisy, iterations=i), clean)
              for i in (1, 2, 3)]
    assert all(s < baseline * 0.5 for s in scores)
    assert max(scores) < 2 * min(scores)


def test_normals_are_unit_length():
    n = estimate_normals(sphere(500))
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-8)


def test_run_case_scores_every_shape(case):
    rows, avg = run_case(case, lambda p: p, with_p2m=False)
    assert len(rows) == 3 and "cd" in avg


def test_identity_scores_the_same_as_the_noisy_input(case):
    _, avg = run_case(case, lambda p: p, with_p2m=False)
    direct = np.mean([chamfer_distance(s.noisy, s.clean) * 1e4 for s in case.shapes])
    assert avg["cd"] == pytest.approx(direct)


def test_calibration_is_inconclusive_off_the_released_set(case):
    """
    Substitute shapes must never report PASS. Their difficulty is confounded
    with the metric convention, which is the thing being tested.
    """
    case.dataset = "local"
    result = calibrate(case)
    assert result["conclusive"] is False
    assert result["within_tolerance"] is False


def test_calibration_reports_a_ratio_on_an_official_set(case):
    result = calibrate(case)
    assert result["conclusive"] is True
    assert np.isfinite(result["cd_ratio"])
    assert result["reference"] == "Bilateral"


def test_comparison_table_leaves_missing_cells_blank():
    table = comparison_table({("sparse", 0.01): {"cd": 2.5, "p2m": 0.45}}, our_name="Ours")
    assert "P2P-Bridge" in table and "2.50/0.45" in table
    # our row has one filled cell and five blanks
    row = next(line for line in table.splitlines() if line.startswith("Ours"))
    assert row.count("-") >= 5


def test_load_released_set_explains_itself_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/benchmark.md"):
        load_released_set(tmp_path)


def test_grid_constants_match_the_published_tables():
    assert RESOLUTIONS == {"sparse": 10_000, "dense": 50_000}
    assert NOISE_LEVELS == (0.01, 0.02, 0.03)
