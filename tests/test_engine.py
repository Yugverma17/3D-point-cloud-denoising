"""Training loop, inference and reporting."""

import numpy as np
import pytest
import torch

from pointdenoise.data import Shape
from pointdenoise.engine import denoise_cloud, evaluate, load_model, train, write_report
from pointdenoise.metrics import chamfer_distance
from pointdenoise.model import Denoiser


def sphere(n=800, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def shapes():
    return [Shape(sphere(seed=s), noise_level=0.02, rng=np.random.default_rng(s))
            for s in (0, 1)]


def test_denoise_cloud_preserves_point_count(shapes):
    model = Denoiser(d_model=32, num_heads=2, num_layers=1, k_coarse=8, k_fine=4)
    out = denoise_cloud(model, shapes[0], points_per_patch=32, batch_size=32)
    assert out.shape == shapes[0].noisy.shape
    assert np.isfinite(out).all()


def test_untrained_model_leaves_the_cloud_alone(shapes):
    """
    The zero-initialised head means an untrained model is the identity, so
    running it can never make the input worse. This is the structural guard
    against the old failure mode.
    """
    model = Denoiser(d_model=32, num_heads=2, num_layers=1, k_coarse=8, k_fine=4)
    out = denoise_cloud(model, shapes[0], points_per_patch=32, batch_size=32)
    assert np.allclose(out, shapes[0].noisy, atol=1e-5)


def test_evaluate_returns_rows_and_baseline(shapes):
    model = Denoiser(d_model=32, num_heads=2, num_layers=1, k_coarse=8, k_fine=4)
    rows, baseline = evaluate(model, shapes, ["a", "b"], points_per_patch=32, batch_size=32)
    assert len(rows) == len(baseline) == 2
    assert all("cd" in r for r in rows)


def test_write_report_includes_baseline_and_delta(tmp_path):
    out = tmp_path / "scores.txt"
    text = write_report(out, ["a"], [{"cd": 1.0}], [{"cd": 2.0}])
    assert "Improvement" in text and "+50.0%" in text
    assert out.read_text().strip() == text.strip()


def test_train_writes_checkpoints_and_history(tmp_path, shapes):
    model, history = train(
        shapes,
        out_dir=tmp_path,
        epochs=2,
        batch_size=4,
        points_per_patch=32,
        patches_per_shape=8,
        model_kwargs={"d_model": 32, "num_heads": 2, "num_layers": 1,
                      "k_coarse": 8, "k_fine": 4},
    )
    assert len(history) == 2
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.json").exists()

    # last.pt is the final epoch; best.pt is whichever epoch had the lowest
    # loss, which is not necessarily the last one
    _, last_ckpt = load_model(tmp_path / "last.pt")
    assert last_ckpt["epoch"] == 2
    _, best_ckpt = load_model(tmp_path / "best.pt")
    assert 1 <= best_ckpt["epoch"] <= 2

    # a checkpoint must rebuild without being told the architecture
    restored, _ = load_model(tmp_path / "last.pt")
    x = torch.randn(1, 16, 3)
    assert torch.allclose(restored(x), model.cpu().eval()(x), atol=1e-5)


@pytest.mark.slow
def test_training_improves_a_real_cloud(tmp_path, shapes):
    """
    Train, then require the denoised cloud to beat the noisy input.

    The budget below is the smallest that reliably converges. Much less and
    the model has learned a partial displacement that overshoots, which scores
    worse than not denoising at all - at 4 epochs / 128 patches this same
    check comes out 55% *worse* than baseline, at 10 / 256 it is 89% better.
    That is undertraining rather than a defect, but it is worth knowing the
    cliff is there before reading anything into a short run.
    """
    torch.manual_seed(0)
    model, _ = train(
        shapes,
        out_dir=tmp_path,
        epochs=10,
        batch_size=8,
        points_per_patch=48,
        patches_per_shape=256,
        repulsion_weight=0.0,
        model_kwargs={"d_model": 64, "num_heads": 4, "num_layers": 2,
                      "k_coarse": 8, "k_fine": 4},
    )
    shape = shapes[0]
    out = denoise_cloud(model, shape, points_per_patch=48, batch_size=64)
    before = chamfer_distance(shape.noisy, shape.clean)
    after = chamfer_distance(out, shape.clean)
    print(f"\n  CD {before*1e4:.2f} -> {after*1e4:.2f} (x1e-4), "
          f"{100*(before-after)/before:.1f}% better")
    assert after < before
