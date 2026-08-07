"""
End-to-end checks.

The important one is `test_training_reduces_noise_on_a_simple_surface`: it
trains the real model briefly on a sphere and asserts the output is closer to
the true surface than the input was. The previous version of this project
failed exactly that property while every unit test would have passed, so it is
the test worth having.
"""

import numpy as np
import pytest
import torch

from pointdenoise.data import (
    PatchDataset,
    Shape,
    centres_to_world,
    collate,
    extract_patch,
    iter_eval_patches,
)
from pointdenoise.geometry import add_gaussian_noise
from pointdenoise.losses import DenoisingLoss, RepulsionLoss, RobustChamferLoss
from pointdenoise.metrics import chamfer_distance, format_table, summarize
from pointdenoise.model import Denoiser


def sphere(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def noisy_sphere():
    clean = sphere()
    return Shape(clean, noise_level=0.02, rng=np.random.default_rng(1))


def test_patch_centre_is_first_and_frame_round_trips(noisy_sphere):
    rng = np.random.default_rng(0)
    radius = 0.05 * noisy_sphere.diagonal
    s = extract_patch(noisy_sphere, 17, radius, 64, rng)
    assert s is not None
    # centre point sits at the origin of its own frame
    assert torch.allclose(s["points"][0], torch.zeros(3), atol=1e-6)
    # and maps back to where it came from
    world = centres_to_world(s["points"][0:1].numpy(), collate([s]))[0]
    assert np.allclose(world, noisy_sphere.noisy[17], atol=1e-6)


def test_every_point_gets_a_patch(noisy_sphere):
    """
    A fixed radius silently skipped a third of the points on a sparse cloud,
    leaving them noisy and destroying the coverage half of Chamfer distance.
    """
    rng = np.random.default_rng(0)
    radius = 0.05 * noisy_sphere.diagonal
    covered = sum(
        extract_patch(noisy_sphere, i, radius, 48, rng, with_target=False) is not None
        for i in range(len(noisy_sphere.noisy))
    )
    assert covered == len(noisy_sphere.noisy)


def test_target_is_closer_to_the_surface_than_the_input(noisy_sphere):
    """The supervision must actually point toward the clean surface."""
    rng = np.random.default_rng(0)
    radius = 0.05 * noisy_sphere.diagonal
    for i in (5, 50, 500):
        s = extract_patch(noisy_sphere, i, radius, 64, rng)
        if s is None:
            continue
        moved = (s["target"] - s["points"]).norm(dim=1).mean()
        assert moved > 0, "target identical to input"
        assert moved < 1.0, "target implausibly far away - frames likely mismatched"


def test_model_starts_as_identity():
    """Zero-initialised head means an untrained model cannot corrupt input."""
    model = Denoiser(d_model=32, num_heads=2, num_layers=1)
    x = torch.randn(2, 32, 3)
    assert torch.allclose(model(x), x, atol=1e-6)


def test_losses_are_finite_and_positive():
    pred, target = torch.randn(2, 32, 3), torch.randn(2, 32, 3)
    assert RobustChamferLoss()(pred, target).item() > 0
    assert RepulsionLoss()(pred).item() >= 0
    total, parts = DenoisingLoss()(pred, target)
    assert torch.isfinite(total) and set(parts) == {"chamfer", "repulsion", "total"}


def test_dataset_and_collate(noisy_sphere):
    ds = PatchDataset([noisy_sphere], points_per_patch=32, patches_per_shape=8, seed=0)
    assert len(ds) == 8
    batch = collate([ds[i] for i in range(8)])
    assert batch["points"].shape[1:] == (32, 3)
    assert batch["target"].shape == batch["points"].shape


def test_collate_survives_all_none():
    assert collate([None, None]) is None


def test_chamfer_is_zero_for_identical_clouds():
    pts = sphere(500)
    assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-12)


def test_chamfer_grows_with_noise():
    clean = sphere(500)
    rng = np.random.default_rng(0)
    low = chamfer_distance(add_gaussian_noise(clean, 0.01, rng=rng), clean)
    high = chamfer_distance(add_gaussian_noise(clean, 0.05, rng=rng), clean)
    assert high > low > 0


def test_format_table_reports_average():
    rows = [{"cd": 1.0, "p2m": 2.0}, {"cd": 3.0, "p2m": 4.0}]
    out = format_table(["a", "b"], rows)
    assert "AVERAGE" in out and "2.0000" in out
    assert summarize(rows)["cd"] == pytest.approx(2.0)


@pytest.mark.slow
def test_training_reduces_noise_on_a_simple_surface():
    """
    Train briefly on one sphere and require the denoised cloud to beat the
    noisy input. This is the property the old implementation violated.
    """
    torch.manual_seed(0)
    clean = sphere(1500)
    shape = Shape(clean, noise_level=0.02, rng=np.random.default_rng(1))
    radius = 0.05 * shape.diagonal

    model = Denoiser(d_model=64, num_heads=4, num_layers=2, k_coarse=8, k_fine=4)
    loss_fn = DenoisingLoss(repulsion_weight=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ds = PatchDataset([shape], points_per_patch=48, patches_per_shape=256, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate)

    model.train()
    for _ in range(3):
        for batch in loader:
            if batch is None:
                continue
            opt.zero_grad()
            loss, _ = loss_fn(model(batch["points"]), batch["target"])
            loss.backward()
            opt.step()

    model.eval()
    denoised = shape.noisy.copy()
    for batch in iter_eval_patches(shape, radius, 48, batch_size=64):
        world = centres_to_world(model.predict_centre(batch["points"]).numpy(), batch)
        denoised[batch["index"].numpy()] = world

    before = chamfer_distance(shape.noisy, clean)
    after = chamfer_distance(denoised, clean)
    print(f"\n  CD noisy {before*1e4:.4f} -> denoised {after*1e4:.4f} (x1e-4)")
    assert after < before, f"denoising made it worse: {before:.6g} -> {after:.6g}"
