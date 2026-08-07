"""
Frame-convention tests.

The first test here is the regression test for the bug that made the previous
version of this project degrade point clouds instead of denoising them: the
noisy patch was rotated by R while its ground-truth target was rotated by R.T.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pointdenoise.geometry import (
    add_gaussian_noise,
    align_patch,
    bbox_diagonal,
    normalize_unit_sphere,
    principal_rotation,
    unalign_patch,
)


def mean_nn_distance(a, b):
    return float(cKDTree(b).query(a)[0].mean())


@pytest.fixture
def patch():
    rng = np.random.default_rng(0)
    # anisotropic slab so the principal directions are well separated
    clean = rng.normal(size=(300, 3)) * np.array([1.0, 0.5, 0.02])
    noisy = clean + rng.normal(scale=0.01, size=clean.shape)
    return clean, noisy


def test_noisy_and_clean_land_in_the_same_frame(patch):
    """
    Rotating the input and the target by the same R must not move them apart.

    The old code applied R to the noisy patch and R.T (returned as
    "inv_transform") to the clean target. That inflated the input/target gap
    well beyond the actual noise, and differently for every patch.
    """
    clean, noisy = patch
    gap_before = mean_nn_distance(noisy, clean)

    aligned_noisy, R = align_patch(noisy)
    aligned_clean = clean @ R.T

    assert mean_nn_distance(aligned_noisy, aligned_clean) == pytest.approx(gap_before, rel=1e-6)

    # and the specific mistake is caught
    wrongly_aligned_clean = clean @ R
    assert mean_nn_distance(aligned_noisy, wrongly_aligned_clean) > 2 * gap_before


def test_rotation_is_orthonormal_and_right_handed(patch):
    _, noisy = patch
    R = principal_rotation(noisy)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_align_unalign_round_trip(patch):
    _, noisy = patch
    centre = noisy[0]
    radius = 0.5
    aligned, R = align_patch(noisy, centre=centre, radius=radius)
    recovered = unalign_patch(aligned, R, centre=centre, radius=radius)
    assert np.allclose(recovered, noisy, atol=1e-9)


def test_frame_is_stable_under_small_perturbation():
    """
    SVD sign ambiguity would otherwise flip the frame between near-identical
    patches, which turns a smooth problem into a discontinuous one.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(400, 3)) * np.array([1.0, 0.4, 0.05])
    R1 = principal_rotation(base)
    R2 = principal_rotation(base + rng.normal(scale=1e-4, size=base.shape))
    assert np.allclose(R1, R2, atol=1e-2)


def test_noise_level_is_relative_to_bbox():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(2000, 3))
    diag = bbox_diagonal(pts)
    noisy = add_gaussian_noise(pts, level=0.02, rng=rng)
    displacement = np.linalg.norm(noisy - pts, axis=1).mean()
    # mean |N(0, s)| in 3D is s*sqrt(8/pi); just check the right order
    assert 0.5 * 0.02 * diag < displacement < 3.0 * 0.02 * diag


def test_unit_sphere_normalization():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(500, 3)) * 7.0 + 30.0
    normed, centre, scale = normalize_unit_sphere(pts)
    assert np.linalg.norm(normed, axis=1).max() == pytest.approx(1.0)
    assert np.allclose(normed * scale + centre, pts, atol=1e-9)
