"""
Patch-local coordinate frames and noise models.

The frame convention here is the thing that was silently broken in the earlier
version of this project, so it is stated explicitly and covered by tests.

A patch is expressed in a local frame so the network sees every patch in a
canonical orientation instead of having to learn rotation invariance. Building
that frame gives a rotation R:

    local = (R @ world.T).T = world @ R.T

`align_patch` returns the aligned points together with R itself. To go back:

    world = (R.T @ local.T).T = local @ R

Both the noisy patch and its ground-truth target must be rotated by the SAME
R. Rotating one by R and the other by R.T puts them in different frames, and
because R is derived per-patch the mismatch differs for every sample, so the
mapping the network is asked to learn is not a function. Training silently
converges to something that actively displaces points. See
tests/test_geometry.py::test_noisy_and_clean_land_in_the_same_frame.
"""

import numpy as np


def principal_rotation(points):
    """
    Rotation R whose rows are the patch's principal directions, ordered from
    most to least variance.

    Uses SVD directly rather than sklearn's PCA so the package has one fewer
    dependency and the sign convention below is explicit.
    """
    centred = points - points.mean(axis=0, keepdims=True)
    # rows of Vt are the principal directions, largest variance first
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    R = Vt

    # SVD leaves each direction's sign arbitrary, which would make the frame
    # flip between near-identical patches. Pin it down by forcing the largest
    # component of each axis positive, then keep the frame right-handed so R
    # stays a rotation rather than a reflection.
    signs = np.sign(R[np.arange(3), np.abs(R).argmax(axis=1)])
    signs[signs == 0] = 1.0
    R = R * signs[:, None]
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    return R


def align_patch(points, centre=None, radius=None):
    """
    Express `points` in the patch-local frame.

    Returns (aligned, R). If `centre` is given it is subtracted first; if
    `radius` is given the result is divided by it so the patch roughly fills
    the unit ball. Undo with `unalign_patch` using the same arguments.
    """
    p = np.asarray(points, dtype=np.float64)
    if centre is not None:
        p = p - centre
    R = principal_rotation(p)
    aligned = p @ R.T
    if radius is not None:
        aligned = aligned / radius
    return aligned, R


def unalign_patch(aligned, R, centre=None, radius=None):
    """Inverse of `align_patch`, given the same R/centre/radius."""
    p = np.asarray(aligned, dtype=np.float64)
    if radius is not None:
        p = p * radius
    world = p @ R
    if centre is not None:
        world = world + centre
    return world


# ---------------------------------------------------------------------------
# Noise models
#
# The benchmark this project targets reports Gaussian noise as a percentage of
# the shape's bounding-box diagonal, so `scale_by_bbox` is the default and the
# level is given as e.g. 0.01 for 1%.
# ---------------------------------------------------------------------------


def bbox_diagonal(points):
    p = np.asarray(points)
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))


def add_gaussian_noise(points, level=0.02, scale_by_bbox=True, rng=None):
    rng = rng or np.random.default_rng()
    p = np.asarray(points, dtype=np.float64)
    sigma = level * bbox_diagonal(p) if scale_by_bbox else level
    return p + rng.normal(scale=sigma, size=p.shape)


def normalize_unit_sphere(points):
    """
    Centre at the origin and scale so the farthest point sits at radius 1.

    Returns (normalized, centre, scale) so metrics can be reported in the same
    normalized units the published benchmarks use.
    """
    p = np.asarray(points, dtype=np.float64)
    centre = p.mean(axis=0, keepdims=True)
    centred = p - centre
    scale = float(np.linalg.norm(centred, axis=1).max())
    return centred / scale, centre, scale
