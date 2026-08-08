"""
Classical denoising baselines.

These exist for two reasons. First, a learned method needs something to beat
that is not just "the noisy input". Second, and more importantly, they
calibrate the evaluation harness: the bilateral filter appears in the
published comparison tables, so if our pipeline scores it near the published
value then our normalization, Chamfer convention and noise model all match and
our own numbers can legitimately sit in the same table. If it does not match,
every comparison would be meaningless and it is better to find that out from a
baseline than from our own results.
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(points, k=16):
    """
    Per-point normals from the smallest-eigenvector of the local covariance.

    Orientation is left arbitrary (no global consistency pass) because the
    filters below only use the normal as an axis, never its sign.
    """
    points = np.asarray(points, dtype=np.float64)
    _, idx = cKDTree(points).query(points, k=min(k, len(points)))
    neighbourhoods = points[idx]
    centred = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / centred.shape[1]
    return np.linalg.eigh(cov)[1][:, :, 0]


def bilateral_filter(points, k=16, sigma_spatial=None, sigma_normal=None, iterations=1):
    """
    Bilateral point cloud filter (Fleishman et al. 2003 / Digne).

    Each point slides along its own normal by a weighted average of its
    neighbours' offsets along that normal. The weight combines a spatial
    Gaussian (near neighbours matter more) with a Gaussian on the offset
    itself, which is what preserves edges: a neighbour on the other side of a
    crease has a large normal-offset and is down-weighted, so the crease is not
    smoothed away.

    sigma_spatial defaults to the mean nearest-neighbour spacing and
    sigma_normal to twice that, both scaled from the data so the filter works
    on any object size.
    """
    pts = np.asarray(points, dtype=np.float64).copy()

    tree = cKDTree(pts)
    spacing = float(np.mean(tree.query(pts, k=2)[0][:, 1]))
    sigma_spatial = sigma_spatial if sigma_spatial is not None else spacing * 2.0
    sigma_normal = sigma_normal if sigma_normal is not None else spacing * 4.0

    for _ in range(iterations):
        tree = cKDTree(pts)
        normals = estimate_normals(pts, k=k)
        _, idx = tree.query(pts, k=min(k, len(pts)))

        offsets = pts[idx] - pts[:, None, :]                     # (N, k, 3)
        distance = np.linalg.norm(offsets, axis=2)               # (N, k)
        height = np.einsum("nkj,nj->nk", offsets, normals)       # signed, along normal

        w = np.exp(-(distance ** 2) / (2 * sigma_spatial ** 2)) * np.exp(
            -(height ** 2) / (2 * sigma_normal ** 2)
        )
        denom = w.sum(axis=1)
        denom[denom < 1e-12] = 1e-12
        shift = (w * height).sum(axis=1) / denom

        pts = pts + shift[:, None] * normals

    return pts


def laplacian_smooth(points, k=8, lam=0.5, iterations=1):
    """
    Move each point toward its neighbourhood centroid.

    The naive baseline: it removes noise and shrinks the object while doing
    so, which is exactly the failure a good denoiser has to avoid. Useful to
    have in a comparison table for that reason.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    for _ in range(iterations):
        _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
        pts = pts + lam * (pts[idx].mean(axis=1) - pts)
    return pts


BASELINES = {
    "bilateral": bilateral_filter,
    "laplacian": laplacian_smooth,
}
