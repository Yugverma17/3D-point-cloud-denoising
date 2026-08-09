"""
Patch extraction.

Each point of a cloud defines one patch: its neighbours within a radius,
expressed in the patch-local frame from `geometry.align_patch`. The patch
centre is always stored at index 0 so evaluation can read off exactly one
prediction per point.

The ground-truth target is rotated by the SAME R as the noisy input. Getting
this wrong is silent - training still runs and the loss still falls - so it is
covered by tests/test_geometry.py rather than left to inspection.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .geometry import add_gaussian_noise, align_patch, bbox_diagonal


class Shape:
    """One clean cloud, its noisy version, and KD-trees over both."""

    def __init__(self, clean, noisy=None, noise_level=0.02, rng=None):
        self.clean = np.asarray(clean, dtype=np.float64)
        self.diagonal = bbox_diagonal(self.clean)
        if noisy is None:
            noisy = add_gaussian_noise(self.clean, level=noise_level, rng=rng)
        self.noisy = np.asarray(noisy, dtype=np.float64)
        self.clean_tree = cKDTree(self.clean)
        self.noisy_tree = cKDTree(self.noisy)

    @classmethod
    def from_npy(cls, clean_path, noisy_path=None, **kw):
        noisy = np.load(noisy_path) if noisy_path else None
        return cls(np.load(clean_path), noisy, **kw)


def extract_patch(shape, index, radius, num_points, rng, with_target=True):
    """
    One training/eval sample.

    Neighbours are the `num_points` nearest, not everything inside a fixed
    radius. A fixed radius silently drops points wherever the cloud is sparser
    than expected - on a 1500-point sphere a 5%-of-bbox radius holds only ~4
    neighbours, so a third of all points got no patch at all and stayed noisy,
    which wrecked the coverage half of Chamfer distance even though the points
    that *were* denoised landed closer to the surface. k-NN adapts to local
    density and guarantees every point is covered exactly once.

    `radius` is still used to scale the patch into roughly unit size, and is
    taken from the actual neighbourhood rather than assumed.
    """
    centre = shape.noisy[index]
    k = min(num_points, len(shape.noisy))
    _, idx = shape.noisy_tree.query(centre, k=k)
    idx = np.atleast_1d(idx)

    # Centre point first, so evaluation can read one prediction per point.
    idx = np.concatenate([[index], idx[idx != index]])[:k]
    if len(idx) < 4:
        return None
    if len(idx) < num_points:
        pad = rng.choice(len(idx), num_points - len(idx), replace=True)
        idx = np.concatenate([idx, idx[pad]])

    # Scale by the patch's own extent so the network always sees a similar
    # size regardless of how dense this part of the cloud is.
    local_radius = float(np.linalg.norm(shape.noisy[idx] - centre, axis=1).max())
    radius = local_radius if local_radius > 1e-12 else radius

    aligned, R = align_patch(shape.noisy[idx], centre=centre, radius=radius)
    sample = {
        "points": torch.from_numpy(aligned).float(),
        "rotation": torch.from_numpy(R).float(),
        "centre": torch.from_numpy(np.asarray(centre)).float(),
        "radius": float(radius),
        "index": int(index),
    }

    if with_target:
        # Look slightly wider than the patch so points near its edge still have
        # clean surface on all sides to match against.
        gt_idx = shape.clean_tree.query_ball_point(centre, radius * 1.5)
        if len(gt_idx) < 3:
            gt_idx = shape.clean_tree.query(centre, k=min(num_points, len(shape.clean)))[1]
            gt_idx = np.atleast_1d(gt_idx)
        if len(gt_idx) < 3:
            return None
        # SAME R, same centre, same radius as the input above.
        gt_local = ((shape.clean[gt_idx] - centre) @ R.T) / radius
        # Target for each input point is its nearest clean surface point.
        nearest = cKDTree(gt_local).query(aligned)[1]
        sample["target"] = torch.from_numpy(gt_local[nearest]).float()

    return sample


class PatchDataset(Dataset):
    """
    Patches drawn from a list of shapes.

    `noise_range` re-corrupts each shape from its clean cloud at a randomly
    drawn level instead of reusing whatever noise the Shape was built with.
    Training at a single fixed level teaches the model one correction size and
    it applies that regardless: a model trained only at 2% improved the
    benchmark by 56% at 2% and 73% at 3%, but only 2% at 1% noise, because it
    kept displacing points that were already nearly right. Sampling the range
    the benchmark actually tests is what fixes the low-noise column.

    Pass `noise_range=None` to keep each Shape's existing noise, which is what
    evaluation wants.
    """

    def __init__(
        self,
        shapes,
        patch_radius=0.05,
        points_per_patch=256,
        patches_per_shape=1000,
        seed=0,
        with_target=True,
        noise_range=(0.005, 0.03),
        resample_every=1,
    ):
        self.shapes = shapes
        self.points_per_patch = points_per_patch
        self.patches_per_shape = patches_per_shape
        self.with_target = with_target
        self.noise_range = noise_range
        # Rebuilding a Shape means rebuilding its KD-tree, which is not free,
        # so one draw is shared across this many consecutive patches.
        self.resample_every = max(1, resample_every)
        # radius is a fraction of each shape's own bounding box, so a fixed
        # value means the same physical scale across differently-sized objects
        self.radii = [patch_radius * s.diagonal for s in shapes]
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def __len__(self):
        return len(self.shapes) * self.patches_per_shape

    def _shape_for(self, shape_i, i):
        """The shape to sample from, re-noised if a fresh draw is due."""
        if self.noise_range is None:
            return self.shapes[shape_i]

        bucket = i // self.resample_every
        cached = self._cache.get(shape_i)
        if cached is not None and cached[0] == bucket:
            return cached[1]

        level = self.rng.uniform(*self.noise_range)
        noised = Shape(self.shapes[shape_i].clean, noise_level=level, rng=self.rng)
        self._cache[shape_i] = (bucket, noised)
        return noised

    def __getitem__(self, i):
        shape_i = i // self.patches_per_shape
        shape = self._shape_for(shape_i, i)
        point_i = self.rng.integers(len(shape.noisy))
        return extract_patch(
            shape,
            point_i,
            self.radii[shape_i],
            self.points_per_patch,
            self.rng,
            with_target=self.with_target,
        )


def collate(batch):
    """Drops samples that came back None; returns None if nothing survives."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {
        "points": torch.stack([b["points"] for b in batch]),
        "rotation": torch.stack([b["rotation"] for b in batch]),
        "centre": torch.stack([b["centre"] for b in batch]),
        "radius": torch.tensor([b["radius"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch]),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])
    return out


def centres_to_world(local_centres, batch):
    """
    Map per-patch centre predictions back to world coordinates.

    `local_centres` is (B, 3) in each patch's own frame; returns (B, 3) in world
    space using the rotation/centre/radius carried alongside the patch.
    """
    local = np.asarray(local_centres, dtype=np.float64)
    R = batch["rotation"].cpu().numpy()
    centre = batch["centre"].cpu().numpy()
    radius = batch["radius"].cpu().numpy()[:, None]
    scaled = local * radius
    rotated = np.einsum("bi,bij->bj", scaled, R)
    return rotated + centre


def iter_eval_patches(shape, radius, num_points, batch_size=64, seed=0):
    """
    Yield batches covering every point of a cloud exactly once, for evaluation.

    Each yielded batch carries the patch tensor plus what is needed to map the
    centre prediction back to world coordinates.
    """
    rng = np.random.default_rng(seed)
    buffer = []
    for i in range(len(shape.noisy)):
        sample = extract_patch(shape, i, radius, num_points, rng, with_target=False)
        if sample is None:
            continue
        buffer.append(sample)
        if len(buffer) == batch_size:
            yield collate(buffer)
            buffer = []
    if buffer:
        yield collate(buffer)
