"""
Chamfer distance and point-to-mesh distance.

Numbers from point-cloud denoising papers are only comparable if three things
match: the normalization, the Chamfer convention, and the reporting scale. The
published benchmark this project targets (the PU-Net / PC-Net tables used by
ScoreDenoise, PD-Flow, I-PFN, P2P-Bridge) uses:

  * both clouds normalized to the unit sphere before measuring, so results do
    not depend on the object's original scale,
  * Chamfer as the mean squared nearest-neighbour distance in BOTH directions,
    summed,
  * results reported multiplied by 1e4.

`chamfer_distance` implements exactly that. If you find a paper using the
half-sum convention (0.5 * each direction), its numbers are 2x smaller and are
not directly comparable - `halve=True` is provided so you can state which
convention a given table uses rather than silently mixing them.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import normalize_unit_sphere


def chamfer_distance(pred, gt, normalize=True, halve=False):
    """
    Symmetric Chamfer distance between two point sets.

    pred, gt: (N, 3) and (M, 3) arrays. Returns a float in normalized units;
    multiply by 1e4 for the scale used in published tables.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if normalize:
        pred, _, _ = normalize_unit_sphere(pred)
        gt, _, _ = normalize_unit_sphere(gt)

    d_pred_to_gt = cKDTree(gt).query(pred)[0]
    d_gt_to_pred = cKDTree(pred).query(gt)[0]

    cd = np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)
    return float(0.5 * cd if halve else cd)


def point_to_mesh_distance(pred, mesh, normalize=True):
    """
    Mean distance from each predicted point to the ground-truth mesh surface.

    P2M is the more honest of the two metrics: Chamfer can be gamed by points
    clustering near sampled ground-truth points, whereas P2M measures distance
    to the true surface. `mesh` is a path or a trimesh.Trimesh.

    The mesh is normalized with the SAME centre and scale as the prediction,
    not independently - normalizing them separately would compare two
    differently-scaled objects.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load(str(mesh), process=False, force="mesh")

    pred = np.asarray(pred, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if normalize:
        pred, centre, scale = normalize_unit_sphere(pred)
        vertices = (vertices - centre) / scale
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    closest, distance, _ = trimesh.proximity.closest_point(mesh, pred)
    return float(np.mean(distance ** 2))


def evaluate_shape(pred, gt_points, mesh_path=None, halve=False):
    """CD and (if a mesh is given) P2M for one shape, in 1e-4 units."""
    result = {"cd": chamfer_distance(pred, gt_points, halve=halve) * 1e4}
    if mesh_path is not None:
        result["p2m"] = point_to_mesh_distance(pred, mesh_path) * 1e4
    return result


def summarize(rows):
    """Mean of each metric across shapes. `rows` is a list of dicts."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def format_table(names, rows, title=None):
    """Plain-text results table, the format used in results/ files."""
    metric_keys = [k for k in ("cd", "p2m") if rows and k in rows[0]]
    header = f"{'Shape':<20}" + "".join(f"{k.upper() + ' (x1e-4)':>16}" for k in metric_keys)
    lines = ([title] if title else []) + [header, "-" * len(header)]
    for name, row in zip(names, rows):
        lines.append(f"{name:<20}" + "".join(f"{row[k]:16.4f}" for k in metric_keys))
    avg = summarize(rows)
    lines.append("-" * len(header))
    lines.append(f"{'AVERAGE':<20}" + "".join(f"{avg[k]:16.4f}" for k in metric_keys))
    return "\n".join(lines)
