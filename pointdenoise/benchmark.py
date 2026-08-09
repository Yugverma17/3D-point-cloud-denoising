"""
The PU-Net / PC-Net denoising benchmark.

This is the protocol behind the comparison tables in ScoreDenoise, PD-Flow,
I-PFN and P2P-Bridge: two test sets, two resolutions (10K "sparse" and 50K
"dense"), three Gaussian noise levels (1%, 2%, 3% of the bounding sphere
radius), scored by Chamfer distance and point-to-mesh, reported x1e-4.

Two ways to get the data:

1. Preferred - use the released test clouds (see docs/benchmark.md). Everyone
   in the table evaluated on those exact files, so using them removes any
   difference in resampling or noise draw.
2. Fallback - `resample_mesh` regenerates clouds from meshes. Convenient, but
   your noise draw and Poisson sampling will differ from theirs, so numbers
   are then only comparable within your own experiments.

`calibrate` is the guard against silently comparing incomparable numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Shape
from .geometry import normalize_unit_sphere
from .metrics import evaluate_shape, summarize

# Resolutions and noise levels used by the published tables.
RESOLUTIONS = {"sparse": 10_000, "dense": 50_000}
NOISE_LEVELS = (0.01, 0.02, 0.03)

# Published results for the PU-Net test set, CD / P2M in 1e-4 units, taken
# from the P2P-Bridge comparison table. Used by `calibrate` to check our
# harness agrees, and by `comparison_table` to place our own numbers in
# context. Keys are (resolution, noise_level).
PUBLISHED_PUNET = {
    "Bilateral":  {("sparse", 0.01): (3.65, 1.34), ("sparse", 0.02): (5.01, 2.02),
                   ("sparse", 0.03): (7.00, 3.56), ("dense", 0.01): (0.88, 0.23),
                   ("dense", 0.02): (2.38, 1.39), ("dense", 0.03): (6.30, 4.73)},
    "PCNet":      {("sparse", 0.01): (3.52, 1.15), ("sparse", 0.02): (7.47, 3.97),
                   ("sparse", 0.03): (13.1, 8.74), ("dense", 0.01): (1.05, 0.35),
                   ("dense", 0.02): (1.45, 0.61), ("dense", 0.03): (2.29, 1.29)},
    "DMRDenoise": {("sparse", 0.01): (4.48, 1.72), ("sparse", 0.02): (4.98, 2.12),
                   ("sparse", 0.03): (5.89, 2.85), ("dense", 0.01): (1.16, 0.47),
                   ("dense", 0.02): (1.57, 0.80), ("dense", 0.03): (2.43, 1.53)},
    "GLR":        {("sparse", 0.01): (2.96, 1.05), ("sparse", 0.02): (3.77, 1.31),
                   ("sparse", 0.03): (4.91, 2.11), ("dense", 0.01): (0.70, 0.16),
                   ("dense", 0.02): (1.59, 0.83), ("dense", 0.03): (3.84, 2.71)},
    "ScoreDenoise": {("sparse", 0.01): (2.52, 0.46), ("sparse", 0.02): (3.69, 1.07),
                     ("sparse", 0.03): (4.71, 1.94), ("dense", 0.01): (0.72, 0.15),
                     ("dense", 0.02): (1.29, 0.57), ("dense", 0.03): (1.93, 1.04)},
    "PD-Flow":    {("sparse", 0.01): (2.13, 0.38), ("sparse", 0.02): (3.25, 1.01),
                   ("sparse", 0.03): (5.19, 2.52), ("dense", 0.01): (0.65, 0.16),
                   ("dense", 0.02): (1.42, 0.78), ("dense", 0.03): (3.90, 2.86)},
    "I-PFN":      {("sparse", 0.01): (2.31, 0.37), ("sparse", 0.02): (3.43, 0.90),
                   ("sparse", 0.03): (5.24, 2.50), ("dense", 0.01): (0.66, 0.12),
                   ("dense", 0.02): (1.05, 0.43), ("dense", 0.03): (2.54, 1.65)},
    "P2P-Bridge": {("sparse", 0.01): (2.28, 0.39), ("sparse", 0.02): (3.20, 0.81),
                   ("sparse", 0.03): (3.99, 1.42), ("dense", 0.01): (0.59, 0.09),
                   ("dense", 0.02): (0.90, 0.32), ("dense", 0.03): (1.56, 0.84)},
}


@dataclass
class BenchmarkCase:
    """One cell of the results grid."""

    dataset: str
    resolution: str
    noise: float
    shapes: list = field(default_factory=list)
    names: list = field(default_factory=list)
    meshes: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.dataset}/{self.resolution}/{self.noise:.0%}"


def resample_mesh(mesh_path, num_points, seed=0):
    """
    Sample `num_points` roughly evenly over a mesh surface.

    Uses trimesh's even-sampling (approximate Poisson disk). The published
    sets use true Poisson-disk sampling, so clouds generated here are not
    bit-identical to theirs - fine for internal comparisons, not for dropping
    into their table.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    rng = np.random.default_rng(seed)
    with_seed = getattr(trimesh.util, "attach_to_log", None)  # noqa: F841 (keep import lint-quiet)
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points, seed=int(rng.integers(2**31)))

    # sample_surface_even can return fewer than requested; top up randomly
    if len(points) < num_points:
        extra, _ = trimesh.sample.sample_surface(mesh, num_points - len(points))
        points = np.vstack([points, extra])
    return np.asarray(points[:num_points], dtype=np.float64)


def add_benchmark_noise(points, level, rng=None):
    """
    Gaussian noise at `level` of the bounding-sphere radius.

    The published protocol normalizes each shape to the unit sphere first and
    then adds noise with sigma = level, which is what makes "1% noise" mean the
    same thing across objects of different sizes.
    """
    rng = rng or np.random.default_rng()
    normed, centre, scale = normalize_unit_sphere(points)
    noisy = normed + rng.normal(scale=level, size=normed.shape)
    return noisy * scale + centre


def load_released_set(root, dataset="PUNet", resolution="sparse", noise=0.01):
    """
    Load a released benchmark case from the ScoreDenoise data archive.

    The archive unpacks to:

        root/
          examples/PUNet_10000_poisson_0.01/   noisy test clouds, one .xyz each
          PUNet/pointclouds/test/10000_poisson/ the matching clean clouds
          PUNet/meshes/test/                    ground-truth meshes for P2M

    The noisy clouds live under `examples/` while their clean counterparts sit
    in a completely different subtree, which is easy to get wrong. Both are
    plain whitespace-separated xyz with one point per line and no header.
    """
    root = Path(root)
    n = RESOLUTIONS[resolution]

    noisy_dir = root / "examples" / f"{dataset}_{n}_poisson_{noise:.2f}"
    clean_dir = root / dataset / "pointclouds" / "test" / f"{n}_poisson"
    mesh_dir = root / dataset / "meshes" / "test"

    missing = [str(d) for d in (noisy_dir, clean_dir) if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            "benchmark data not found: " + ", ".join(missing) + "\n"
            "See docs/benchmark.md for where to get it and how to unpack it."
        )

    case = BenchmarkCase(dataset=dataset, resolution=resolution, noise=noise)
    for noisy_path in sorted(noisy_dir.glob("*.xyz")):
        stem = noisy_path.stem
        clean_path = clean_dir / f"{stem}.xyz"
        if not clean_path.exists():
            continue
        clean = np.loadtxt(clean_path)[:, :3]
        noisy = np.loadtxt(noisy_path)[:, :3]
        case.shapes.append(Shape(clean, noisy=noisy))
        case.names.append(stem)
        mesh = next(
            (mesh_dir / f"{stem}{e}" for e in (".off", ".ply", ".obj")
             if (mesh_dir / f"{stem}{e}").exists()),
            None,
        )
        case.meshes.append(mesh)

    if not case.shapes:
        raise RuntimeError(f"no clean/noisy pairs matched between {noisy_dir} and {clean_dir}")
    return case


def load_training_clouds(root, dataset="PUNet", resolution="sparse"):
    """
    Clean training clouds, as (name, points) pairs.

    Training uses only the clean shapes: noise is generated on the fly so one
    download covers every noise level, and the model sees a fresh draw each
    epoch rather than memorising one fixed corruption.
    """
    n = RESOLUTIONS.get(resolution, resolution)
    train_dir = Path(root) / dataset / "pointclouds" / "train" / f"{n}_poisson"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"training clouds not found at {train_dir}")
    return [(p.stem, np.loadtxt(p)[:, :3]) for p in sorted(train_dir.glob("*.xyz"))]


def build_case_from_meshes(mesh_dir, resolution="sparse", noise=0.01, dataset="local", seed=0):
    """Regenerate a case from meshes when the released clouds are unavailable."""
    mesh_paths = sorted(
        p for p in Path(mesh_dir).iterdir() if p.suffix.lower() in (".off", ".ply", ".obj")
    )
    if not mesh_paths:
        raise FileNotFoundError(f"no meshes in {mesh_dir}")

    n = RESOLUTIONS[resolution]
    rng = np.random.default_rng(seed)
    case = BenchmarkCase(dataset=dataset, resolution=resolution, noise=noise)
    for path in mesh_paths:
        clean = resample_mesh(path, n, seed=seed)
        case.shapes.append(Shape(clean, noisy=add_benchmark_noise(clean, noise, rng)))
        case.names.append(path.stem)
        case.meshes.append(path)
    return case


def run_case(case, denoise_fn, with_p2m=True, progress=None):
    """
    Apply `denoise_fn(points) -> points` to every shape and score it.

    Taking a plain callable means the learned model, a classical baseline and
    the no-op identity all go through exactly the same measurement path.
    """
    rows = []
    iterator = zip(case.shapes, case.names, case.meshes)
    if progress is not None:
        iterator = progress(list(iterator), desc=case.label)
    for shape, name, mesh in iterator:
        pred = denoise_fn(shape.noisy)
        rows.append(evaluate_shape(pred, shape.clean, mesh if with_p2m else None))
    return rows, summarize(rows)


#: Datasets whose published numbers we hold, so calibration is meaningful.
OFFICIAL_DATASETS = {"PUNet", "PCNet"}


def calibrate(case, tolerance=0.35, reference="Bilateral", published=None):
    """
    Check the harness reproduces a published number.

    Runs the bilateral filter, whose scores appear in the published tables, and
    compares. Agreement means our normalization, Chamfer convention and noise
    model line up with theirs and our own numbers may be quoted alongside.

    This only means anything on the *released* test set. Run it on substitute
    shapes and the difference between our number and theirs conflates the
    metric convention (what we want to test) with the shapes being easier or
    harder (what we do not), so the result is reported as inconclusive rather
    than pass or fail. Different shapes are not a calibration.

    `tolerance` is a deliberately loose relative band: implementations of
    "bilateral" differ in their parameter choices, so this is meant to catch a
    factor of 2 in the Chamfer convention or a wrong noise scale, not a 10%
    difference.
    """
    from .baselines import bilateral_filter

    published = published or PUBLISHED_PUNET
    key = (case.resolution, case.noise)
    if reference not in published or key not in published[reference]:
        raise KeyError(f"no published value for {reference} at {key}")

    expected_cd, expected_p2m = published[reference][key]
    _, ours = run_case(case, bilateral_filter, with_p2m=True)

    conclusive = case.dataset in OFFICIAL_DATASETS
    result = {
        "reference": reference,
        "case": case.label,
        "dataset": case.dataset,
        "num_shapes": len(case.shapes),
        "conclusive": conclusive,
        "expected_cd": expected_cd,
        "measured_cd": ours.get("cd"),
        "expected_p2m": expected_p2m,
        "measured_p2m": ours.get("p2m"),
    }

    # Every reported metric is checked separately. Gating on Chamfer alone was
    # a real mistake here: CD agreed at 0.84x while P2M sat at 0.17x for the
    # same algorithm on the same shapes, and the run passed. A metric that
    # disagrees by ~6x is measuring something different from what the table
    # reports, so its numbers must not be quoted even when CD is fine.
    for name, expected in (("cd", expected_cd), ("p2m", expected_p2m)):
        measured = ours.get(name)
        ratio = measured / expected if expected and measured is not None else float("nan")
        result[f"{name}_ratio"] = ratio
        result[f"{name}_ok"] = conclusive and abs(ratio - 1.0) <= tolerance

    # Kept for callers that only care about the headline metric.
    result["within_tolerance"] = result["cd_ok"]
    result["comparable_metrics"] = [m for m in ("cd", "p2m") if result[f"{m}_ok"]]
    result["uncalibrated_metrics"] = [m for m in ("cd", "p2m") if not result[f"{m}_ok"]]
    return result


def comparison_table(our_scores, our_name="Ours", dataset="PUNet", published=None):
    """
    Render our results next to the published ones.

    `our_scores` maps (resolution, noise) -> {"cd": float, "p2m": float}.
    Cells we have no number for are left blank rather than guessed.
    """
    published = dict(published or PUBLISHED_PUNET)
    published[our_name] = {
        k: (v.get("cd", float("nan")), v.get("p2m", float("nan")))
        for k, v in our_scores.items()
    }

    columns = [(r, n) for r in ("sparse", "dense") for n in NOISE_LEVELS]
    header = f"{dataset:<14}" + "".join(
        f"{res[:2] + '/' + str(int(lvl * 100)) + '%':>13}" for res, lvl in columns
    )
    lines = [header, "-" * len(header)]
    for method, cells in published.items():
        row = f"{method:<14}"
        for key in columns:
            if key in cells and np.isfinite(cells[key][0]):
                cd, p2m = cells[key]
                row += f"{cd:6.2f}/{p2m:<6.2f}"
            else:
                row += f"{'-':>13}"
        lines.append(row)
    lines.append("")
    lines.append("Each cell is CD / P2M, x1e-4. Lower is better.")
    return "\n".join(lines)
