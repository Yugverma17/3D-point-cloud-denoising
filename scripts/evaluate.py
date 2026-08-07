"""
Evaluate a checkpoint and write a results table.

    python scripts/evaluate.py --checkpoint runs/default/best.pt --data data/test

Scores the noisy input alongside the denoised output, because the only
question that matters is whether denoising helped, and a table without the
baseline cannot answer that.

Expects clean clouds as .npy in --data. If --meshes points at a directory of
matching .off/.ply files, P2M is reported too; P2M is the more honest metric
because Chamfer can be gamed by points clustering near ground-truth samples.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointdenoise.data import Shape
from pointdenoise.engine import evaluate, load_model, write_report


def find_mesh(mesh_dir, stem):
    if not mesh_dir:
        return None
    for ext in (".off", ".ply", ".obj"):
        candidate = Path(mesh_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--meshes", help="directory of ground-truth meshes for P2M")
    ap.add_argument("--out", default="results/scores.txt")
    ap.add_argument("--save-clouds", help="directory to write denoised .npy/.xyz")
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--points-per-patch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1,
                    help="refinement passes over the whole cloud")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-shapes", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(Path(args.data).glob("*.npy"))
    paths = [p for p in paths if not p.stem.endswith("_normal")]
    if args.limit_shapes:
        paths = paths[: args.limit_shapes]
    if not paths:
        raise FileNotFoundError(f"no .npy point clouds in {args.data}")

    rng = np.random.default_rng(args.seed)
    shapes, names, meshes = [], [], []
    for p in paths:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] != 3:
            print(f"  skipping {p.name}: expected (N, 3), got {pts.shape}")
            continue
        shapes.append(Shape(pts, noise_level=args.noise_level, rng=rng))
        names.append(p.stem)
        meshes.append(find_mesh(args.meshes, p.stem))

    if args.meshes and not any(meshes):
        print(f"warning: no meshes matched in {args.meshes}; reporting CD only")

    model, ckpt = load_model(args.checkpoint,
                             device=None if args.device == "auto" else args.device)
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    print(f"{len(shapes)} shapes at {args.noise_level:.0%} noise, {args.iters} pass(es)\n")

    rows, baseline = evaluate(
        model, shapes, names, meshes,
        points_per_patch=args.points_per_patch,
        batch_size=args.batch_size,
        iters=args.iters,
        device=None if args.device == "auto" else args.device,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print("\n" + write_report(args.out, names, rows, baseline))
    print(f"\nwritten to {args.out}")

    if args.save_clouds:
        from pointdenoise.engine import denoise_cloud

        out_dir = Path(args.save_clouds)
        out_dir.mkdir(parents=True, exist_ok=True)
        for shape, name in zip(shapes, names):
            pts = denoise_cloud(model, shape, points_per_patch=args.points_per_patch,
                                batch_size=args.batch_size, iters=args.iters)
            np.save(out_dir / f"{name}_denoised.npy", pts)
            np.savetxt(out_dir / f"{name}_denoised.xyz", pts, fmt="%.6f")
        print(f"clouds written to {out_dir}")


if __name__ == "__main__":
    main()
