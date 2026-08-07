"""
Train the denoiser.

    python scripts/train.py --data data/train --epochs 50

Expects a directory of clean point clouds saved as .npy of shape (N, 3).
Noise is generated on the fly at the requested level, so one clean dataset
covers every noise setting.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointdenoise.data import Shape
from pointdenoise.engine import train


def load_shapes(data_dir, noise_level, limit=None, seed=0):
    paths = sorted(Path(data_dir).glob("*.npy"))
    paths = [p for p in paths if not p.stem.endswith("_normal")]
    if not paths:
        raise FileNotFoundError(f"no .npy point clouds in {data_dir}")
    if limit:
        paths = paths[:limit]

    rng = np.random.default_rng(seed)
    shapes = []
    for p in paths:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] != 3:
            print(f"  skipping {p.name}: expected (N, 3), got {pts.shape}")
            continue
        shapes.append(Shape(pts, noise_level=noise_level, rng=rng))
    print(f"loaded {len(shapes)} shapes from {data_dir}")
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file; command-line flags override it")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--points-per-patch", type=int, default=256)
    ap.add_argument("--patches-per-shape", type=int, default=1000)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--repulsion-weight", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume")
    ap.add_argument("--limit-shapes", type=int)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        defaults = {a.dest: a.default for a in ap._actions}
        for key, value in cfg.items():
            # only apply the config where the user did not pass a flag
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    shapes = load_shapes(args.data, args.noise_level, args.limit_shapes, args.seed)

    train(
        shapes,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        points_per_patch=args.points_per_patch,
        patches_per_shape=args.patches_per_shape,
        lr=args.lr,
        repulsion_weight=args.repulsion_weight,
        model_kwargs={
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        },
        device=None if args.device == "auto" else args.device,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"\ndone. checkpoints in {args.out}")


if __name__ == "__main__":
    main()
