"""
Run the PU-Net / PC-Net benchmark grid.

    # verify the harness agrees with a published number first
    python scripts/benchmark.py --data data/benchmark --calibrate

    # then score a checkpoint across the whole grid
    python scripts/benchmark.py --data data/benchmark --checkpoint runs/exp1/best.pt

    # or score a classical baseline
    python scripts/benchmark.py --data data/benchmark --method bilateral

Always run --calibrate before quoting numbers anywhere. It scores the
bilateral filter, whose results are in the published tables, and tells you
whether this harness reproduces them. If it does not, the comparison table is
meaningless no matter how good the model is.
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointdenoise.baselines import BASELINES
from pointdenoise.benchmark import (
    NOISE_LEVELS,
    build_case_from_meshes,
    calibrate,
    comparison_table,
    load_released_set,
    run_case,
)


def make_denoiser(args):
    """Return (name, callable) taking (N,3) points to (N,3) points."""
    if args.method in BASELINES:
        fn = BASELINES[args.method]
        return args.method, lambda pts: fn(pts, iterations=args.iters)

    if args.method == "identity":
        return "noisy input", lambda pts: pts

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --method is given")

    import numpy as np

    from pointdenoise.data import Shape
    from pointdenoise.engine import denoise_cloud, load_model

    model, ckpt = load_model(args.checkpoint,
                             device=None if args.device == "auto" else args.device)
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    def run(points):
        # Shape wants a clean cloud too; for inference it is never read.
        shape = Shape(np.asarray(points), noisy=np.asarray(points))
        return denoise_cloud(model, shape, points_per_patch=args.points_per_patch,
                             batch_size=args.batch_size, iters=args.iters,
                             device=None if args.device == "auto" else args.device)

    return Path(args.checkpoint).parent.name, run


def load_case(args, resolution, noise):
    if args.meshes:
        return build_case_from_meshes(args.meshes, resolution, noise, seed=args.seed)
    return load_released_set(args.data, args.dataset, resolution, noise)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/benchmark",
                    help="directory holding the released benchmark clouds")
    ap.add_argument("--meshes", help="regenerate clouds from meshes instead (not "
                                     "comparable to published numbers)")
    ap.add_argument("--dataset", default="PUNet", choices=["PUNet", "PCNet"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--method", choices=[*BASELINES, "identity"],
                    help="score a classical baseline instead of a checkpoint")
    ap.add_argument("--calibrate", action="store_true",
                    help="check the harness against a published number and exit")
    ap.add_argument("--resolutions", nargs="+", default=["sparse", "dense"])
    ap.add_argument("--noise", nargs="+", type=float, default=list(NOISE_LEVELS))
    ap.add_argument("--points-per-patch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--no-p2m", action="store_true", help="skip P2M (much faster)")
    ap.add_argument("--out", default="results/benchmark.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.calibrate:
        case = load_case(args, "sparse", 0.01)
        print(f"calibrating on {case.label} ({len(case.shapes)} shapes)\n")
        result = calibrate(case)
        print(f"  reference       {result['reference']}")
        print(f"  published CD    {result['expected_cd']:.2f}")
        print(f"  our harness CD  {result['measured_cd']:.2f}   "
              f"(ratio {result['cd_ratio']:.2f}x)")
        print(f"  published P2M   {result['expected_p2m']:.2f}")
        print(f"  our harness P2M {result['measured_p2m']:.2f}")
        print()
        if not result["conclusive"]:
            print(f"  INCONCLUSIVE - ran on '{result['dataset']}' "
                  f"({result['num_shapes']} shapes), not the released test set.")
            print("  The published numbers are for specific shapes, so any difference")
            print("  here mixes up the metric convention with the shapes simply being")
            print("  easier or harder. That is not a calibration.")
            print("  Get the released data first: see docs/benchmark.md")
            raise SystemExit(2)

        ok = result["comparable_metrics"]
        bad = result["uncalibrated_metrics"]
        for m in ("cd", "p2m"):
            verdict = "PASS" if result[f"{m}_ok"] else "FAIL"
            print(f"  {m.upper():<4} ratio {result[f'{m}_ratio']:.2f}x  {verdict}")
        print()
        if ok:
            print(f"  QUOTABLE: {', '.join(m.upper() for m in ok)}")
        if bad:
            print(f"  DO NOT QUOTE: {', '.join(m.upper() for m in bad)}")
            print("  Those metrics measure something different from the published table.")
            print("  A ratio near 2 or 0.5 usually means a squared/unsquared or one-sided")
            print("  convention difference; a larger gap means a different definition.")
        raise SystemExit(0 if not bad else 1)

    name, denoise_fn = make_denoiser(args)
    scores, lines = {}, []

    for resolution in args.resolutions:
        for noise in args.noise:
            try:
                case = load_case(args, resolution, noise)
            except FileNotFoundError as exc:
                print(f"skipping {resolution}/{noise:.0%}: {exc}")
                continue
            rows, avg = run_case(case, denoise_fn, with_p2m=not args.no_p2m, progress=tqdm)
            scores[(resolution, noise)] = avg
            summary = "  ".join(f"{k.upper()} {v:.4f}" for k, v in avg.items())
            print(f"{case.label:<24} {summary}")
            lines.append(f"{case.label:<24} {summary}")

    if not scores:
        raise SystemExit("no benchmark cases could be loaded; see docs/benchmark.md")

    table = comparison_table(scores, our_name=name, dataset=args.dataset)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n\n" + table + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
