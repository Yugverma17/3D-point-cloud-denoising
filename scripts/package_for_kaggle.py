"""
Build the two zips to upload to Kaggle.

    python scripts/package_for_kaggle.py

Writes to kaggle/:
  pointdenoise-code.zip   the package plus configs, a few hundred KB
  pointdenoise-data.zip   the unpacked benchmark data

They are separate on purpose. Code changes every time you touch the model;
the data never changes. Uploading them as two Kaggle datasets means you
re-upload a few hundred KB per iteration instead of 200 MB.
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def zip_code(out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "pointdenoise").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            z.write(path, path.relative_to(ROOT))
            written += 1
        for extra in ("configs/default.yaml", "requirements.txt"):
            p = ROOT / extra
            if p.exists():
                z.write(p, extra)
                written += 1
    return written


def zip_data(data_dir, out_path):
    """
    Zip only what training and benchmarking read.

    Skips the 30K resolution (unused - the benchmark is 10K and 50K) and the
    RueMadame scan, which is a real-world example rather than part of the
    benchmark. That keeps the upload well under Kaggle's limits.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} not found; see docs/benchmark.md")

    def wanted(path):
        parts = path.parts
        if "30000_poisson" in parts or "RueMadame" in parts:
            return False
        if path.suffix.lower() not in (".xyz", ".off", ".ply", ".obj"):
            return False
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(data_dir.rglob("*")):
            if path.is_file() and wanted(path):
                z.write(path, path.relative_to(data_dir))
                written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "kaggle"))
    ap.add_argument("--code-only", action="store_true",
                    help="skip the data zip; use after the first upload")
    args = ap.parse_args()

    out = Path(args.out)
    code_zip = out / "pointdenoise-code.zip"
    n = zip_code(code_zip)
    print(f"{code_zip}  {n} files, {code_zip.stat().st_size/1e3:.0f} KB")

    if not args.code_only:
        data_zip = out / "pointdenoise-data.zip"
        n = zip_data(args.data, data_zip)
        print(f"{data_zip}  {n} files, {data_zip.stat().st_size/1e6:.0f} MB")

    print("\nUpload both as Kaggle datasets, then run kaggle/train_pointdenoise.ipynb")
    print("After the first upload, re-run with --code-only when you change the model.")


if __name__ == "__main__":
    main()
