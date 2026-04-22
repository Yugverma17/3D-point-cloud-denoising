import os
import numpy as np

from new_Pointfilter_Utils import chamfer_distance, calculate_p2m


clean_dir = "./results2/clean_resampled"
denoised_dir = "./results2/denoised_2p"
mesh_dir = "./Dataset/off_files_test"
scores_path = "scores.txt"

cd_scores = []
p2m_scores = []
shape_names = []

if not os.path.isdir(denoised_dir):
    raise FileNotFoundError(f"Denoised results directory not found: {denoised_dir}")

for filename in sorted(os.listdir(denoised_dir)):
    if not filename.endswith(".npy") or "_denoised_" not in filename:
        continue

    shape_name = filename.split("_denoised_")[0]
    clean_path = os.path.join(clean_dir, f"{shape_name}.npy")
    denoised_path = os.path.join(denoised_dir, filename)
    mesh_path = os.path.join(mesh_dir, f"{shape_name}.off")

    if not os.path.exists(clean_path) or not os.path.exists(mesh_path):
        print(f"Skipping {shape_name} - missing clean or mesh file")
        continue

    clean = np.load(clean_path)
    denoised = np.load(denoised_path)

    cd = chamfer_distance(denoised, clean)
    p2m = calculate_p2m(denoised, mesh_path)
    if np.isnan(cd) or np.isnan(p2m):
        print(f"Skipping {shape_name} - metric calculation returned NaN")
        continue

    cd_scores.append(cd)
    p2m_scores.append(p2m)
    shape_names.append(shape_name)

    print(f"{shape_name:20s} | CD: {cd * 1e4:.4f} | P2M: {p2m * 1e4:.4f} (x1e-4)")

if not shape_names:
    raise RuntimeError("No valid denoised shapes were found for scoring.")

print("\n" + "=" * 60)
print(f"{'AVERAGE':20s} | CD: {np.mean(cd_scores) * 1e4:.4f} | P2M: {np.mean(p2m_scores) * 1e4:.4f} (x1e-4)")
print(f"{'BEST CD':20s} | {shape_names[int(np.argmin(cd_scores))]} -> {min(cd_scores) * 1e4:.4f}")
print(f"{'WORST CD':20s} | {shape_names[int(np.argmax(cd_scores))]} -> {max(cd_scores) * 1e4:.4f}")
print("=" * 60)

with open(scores_path, "w") as out:
    out.write(f"{'Shape':20s} | {'CD (x1e-4)':>12} | {'P2M (x1e-4)':>12}\n")
    out.write("-" * 54 + "\n")
    for name, cd, p2m in zip(shape_names, cd_scores, p2m_scores):
        out.write(f"{name:20s} | {cd * 1e4:12.4f} | {p2m * 1e4:12.4f}\n")
    out.write("-" * 54 + "\n")
    out.write(f"{'AVERAGE':20s} | {np.mean(cd_scores) * 1e4:12.4f} | {np.mean(p2m_scores) * 1e4:12.4f}\n")

print(f"\nScores saved to {scores_path}")
