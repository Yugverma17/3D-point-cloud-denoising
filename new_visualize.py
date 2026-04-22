import os
import numpy as np
import matplotlib.pyplot as plt


noisy_dir = "./results2/noisy_2p"
denoised_dir = "./results2/denoised_2p"
clean_dir = "./results2/clean_resampled"
output_dir = "./results2/visualizations"
os.makedirs(output_dir, exist_ok=True)


def visualize_shape(shape_name):
    noisy = np.load(os.path.join(noisy_dir, f"{shape_name}_noisy.npy"))

    denoised_candidates = [
        os.path.join(denoised_dir, f"{shape_name}_denoised_2p.npy"),
        os.path.join(denoised_dir, f"{shape_name}_denoised_2.npy"),
    ]
    denoised_path = next((path for path in denoised_candidates if os.path.exists(path)), None)
    if denoised_path is None:
        raise FileNotFoundError(f"No denoised file found for {shape_name} in {denoised_dir}")

    denoised = np.load(denoised_path)
    clean = np.load(os.path.join(clean_dir, f"{shape_name}.npy"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"})
    fig.suptitle(f"Shape: {shape_name}", fontsize=16)

    datasets = [
        (noisy, "Noisy Input", "red"),
        (denoised, "Denoised (Ours)", "green"),
        (clean, "Clean GT", "blue"),
    ]

    for ax, (pts, title, color) in zip(axes, datasets):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=0.1, alpha=0.5)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{shape_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


shapes = [
    "camel", "cow", "horse", "duck", "elephant",
    "chair", "pig", "kitten", "moai", "sculpt",
    "eight", "elk", "fandisk", "casting", "star",
    "genus3", "quadric", "coverrear_Lp", "Icosahedron", "Octahedron",
]

for shape in shapes:
    try:
        visualize_shape(shape)
    except Exception as exc:
        print(f"Skipped {shape}: {exc}")

print("\nAll done. Check results2/visualizations/")
