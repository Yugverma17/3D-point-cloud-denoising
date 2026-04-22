# test_TDNetDenoiser_improved.py
import os
import torch
import numpy as np
from tqdm import tqdm

from new_Pointfilter_DataLoader import PointcloudPatchDataset
from new_Pointfilter_Utils import parse_arguments, add_noise_to_batch, chamfer_distance, calculate_p2m
from new_Pointfilter_Network_Architecture_2 import TDNetDenoiser


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def evaluation_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = [d[key] for d in batch]

    collated_batch["points"] = torch.stack(collated_batch["points"], 0)
    collated_batch["noise_inv"] = torch.stack(collated_batch["noise_inv"], 0)
    collated_batch["noise_disp"] = torch.stack(collated_batch["noise_disp"], 0)
    return collated_batch


def prepare_resampled_dataset(original_clean_dir, target_dir, num_points):
    """Uniformly resample all .npy point clouds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Preparing dataset with {num_points} points ---")
    os.makedirs(target_dir, exist_ok=True)
    files = [f for f in os.listdir(original_clean_dir) if f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {original_clean_dir}")

    for fname in files:
        src = np.load(os.path.join(original_clean_dir, fname))
        if len(src) < num_points:
            idx = np.random.choice(len(src), num_points, replace=True)
        else:
            pts = torch.from_numpy(src).float().unsqueeze(0).to(device)
            idx = farthest_point_sample(pts, num_points)[0].cpu().numpy()
        np.save(os.path.join(target_dir, fname), src[idx])

    print(f"Resampled {len(files)} shapes into {target_dir}")
    return target_dir


def create_noisy_data_and_list(clean_dir, noisy_dir, list_path, noise_level=0.02):
    """Generate Gaussian noisy dataset."""
    os.makedirs(noisy_dir, exist_ok=True)
    files = [f for f in os.listdir(clean_dir) if f.endswith(".npy")]
    names = []

    for filename in files:
        base = filename.replace(".npy", "")
        noisy_name = base + "_noisy"
        names.append(noisy_name)
        clean = np.load(os.path.join(clean_dir, filename))
        noisy = add_noise_to_batch(torch.from_numpy(clean).float(), "gaussian", noise_level).numpy()
        np.save(os.path.join(noisy_dir, f"{noisy_name}.npy"), noisy)

    with open(list_path, "w") as file_obj:
        file_obj.write("\n".join(names))

    print(f"Generated {len(names)} noisy files @ {noise_level * 100:.1f}% noise")
    return names


def save_as_xyz(points, filename):
    np.savetxt(filename, points, fmt="%.6f")


def eval(opt, clean_data_dir, mesh_data_dir, npy_save_dir, xyz_save_dir, noise_level):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDNetDenoiser(
        d_model=256,
        num_heads=8,
        num_layers=6,
        k1=20,
        k2=10,
        residual_scale=0.1,
    ).to(device)

    checkpoint = torch.load(opt.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Model loaded from {opt.model_path} ({device})")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(xyz_save_dir, exist_ok=True)

    cd_scores = []
    p2m_scores = []

    for shape_name in tqdm(opt.shape_names, desc="Evaluating shapes"):
        clean_name = shape_name.replace("_noisy", "")
        gt_path = os.path.join(clean_data_dir, f"{clean_name}.npy")
        gt_mesh_path = os.path.join(mesh_data_dir, f"{clean_name}.off")
        noisy_path = os.path.join(opt.testset, f"{shape_name}.npy")

        gt_pts = np.load(gt_path)
        noisy_pts = np.load(noisy_path)
        current_pts = noisy_pts.copy()

        for iteration in range(opt.eval_iter_nums):
            tmp_shape_name = f"{shape_name}_iter_{iteration}"
            tmp_path = os.path.join(npy_save_dir, f"{tmp_shape_name}.npy")
            np.save(tmp_path, current_pts)

            dataset = PointcloudPatchDataset(
                root=npy_save_dir,
                shape_name=tmp_shape_name,
                patch_radius=0.05,
                train_state="evaluation",
                points_per_patch=opt.points_per_patch,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=opt.workers,
                collate_fn=evaluation_collate_fn,
            )

            pred_pts = np.zeros_like(current_pts)
            valid_mask = np.zeros(len(current_pts), dtype=bool)

            for batch in loader:
                if batch is None:
                    continue

                noise_patches = batch["points"].to(device)
                with torch.no_grad():
                    preds = model(noise_patches)

                for i in range(len(preds)):
                    idx = batch["indices"][i].cpu().numpy()
                    inv = batch["noise_inv"][i].to(device)
                    center = batch["noise_disp"][i].to(device)
                    radius = dataset.patch_radius_absolute[0]
                    pred = preds[i] * radius
                    pred = torch.mm(pred, inv.t()) + center.unsqueeze(0)
                    pred_pts[idx] = pred.cpu().numpy()
                    valid_mask[idx] = True

            current_pts = np.where(valid_mask[:, None], pred_pts, current_pts)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        label = f"{int(noise_level * 100)}p"
        np.save(os.path.join(npy_save_dir, f"{clean_name}_denoised_{label}.npy"), current_pts)
        save_as_xyz(current_pts, os.path.join(xyz_save_dir, f"{clean_name}_denoised_{label}.xyz"))

        cd = chamfer_distance(current_pts, gt_pts)
        p2m = calculate_p2m(current_pts, gt_mesh_path)
        print(f"{clean_name} - CD: {cd:.6f}, P2M: {p2m:.6f}")

        cd_scores.append(cd)
        p2m_scores.append(p2m)

    print("\n---------------------------------------------------")
    print(f"Average Chamfer Distance: {np.mean(cd_scores):.6f}")
    print(f"Average Point-to-Mesh Distance: {np.mean(p2m_scores):.6f}")
    print("---------------------------------------------------")


def main():
    opt = parse_arguments()

    num_points = 10000
    noise_level = 0.02
    base_dir = "./results2"
    clean_dir = "./Dataset/Test"
    mesh_dir = "./Dataset/off_files_test"

    os.makedirs(base_dir, exist_ok=True)

    resampled_dir = prepare_resampled_dataset(clean_dir, os.path.join(base_dir, "clean_resampled"), num_points)
    noisy_dir = os.path.join(base_dir, f"noisy_{int(noise_level * 100)}p")
    list_path = os.path.join(base_dir, "test_list.txt")

    opt.shape_names = create_noisy_data_and_list(resampled_dir, noisy_dir, list_path, noise_level)
    opt.testset = noisy_dir
    opt.model_path = "./Summar2/Models/Train/best_model.pth"
    opt.points_per_patch = 500
    opt.batchSize = 4
    opt.workers = 2
    opt.eval_iter_nums = 2
    opt.noise_level = noise_level

    save_dir = os.path.join(base_dir, f"denoised_{int(noise_level * 100)}p")
    eval(opt, resampled_dir, mesh_dir, save_dir, save_dir, noise_level)


if __name__ == "__main__":
    main()
