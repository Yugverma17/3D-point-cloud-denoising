"""
Training and inference loops.

Kept in the package rather than in the scripts so the tests can import and
exercise the same code path the CLI uses.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PatchDataset, centres_to_world, collate, iter_eval_patches
from .losses import DenoisingLoss
from .metrics import evaluate_shape, format_table, summarize
from .model import Denoiser


def pick_device(preference="auto"):
    if preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def denoise_cloud(model, shape, points_per_patch=256, batch_size=64, iters=1, device=None,
                  progress=False):
    """
    Denoise every point of a cloud.

    Each point is the centre of exactly one patch and takes exactly one
    prediction, so no point is written twice. `iters` re-runs the whole pass,
    feeding the previous output back in - the model only ever predicts a small
    displacement, so a second pass can clean up what one pass could not reach.
    """
    device = device or pick_device()
    model = model.to(device).eval()

    current = shape.noisy.copy()
    for _ in range(iters):
        working = type(shape)(shape.clean, noisy=current)
        radius = 0.05 * working.diagonal  # only a fallback; patches self-scale
        updated = current.copy()

        batches = iter_eval_patches(working, radius, points_per_patch, batch_size=batch_size)
        if progress:
            batches = tqdm(batches, desc="denoising", leave=False)

        for batch in batches:
            if batch is None:
                continue
            local = model.predict_centre(batch["points"].to(device)).cpu().numpy()
            updated[batch["index"].numpy()] = centres_to_world(local, batch)
        current = updated

    return current


def evaluate(model, shapes, names, mesh_paths=None, points_per_patch=256, batch_size=64,
             iters=1, device=None, include_baseline=True):
    """
    Denoise each shape and score it. Returns (rows, baseline_rows).

    The noisy input is scored too, because "did this actually help?" is the
    only question that matters and a table without the baseline cannot answer
    it.
    """
    mesh_paths = mesh_paths or [None] * len(shapes)
    rows, baseline = [], []

    for shape, name, mesh in zip(shapes, tqdm(names, desc="shapes"), mesh_paths):
        denoised = denoise_cloud(
            model, shape, points_per_patch=points_per_patch,
            batch_size=batch_size, iters=iters, device=device,
        )
        rows.append(evaluate_shape(denoised, shape.clean, mesh))
        if include_baseline:
            baseline.append(evaluate_shape(shape.noisy, shape.clean, mesh))

    return rows, baseline


def train(
    shapes,
    out_dir,
    epochs=50,
    batch_size=32,
    points_per_patch=256,
    patches_per_shape=1000,
    lr=1e-3,
    weight_decay=1e-4,
    repulsion_weight=0.05,
    noise_range=(0.005, 0.03),
    model_kwargs=None,
    device=None,
    resume=None,
    seed=0,
    num_workers=0,
):
    """Train a denoiser and write checkpoints + a loss history to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Denoiser(**(model_kwargs or {})).to(device)
    loss_fn = DenoisingLoss(repulsion_weight=repulsion_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch, best = 0, float("inf")
    history = []
    if resume and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best = ckpt.get("best", best)
        history = ckpt.get("history", [])
        print(f"resumed from {resume} at epoch {start_epoch}")

    dataset = PatchDataset(
        shapes,
        points_per_patch=points_per_patch,
        patches_per_shape=patches_per_shape,
        seed=seed,
        noise_range=noise_range,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate,
        num_workers=num_workers, drop_last=True,
    )

    noise_desc = (f"noise {noise_range[0]:.1%}-{noise_range[1]:.1%} sampled per patch"
                  if noise_range else "fixed noise from each Shape")
    print(f"{len(shapes)} shapes, {len(dataset)} patches/epoch, {noise_desc}, device={device}")

    for epoch in range(start_epoch, epochs):
        model.train()
        totals, seen, t0 = {}, 0, time.time()
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")

        for batch in bar:
            if batch is None:
                continue
            points = batch["points"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            loss, parts = loss_fn(model(points), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            seen += 1
            bar.set_postfix(loss=f"{parts['total']:.5f}")

        if seen == 0:
            raise RuntimeError("no valid batches - check patch settings and input data")

        record = {k: v / seen for k, v in totals.items()}
        record.update(epoch=epoch + 1, lr=scheduler.get_last_lr()[0],
                      seconds=round(time.time() - t0, 1))
        history.append(record)
        scheduler.step()
        print(f"  loss {record['total']:.6f}  (chamfer {record['chamfer']:.6f})  "
              f"lr {record['lr']:.2e}  {record['seconds']}s")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best": best,
            "model_kwargs": model_kwargs or {},
        }
        torch.save(state, out_dir / "last.pt")
        if record["total"] < best:
            best = record["total"]
            state["best"] = best
            torch.save(state, out_dir / "best.pt")
            print(f"  new best: {best:.6f}")

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return model, history


def load_model(checkpoint_path, device=None):
    """Rebuild a model from a checkpoint, using the kwargs it was saved with."""
    device = device or pick_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Denoiser(**ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def write_report(path, names, rows, baseline=None):
    """Results table with the noisy baseline alongside, so the delta is visible."""
    path = Path(path)
    text = format_table(names, rows, title="Denoised")
    if baseline:
        text += "\n\n" + format_table(names, baseline, title="Noisy input (baseline)")
        after, before = summarize(rows), summarize(baseline)
        text += "\n\nImprovement\n" + "-" * 40
        for key in after:
            delta = 100.0 * (before[key] - after[key]) / before[key]
            text += f"\n{key.upper():<8} {before[key]:10.4f} -> {after[key]:10.4f}   {delta:+6.1f}%"
    path.write_text(text + "\n")
    return text
