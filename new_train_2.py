# coding=utf-8
from __future__ import print_function
from tensorboardX import SummaryWriter
from new_Pointfilter_Network_Architecture_2 import TDNetDenoiser, RobustChamferLoss, RepulsionLoss
from new_Pointfilter_DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, collate_fn
from new_Pointfilter_Utils import parse_arguments
import torch
import torch.nn as nn
import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def save_loss_graph(loss_history, filename="loss_vs_epoch.png"):
    """Save training loss and learning rate curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curve
    ax1.plot(loss_history['train'], label='Training Loss', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Chamfer Distance')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # LR curve
    ax2.plot(loss_history['lr'], label='Learning Rate', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')  # log scale for LR

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Loss graph saved to {filename}")


def train(opt):
    # Setup directories
    os.makedirs(opt.summary_dir, exist_ok=True)
    os.makedirs(opt.network_model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)

    # Initialize model and loss
    model = TDNetDenoiser().to(device)
    criterion_chamfer = RobustChamferLoss().to(device)
    criterion_repulsion = RepulsionLoss(k=4, h=0.005).to(device)
    alpha_repulsion = 0.05  # weight for repulsion loss

    # ✅ CHANGE 1: Lower LR (0.001 instead of 0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # ✅ CHANGE 2: Add CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.nepoch,   # total number of epochs
        eta_min=1e-6        # minimum LR at the end
    )
    print(f"Using CosineAnnealingLR on {device}: LR starts at 0.001, ends at 1e-6 over {opt.nepoch} epochs")

    # Tensorboard
    writer = SummaryWriter(opt.summary_dir)

    # Load checkpoint if resuming
    if hasattr(opt, 'resume') and opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> Loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location=device)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # ✅ CHANGE 3: Also restore scheduler state when resuming
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> Restored scheduler state")
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{opt.resume}'")
    else:
        print("=> No checkpoint to resume - training from scratch")

    # Dataset and dataloader
    train_dataset = PointcloudPatchDataset(
        root=opt.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.05,
        points_per_patch=500,
        noise_type=opt.noise_type,
        noise_level=opt.noise_level,
        corruption_rate=opt.corruption_rate,
        use_augmentation=opt.use_augmentation,
        scale_range=opt.scale_range
    )

    train_sampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patch_per_shape
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.batchSize,
        collate_fn=collate_fn,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True
    )

    # Training loop
    best_loss = float('inf')
    loss_history = {'train': [], 'lr': []}  # ✅ also track LR history

    for epoch in range(opt.start_epoch, opt.nepoch):
        model.train()
        epoch_loss = 0
        valid_batches = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt.nepoch}')

        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                continue

            # Move data to GPU
            noisy = batch_data['noisy_points'].to(device, non_blocking=True)
            gt = batch_data['gt_points'].to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            denoised = model(noisy)

            # Compute loss
            loss_chamfer = criterion_chamfer(denoised, gt)
            loss_rep = criterion_repulsion(denoised)
            loss = loss_chamfer + alpha_repulsion * loss_rep

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update stats
            epoch_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'Loss': loss.item()})

            # Tensorboard logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), global_step)

        # Epoch statistics
        if valid_batches == 0:
            raise RuntimeError("No valid training batches were produced. Check patch extraction settings and dataset contents.")

        avg_loss = epoch_loss / valid_batches
        loss_history['train'].append(avg_loss)

        # ✅ CHANGE 4: Get current LR before stepping
        current_lr = scheduler.get_last_lr()[0]
        loss_history['lr'].append(current_lr)

        print(f'Epoch {epoch+1}/{opt.nepoch} - Avg Loss: {avg_loss:.6f} - LR: {current_lr:.8f}')

        # ✅ CHANGE 5: Step the scheduler at end of each epoch
        scheduler.step()

        # Log LR to tensorboard
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('avg_loss', avg_loss, epoch)

        # ✅ CHANGE 6: Save scheduler state in checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),  # ← saves scheduler state
            'loss': avg_loss
        }

        # Save regular checkpoint
        torch.save(state, os.path.join(opt.network_model_dir, 'checkpoint.pth'))

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, os.path.join(opt.network_model_dir, 'best_model.pth'))
            print(f"  ✅ New best model saved! Loss: {avg_loss:.6f}")

        # Save periodic snapshot
        if (epoch + 1) % opt.model_interval == 0:
            torch.save(state, os.path.join(opt.network_model_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save loss + LR curve
        save_loss_graph(loss_history, os.path.join(opt.summary_dir, "loss_vs_epoch.png"))

    # Save final model
    torch.save(state, os.path.join(opt.network_model_dir, 'final_model.pth'))
    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    print(f"Best model saved at: {os.path.join(opt.network_model_dir, 'best_model.pth')}")
    writer.close()


def prepare_dataset(opt):
    """Automatically create train.txt and handle both Kaggle and local paths"""
    base_path = opt.trainset if opt.trainset else "./Dataset/Train"
    os.makedirs(base_path, exist_ok=True)

    train_txt_path = os.path.join(base_path, "train.txt")

    if not os.path.exists(train_txt_path) or os.path.getsize(train_txt_path) == 0:
        print(f"Creating/updating train.txt at {train_txt_path}")
        npy_files = [f for f in os.listdir(base_path)
                     if f.endswith('.npy') and not f.endswith('_normal.npy')]

        if not npy_files:
            available_files = "\n".join(os.listdir(base_path))
            raise FileNotFoundError(
                f"No .npy files found in {base_path}\n"
                f"Available files:\n{available_files}"
            )

        with open(train_txt_path, 'w') as f:
            for npy_file in sorted(npy_files):
                f.write(npy_file.replace('.npy', '') + '\n')
        print(f"Created train.txt with {len(npy_files)} shapes")

    # Normal estimation
    shape_names = []
    with open(train_txt_path) as f:
        shape_names = [x.strip() for x in f.readlines() if x.strip()]

    for name in shape_names:
        normal_path = os.path.join(base_path, f"{name}_normal.npy")
        if not os.path.exists(normal_path):
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    np.load(os.path.join(base_path, f"{name}.npy")))
                pcd.estimate_normals()
                np.save(normal_path, np.asarray(pcd.normals))
            except ImportError:
                print("Open3D not available - skipping normal estimation")
            except Exception as e:
                print(f"Error estimating normals for {name}: {str(e)}")


if __name__ == '__main__':
    opt = parse_arguments()

    # Default paths
    opt.trainset = opt.trainset or './Dataset/Train'
    opt.summary_dir = opt.summary_dir or './Summary2/Train/logs'
    opt.network_model_dir = opt.network_model_dir or './Summary2/Train'

    # Prepare dataset
    prepare_dataset(opt)

    # Start training
    train(opt)
