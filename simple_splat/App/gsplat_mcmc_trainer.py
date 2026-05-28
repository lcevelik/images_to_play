"""
gsplat MCMC Gaussian Splat Trainer

Alternative to Brush for training Gaussian Splats from COLMAP reconstructions.
Uses MCMC densification strategy from gsplat for better quality.

Usage:
    from gsplat_mcmc_trainer import train_mcmc
    train_mcmc(parent_dir, total_steps=10000, progress_callback=callback)
"""

import os
import numpy as np
from pathlib import Path


def load_colmap_dataset(parent_dir):
    """Load COLMAP reconstruction and undistorted images.
    Images and c2w matrices stay on CPU to avoid GPU OOM.
    Only moved to GPU on-demand in the training loop.

    Args:
        parent_dir: Path containing sparse/0/ and images/ directories

    Returns:
        dict with cameras, c2w_mats, images, sparse_xyz, sparse_rgb, image_names
    """
    import pycolmap
    import torch
    from PIL import Image

    sparse_dir = os.path.join(parent_dir, "sparse", "0")
    images_dir = os.path.join(parent_dir, "images")

    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"Sparse reconstruction not found at {sparse_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Undistorted images not found at {images_dir}")

    recon = pycolmap.Reconstruction(sparse_dir)

    available_images = {f.lower(): f for f in os.listdir(images_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    cameras = []
    c2w_mats = []
    image_tensors = []
    image_names = []

    for image_id, image in recon.images.items():
        name_lower = image.name.lower()
        if name_lower not in available_images:
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = name_lower.rsplit('.', 1)[0] + ext
                if candidate in available_images:
                    name_lower = candidate
                    break
            else:
                continue

        img_path = os.path.join(images_dir, available_images[name_lower])
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        # Keep on CPU — moved to GPU on-demand in training loop
        image_tensors.append(torch.from_numpy(img_np))

        cam = recon.cameras[image.camera_id]
        cameras.append({
            'width': cam.width,
            'height': cam.height,
            'fx': float(cam.focal_length_x),
            'fy': float(cam.focal_length_y),
            'cx': float(cam.principal_point_x),
            'cy': float(cam.principal_point_y),
        })

        # pycolmap's cam_from_world is world-to-camera (w2c), NOT camera-to-world
        # gsplat viewmats expect w2c as 4x4, so pad the 3x4 matrix
        w2c_3x4 = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :] = w2c_3x4
        # Keep on CPU — moved to GPU on-demand
        c2w_mats.append(torch.from_numpy(w2c_4x4))
        image_names.append(available_images[name_lower])

    sparse_xyz = []
    sparse_rgb = []
    for point3D_id, point in recon.points3D.items():
        sparse_xyz.append(point.xyz)
        sparse_rgb.append(point.color)

    sparse_xyz = np.array(sparse_xyz, dtype=np.float32) if sparse_xyz else np.zeros((0, 3), dtype=np.float32)
    sparse_rgb = np.array(sparse_rgb, dtype=np.float32) / 255.0 if sparse_rgb else np.zeros((0, 3), dtype=np.float32)

    return {
        'cameras': cameras,
        'c2w_mats': c2w_mats,
        'images': image_tensors,
        'sparse_xyz': sparse_xyz,
        'sparse_rgb': sparse_rgb,
        'image_names': image_names,
    }


def init_gaussians_from_sparse(xyz, rgb, sh_degree=3):
    """Initialize Gaussian parameters from sparse point cloud.
    Uses random subsample for scale estimation (avoids slow cKDTree on millions of points)."""
    import torch
    N = xyz.shape[0]
    means = torch.from_numpy(xyz).float()

    # Estimate scale from random subsample nearest-neighbor distances
    # (avoids building a full cKDTree on millions of points — 5-30s savings)
    sample_size = min(10000, N)
    indices = np.random.choice(N, sample_size, replace=False)
    sample_xyz = xyz[indices]

    # Compute pairwise distances for the subsample
    from scipy.spatial import cKDTree
    tree = cKDTree(sample_xyz)
    dists, _ = tree.query(sample_xyz, k=min(6, sample_size))
    avg_sample_dist = dists[:, 1:].mean(axis=1)
    global_avg_scale = float(avg_sample_dist.mean()) * 0.5

    # Use uniform scale from subsample estimate (fast, good enough for init)
    scales = torch.full((N, 3), global_avg_scale)

    # Identity quaternions (w, x, y, z)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    # Initial opacity (sigmoid inverse of 0.5)
    opacities = torch.zeros(N)

    # SH coefficients from RGB
    C0 = 0.28209479177387814
    rgb_tensor = torch.from_numpy(rgb).float()
    sh0 = (rgb_tensor - 0.5) / C0
    sh0 = sh0.unsqueeze(1)  # (N, 1, 3)

    num_sh_coeffs = (sh_degree + 1) ** 2
    shN = torch.zeros(N, num_sh_coeffs - 1, 3)

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'sh0': sh0,
        'shN': shN,
    }


def train_mcmc(parent_dir, total_steps=10000, cap_max=1_500_000,
               lr_base=1e-3, sh_degree=3, progress_callback=None,
               output_ply_path=None):
    """Train Gaussian Splat using MCMC densification strategy.

    Args:
        parent_dir: COLMAP reconstruction directory (with sparse/0/ and images/)
        total_steps: Total training iterations
        cap_max: Maximum number of Gaussians (MCMC cap)
        lr_base: Base learning rate
        sh_degree: Spherical harmonics degree (0=color only, 1-3=more detail)
        progress_callback: fn(message, level) for logging
        output_ply_path: Where to save the output PLY

    Returns:
        Path to output PLY or None on failure
    """
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization, export_splats
    from gsplat.strategy import MCMCStrategy
    from gsplat.optimizers import SelectiveAdam

    def log(msg, level="INFO"):
        if progress_callback:
            progress_callback(msg, level)
        else:
            print(f"[{level}] {msg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Load dataset
    log("Loading COLMAP dataset...")
    dataset = load_colmap_dataset(parent_dir)

    num_images = len(dataset['images'])
    num_sparse = dataset['sparse_xyz'].shape[0]
    log(f"Loaded {num_images} images and {num_sparse} sparse points")

    if num_images == 0:
        log("No images found in undistorted images folder", "ERROR")
        return None

    if num_sparse == 0:
        log("No sparse points found in reconstruction", "ERROR")
        return None

    # Initialize Gaussians
    log("Initializing Gaussians from sparse points...")
    splats = init_gaussians_from_sparse(dataset['sparse_xyz'], dataset['sparse_rgb'], sh_degree)

    # Move to device
    for k, v in splats.items():
        splats[k] = v.to(device).requires_grad_(True)

    # Build optimizers (one per parameter group for SelectiveAdam)
    from torch.nn import Parameter
    from torch import nn

    param_dict = nn.ParameterDict({
        'means': splats['means'],
        'scales': splats['scales'],
        'quats': splats['quats'],
        'opacities': splats['opacities'],
        'sh0': splats['sh0'],
        'shN': splats['shN'],
    })

    optimizers = {
        'means': SelectiveAdam([{'params': param_dict['means'], 'lr': lr_base}], eps=1e-15, betas=(0.9, 0.999)),
        'scales': SelectiveAdam([{'params': param_dict['scales'], 'lr': lr_base * 0.1}], eps=1e-15, betas=(0.9, 0.999)),
        'quats': SelectiveAdam([{'params': param_dict['quats'], 'lr': lr_base * 0.01}], eps=1e-15, betas=(0.9, 0.999)),
        'opacities': SelectiveAdam([{'params': param_dict['opacities'], 'lr': lr_base * 0.05}], eps=1e-15, betas=(0.9, 0.999)),
        'sh0': SelectiveAdam([{'params': param_dict['sh0'], 'lr': lr_base * 0.05}], eps=1e-15, betas=(0.9, 0.999)),
        'shN': SelectiveAdam([{'params': param_dict['shN'], 'lr': lr_base * 0.0125}], eps=1e-15, betas=(0.9, 0.999)),
    }

    # Initialize MCMC strategy
    strategy = MCMCStrategy(
        cap_max=cap_max,
        refine_start_iter=500,
        refine_stop_iter=total_steps - 500,
        refine_every=100,
        verbose=False,
    )
    strategy_state = strategy.initialize_state()

    # NOTE: c2w_mats and img_tensors stay on CPU (loaded in load_colmap_dataset)
    # Only the current batch image is moved to GPU each step — prevents OOM
    # on GPUs with <16GB VRAM when processing 200+ images at 4K

    def ssim_loss(pred, target, window_size=11):
        """Simplified SSIM loss."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    # Training loop
    num_gaussians = splats['means'].shape[0]
    log(f"Starting MCMC training: {total_steps} steps, cap_max={cap_max}, initial gaussians={num_gaussians:,}")
    for step in range(total_steps):
        idx = np.random.randint(num_images)
        cam = dataset['cameras'][idx]
        # Move only this image's data to GPU (not all images)
        c2w = dataset['c2w_mats'][idx].to(device)
        gt_image = dataset['images'][idx].to(device)
        # Use actual image dimensions (undistortion may change size from camera params)
        H, W = gt_image.shape[0], gt_image.shape[1]

        K = torch.tensor([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ], device=device, dtype=torch.float32).unsqueeze(0)

        viewmat = c2w.unsqueeze(0)

        # Build colors tensor (SH coefficients)
        colors_sh = torch.cat([splats['sh0'], splats['shN']], dim=1)

        rendered, alpha, info = rasterization(
            means=splats['means'],
            quats=splats['quats'],
            scales=torch.exp(splats['scales']),
            opacities=torch.sigmoid(splats['opacities']),
            colors=colors_sh,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            sh_degree=sh_degree,
            render_mode="RGB",
            absgrad=True,
        )

        # Compute loss
        pred_rgb = rendered[0]  # (H, W, 3)
        loss_l1 = F.l1_loss(pred_rgb, gt_image)
        loss_ssim = ssim_loss(pred_rgb, gt_image)
        loss = 0.8 * loss_l1 + 0.2 * loss_ssim

        # Zero all optimizers
        for opt in optimizers.values():
            opt.zero_grad()

        loss.backward()

        # MCMC post-backward: relocate dead GSs + add new GSs + inject noise
        strategy.step_post_backward(param_dict, optimizers, strategy_state, step, info, lr_base)

        # Get visibility mask from rasterization info for SelectiveAdam
        visibility = info.get("gaussian_ids", None)
        if visibility is not None:
            # Build a visibility mask: which gaussians are visible
            vis_mask = torch.zeros(splats['means'].shape[0], device=device, dtype=torch.bool)
            vis_mask[visibility.unique()] = True
        else:
            vis_mask = torch.ones(splats['means'].shape[0], device=device, dtype=torch.bool)

        # Step all optimizers with visibility mask
        for opt in optimizers.values():
            opt.step(visibility=vis_mask)

        # Update splats references
        for key in splats:
            splats[key] = param_dict[key]

        # Logging
        if step % 100 == 0 or step == total_steps - 1:
            num_gaussians = splats['means'].shape[0]
            log(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | "
                f"Gaussians: {num_gaussians:,} | L1: {loss_l1.item():.4f}")
            if progress_callback:
                progress_callback(f"Step {step}/{total_steps}", "INFO")

    # Export
    num_gaussians = splats['means'].shape[0]
    log(f"Training complete! Final Gaussians: {num_gaussians:,}")

    if output_ply_path is None:
        output_ply_path = os.path.join(parent_dir, "gaussian_splat.ply")

    log(f"Exporting PLY to {output_ply_path}...")
    export_splats(
        means=splats['means'].detach(),
        scales=torch.exp(splats['scales']).detach(),
        quats=splats['quats'].detach(),
        opacities=torch.sigmoid(splats['opacities']).detach(),
        sh0=splats['sh0'].detach(),
        shN=splats['shN'].detach(),
        format="ply",
        save_to=output_ply_path,
    )

    log(f"PLY exported: {output_ply_path} ({os.path.getsize(output_ply_path) / 1024 / 1024:.1f} MB)")
    return output_ply_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Gaussian Splat using MCMC strategy")
    parser.add_argument("--input", "-i", required=True, help="COLMAP reconstruction directory")
    parser.add_argument("--output", "-o", help="Output PLY path")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--cap-max", type=int, default=1_500_000, help="Max Gaussians")
    args = parser.parse_args()

    result = train_mcmc(
        args.input,
        total_steps=args.steps,
        cap_max=args.cap_max,
        output_ply_path=args.output,
    )
    if result:
        print(f"Success: {result}")
    else:
        print("Training failed")
        exit(1)
