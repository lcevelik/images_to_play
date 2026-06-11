"""
gsplat MCMC Gaussian Splat Trainer

Trains Gaussian Splats from COLMAP reconstructions using gsplat's MCMCStrategy.
MCMC uses stochastic relocation instead of clone/split — avoids the 15K
densification cutoff of standard 3DGS and gives an explicit Gaussian cap.

Bug-fix history applied here:
  - d91d27b: viewmat must be c2w (inv of pycolmap's w2c), not w2c directly
  - d7185d4: init opacity 10% (not 50%), random quats, per-point knn scale
"""

import os
import numpy as np
from pathlib import Path


def load_colmap_dataset(parent_dir):
    """Load COLMAP reconstruction and undistorted images.

    Images stay on CPU; only the current training image is moved to GPU
    per step to avoid OOM on GPUs with <16 GB VRAM and 200+ images.
    """
    import pycolmap
    import torch
    from PIL import Image

    sparse_dir = os.path.join(parent_dir, "sparse", "0")
    images_dir = os.path.join(parent_dir, "images")

    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"Sparse reconstruction not found: {sparse_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Undistorted images not found: {images_dir}")

    recon = pycolmap.Reconstruction(sparse_dir)
    available_images = {f.lower(): f for f in os.listdir(images_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    cameras, c2w_mats, image_tensors, image_names = [], [], [], []

    for _, image in recon.images.items():
        name_lower = image.name.lower()
        if name_lower not in available_images:
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = name_lower.rsplit('.', 1)[0] + ext
                if candidate in available_images:
                    name_lower = candidate
                    break
            else:
                continue

        img = Image.open(os.path.join(images_dir, available_images[name_lower])).convert('RGB')
        image_tensors.append(torch.from_numpy(np.array(img, dtype=np.float32) / 255.0))

        cam = recon.cameras[image.camera_id]
        cameras.append({
            'width': cam.width, 'height': cam.height,
            'fx': float(cam.focal_length_x), 'fy': float(cam.focal_length_y),
            'cx': float(cam.principal_point_x), 'cy': float(cam.principal_point_y),
        })

        # pycolmap.cam_from_world returns w2c; gsplat rasterization() expects c2w
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        c2w_mats.append(torch.from_numpy(np.linalg.inv(w2c)))
        image_names.append(available_images[name_lower])

    sparse_xyz, sparse_rgb = [], []
    for _, pt in recon.points3D.items():
        sparse_xyz.append(pt.xyz)
        sparse_rgb.append(pt.color)

    sparse_xyz = np.array(sparse_xyz, dtype=np.float32) if sparse_xyz else np.zeros((0, 3), dtype=np.float32)
    sparse_rgb = np.array(sparse_rgb, dtype=np.float32) / 255.0 if sparse_rgb else np.zeros((0, 3), dtype=np.float32)

    return {
        'cameras': cameras, 'c2w_mats': c2w_mats,
        'images': image_tensors, 'image_names': image_names,
        'sparse_xyz': sparse_xyz, 'sparse_rgb': sparse_rgb,
    }


def init_gaussians_from_sparse(xyz, rgb, sh_degree=3):
    """Initialise Gaussian parameters from sparse COLMAP points."""
    import torch
    from scipy.spatial import cKDTree

    N = xyz.shape[0]
    means = torch.from_numpy(xyz).float()

    # Per-point scale from nearest-neighbour distances (matches gsplat official init)
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=min(4, N))
    dist_avg = np.sqrt((dists[:, 1:] ** 2).mean(axis=1))
    scales = torch.from_numpy(np.log(dist_avg).astype(np.float32)).unsqueeze(-1).repeat(1, 3)

    # Random unit quaternions (better initial coverage than identity)
    quats = torch.randn(N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # Init at 10% opacity — sigmoid_inverse(0.1) ≈ -2.197
    opacities = torch.full((N,), -2.197)

    # DC SH coefficients from RGB: sh0 = (rgb - 0.5) / C0
    C0 = 0.28209479177387814
    sh0 = ((torch.from_numpy(rgb).float() - 0.5) / C0).unsqueeze(1)  # (N, 1, 3)
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3)

    return {'means': means, 'scales': scales, 'quats': quats,
            'opacities': opacities, 'sh0': sh0, 'shN': shN}


def train_mcmc(parent_dir, total_steps=15000, cap_max=1_000_000,
               sh_degree=3, use_lpips=False, progress_callback=None, output_ply_path=None):
    """Train Gaussian Splat using gsplat MCMCStrategy.

    Args:
        parent_dir: Job directory with sparse/0/ and images/
        total_steps: Training iterations
        cap_max: Hard Gaussian count cap (MCMC never exceeds this)
        sh_degree: Spherical harmonics degree 0-3
        progress_callback: fn(message, level) for log streaming
        output_ply_path: Where to save the PLY (default: parent_dir/gaussian_splat.ply)

    Returns:
        Path to output PLY or None on failure
    """
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    from gsplat.optimizers import SelectiveAdam

    def log(msg, level="INFO"):
        if progress_callback:
            progress_callback(msg, level)
        print(f"[{level}] {msg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"MCMC trainer using device: {device}")

    log("Loading COLMAP dataset...")
    dataset = load_colmap_dataset(parent_dir)
    num_images = len(dataset['images'])
    num_sparse = dataset['sparse_xyz'].shape[0]
    log(f"Loaded {num_images} images, {num_sparse:,} sparse points")

    if num_images == 0 or num_sparse == 0:
        log("No images or sparse points — cannot train", "ERROR")
        return None

    log("Initialising Gaussians...")
    raw = init_gaussians_from_sparse(dataset['sparse_xyz'], dataset['sparse_rgb'], sh_degree)
    splats = {k: v.to(device).requires_grad_(True) for k, v in raw.items()}

    lr_map = {'means': 1e-3, 'scales': 1e-4, 'quats': 1e-2,
              'opacities': 5e-2, 'sh0': 5e-3, 'shN': 1.25e-4}
    optimizers = {
        k: SelectiveAdam([{'params': splats[k], 'lr': lr_map[k]}], eps=1e-15, betas=(0.9, 0.999))
        for k in splats
    }

    strategy = MCMCStrategy(
        cap_max=cap_max,
        refine_start_iter=500,
        refine_stop_iter=total_steps - 500,
        refine_every=100,
        verbose=False,
    )
    strategy_state = strategy.initialize_state()

    def ssim_loss(pred, target, w=11):
        C1, C2 = 0.01**2, 0.03**2
        p = pred.permute(2, 0, 1).unsqueeze(0)
        t = target.permute(2, 0, 1).unsqueeze(0)
        mu1 = F.avg_pool2d(p, w, stride=1, padding=w//2)
        mu2 = F.avg_pool2d(t, w, stride=1, padding=w//2)
        s1 = F.avg_pool2d(p*p, w, 1, w//2) - mu1**2
        s2 = F.avg_pool2d(t*t, w, 1, w//2) - mu2**2
        s12 = F.avg_pool2d(p*t, w, 1, w//2) - mu1*mu2
        return 1 - ((2*mu1*mu2+C1)*(2*s12+C2) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()

    lpips_fn = None
    if use_lpips:
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
            lpips_fn.eval()
            log(f"LPIPS (VGG) enabled — PyTorch CUDA, no buffer limit")
        except ImportError:
            log("lpips package not installed — run: pip install lpips. Continuing without LPIPS.", "WARNING")

    log(f"Starting MCMC training: {total_steps} steps, cap={cap_max:,}, lpips={'on' if lpips_fn else 'off'}")

    from torch.nn import ParameterDict
    param_dict = ParameterDict({k: torch.nn.Parameter(splats[k]) for k in splats})

    for step in range(total_steps):
        idx = np.random.randint(num_images)
        cam = dataset['cameras'][idx]
        c2w = dataset['c2w_mats'][idx].to(device)
        gt = dataset['images'][idx].to(device)
        H, W = gt.shape[0], gt.shape[1]

        K = torch.tensor([[cam['fx'], 0, cam['cx']],
                          [0, cam['fy'], cam['cy']],
                          [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

        colors_sh = torch.cat([param_dict['sh0'], param_dict['shN']], dim=1)
        rendered, _, info = rasterization(
            means=param_dict['means'],
            quats=param_dict['quats'],
            scales=torch.exp(param_dict['scales']),
            opacities=torch.sigmoid(param_dict['opacities']),
            colors=colors_sh,
            viewmats=c2w.unsqueeze(0),
            Ks=K, width=W, height=H,
            near_plane=0.01, far_plane=1000.0,
            sh_degree=sh_degree, render_mode="RGB", absgrad=True,
        )

        pred = rendered[0]
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * ssim_loss(pred, gt)

        if lpips_fn is not None:
            # Normalize to [-1, 1] and convert to [N, C, H, W] for LPIPS
            p_lp = (pred.clamp(0, 1) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            g_lp = (gt.clamp(0, 1) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            loss = loss + 0.05 * lpips_fn(p_lp, g_lp).mean()

        for opt in optimizers.values():
            opt.zero_grad()
        loss.backward()

        strategy.step_post_backward(param_dict, optimizers, strategy_state, step, info, 1e-3)

        vis_ids = info.get("gaussian_ids")
        if vis_ids is not None:
            vis_mask = torch.zeros(param_dict['means'].shape[0], device=device, dtype=torch.bool)
            vis_mask[vis_ids.unique()] = True
        else:
            vis_mask = torch.ones(param_dict['means'].shape[0], device=device, dtype=torch.bool)

        for opt in optimizers.values():
            opt.step(visibility=vis_mask)

        if step % 500 == 0 or step == total_steps - 1:
            n = param_dict['means'].shape[0]
            log(f"[MCMC] Step {step}/{total_steps} | loss={loss.item():.4f} | gaussians={n:,}")

    if output_ply_path is None:
        output_ply_path = os.path.join(parent_dir, "gaussian_splat.ply")

    log(f"Exporting {param_dict['means'].shape[0]:,} Gaussians to PLY...")
    try:
        from gsplat import export_splats
        export_splats(
            means=param_dict['means'].detach(),
            scales=torch.exp(param_dict['scales']).detach(),
            quats=param_dict['quats'].detach(),
            opacities=torch.sigmoid(param_dict['opacities']).detach(),
            sh0=param_dict['sh0'].detach(),
            shN=param_dict['shN'].detach(),
            format="ply",
            save_to=output_ply_path,
        )
    except ImportError:
        # Fallback: manual PLY export if export_splats not available
        _export_3dgs_ply(param_dict, output_ply_path, sh_degree)

    size_mb = os.path.getsize(output_ply_path) / 1024 / 1024
    log(f"Exported: {output_ply_path} ({size_mb:.1f} MB)")
    return output_ply_path


def _export_3dgs_ply(param_dict, output_path, sh_degree):
    """Fallback PLY writer when gsplat.export_splats is unavailable."""
    import torch
    import numpy as np

    means = param_dict['means'].detach().cpu().numpy()
    scales = torch.exp(param_dict['scales']).detach().cpu().numpy()
    quats = param_dict['quats'].detach().cpu().numpy()
    opacities = torch.sigmoid(param_dict['opacities']).detach().cpu().numpy()
    sh0 = param_dict['sh0'].detach().cpu().numpy().reshape(len(means), -1)
    shN = param_dict['shN'].detach().cpu().numpy().reshape(len(means), -1)

    N = len(means)
    num_sh_rest = shN.shape[1]

    props = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] +
        [f'f_dc_{i}' for i in range(3)] +
        [f'f_rest_{i}' for i in range(num_sh_rest)] +
        ['opacity'] +
        [f'scale_{i}' for i in range(3)] +
        [f'rot_{i}' for i in range(4)]
    )

    dtype = np.dtype([(p, np.float32) for p in props])
    data = np.zeros(N, dtype=dtype)
    data['x'], data['y'], data['z'] = means[:, 0], means[:, 1], means[:, 2]
    from scipy.special import logit
    data['opacity'] = logit(np.clip(opacities, 1e-6, 1 - 1e-6))
    for i in range(3):
        data[f'scale_{i}'] = np.log(np.clip(scales[:, i], 1e-10, None))
    data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3'] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    for i in range(3):
        data[f'f_dc_{i}'] = sh0[:, i]
    for i in range(num_sh_rest):
        data[f'f_rest_{i}'] = shN[:, i]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n" +
        ''.join(f"property float {p}\n" for p in props) +
        "end_header\n"
    )
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--cap-max", type=int, default=1_000_000)
    args = parser.parse_args()
    result = train_mcmc(args.input, total_steps=args.steps, cap_max=args.cap_max, output_ply_path=args.output)
    print("Success:" if result else "Failed:", result)
