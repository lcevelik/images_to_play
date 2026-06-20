"""
gsplat MCMC Gaussian Splat Trainer

Trains Gaussian Splats from COLMAP reconstructions using gsplat's MCMCStrategy.
MCMC uses stochastic relocation instead of clone/split — avoids the 15K
densification cutoff of standard 3DGS and gives an explicit Gaussian cap.

Hyperparameters follow gsplat's reference examples/simple_trainer.py (mcmc
preset): per-param learning rates, means lr scaled by scene scale with
exponential decay, opacity/scale L1 regularization, and SH degree warmup.

Conventions (verified against gsplat 1.5.3 docs):
  - pycolmap's image.cam_from_world is world-to-camera (w2c)
  - gsplat rasterization() expects viewmats = w2c — pass it directly, do NOT invert
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
    masks_dir = os.path.join(parent_dir, "masks")   # optional sky masks (255=keep, 0=sky)
    has_masks = os.path.isdir(masks_dir)

    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"Sparse reconstruction not found: {sparse_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Undistorted images not found: {images_dir}")

    recon = pycolmap.Reconstruction(sparse_dir)
    available_images = {f.lower(): f for f in os.listdir(images_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    cameras, w2c_mats, image_tensors, image_names, image_masks = [], [], [], [], []

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

        # optional foreground mask (1=keep, 0=sky); matched by basename
        mk = None
        if has_masks:
            mpath = os.path.join(masks_dir, os.path.splitext(available_images[name_lower])[0] + '.png')
            if os.path.exists(mpath):
                m = Image.open(mpath).convert('L')
                mk = torch.from_numpy((np.array(m) > 127).astype(np.float32))
        image_masks.append(mk)

        cam = recon.cameras[image.camera_id]
        cameras.append({
            'width': cam.width, 'height': cam.height,
            'fx': float(cam.focal_length_x), 'fy': float(cam.focal_length_y),
            'cx': float(cam.principal_point_x), 'cy': float(cam.principal_point_y),
        })

        # pycolmap cam_from_world is w2c — exactly what gsplat's viewmats expects
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        w2c_mats.append(torch.from_numpy(w2c))
        image_names.append(available_images[name_lower])

    sparse_xyz, sparse_rgb = [], []
    for _, pt in recon.points3D.items():
        sparse_xyz.append(pt.xyz)
        sparse_rgb.append(pt.color)

    sparse_xyz = np.array(sparse_xyz, dtype=np.float32) if sparse_xyz else np.zeros((0, 3), dtype=np.float32)
    sparse_rgb = np.array(sparse_rgb, dtype=np.float32) / 255.0 if sparse_rgb else np.zeros((0, 3), dtype=np.float32)

    return {
        'cameras': cameras, 'w2c_mats': w2c_mats,
        'images': image_tensors, 'image_names': image_names,
        'masks': image_masks, 'has_masks': has_masks and any(m is not None for m in image_masks),
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
    scales = torch.from_numpy(np.log(np.clip(dist_avg, 1e-8, None)).astype(np.float32)).unsqueeze(-1).repeat(1, 3)

    # Random unit quaternions (better initial coverage than identity)
    quats = torch.randn(N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # MCMC init opacity 0.5 (gsplat mcmc preset) — logit(0.5) = 0
    opacities = torch.zeros(N)

    # DC SH coefficients from RGB: sh0 = (rgb - 0.5) / C0
    C0 = 0.28209479177387814
    sh0 = ((torch.from_numpy(rgb).float() - 0.5) / C0).unsqueeze(1)  # (N, 1, 3)
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3)

    return {'means': means, 'scales': scales, 'quats': quats,
            'opacities': opacities, 'sh0': sh0, 'shN': shN}


def train_mcmc(parent_dir, total_steps=15000, cap_max=None,
               sh_degree=3, use_lpips=False, progress_callback=None, output_ply_path=None,
               strategy_name='mcmc', export_opacity_min=0.03):
    """Train Gaussian Splat using gsplat MCMCStrategy.

    Args:
        parent_dir: Job directory with sparse/0/ and images/
        total_steps: Training iterations
        cap_max: Hard Gaussian count cap (MCMC never exceeds this). None or 'auto'
                 scales it to scene complexity (~30x sparse points, clamped 0.5M-2M)
        sh_degree: Spherical harmonics degree 0-3
        progress_callback: fn(message, level) for log streaming
        output_ply_path: Where to save the PLY (default: parent_dir/gaussian_splat.ply)

    Returns:
        Path to output PLY or None on failure
    """
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy, DefaultStrategy
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
    if dataset.get('has_masks'):
        log(f"Sky masks ACTIVE — sky pixels excluded from the loss (no sky floaters)")

    if num_images == 0 or num_sparse == 0:
        log("No images or sparse points — cannot train", "ERROR")
        return None

    # Auto-cap: scale the Gaussian ceiling to scene complexity instead of a fixed
    # guess. Sparse-point count is a better signal than photo count (it folds in
    # both coverage AND texture). ~30x sparse points, clamped to a sane range so a
    # tiny scene still gets enough and a huge one doesn't blow up VRAM / PLY size.
    # GPU ceiling: never let the cap exceed what VRAM can hold during training.
    # ~1.5 KB/Gaussian persistent (params + Adam moments + grads) at SH deg 3;
    # 0.7 safety leaves room for per-frame render buffers. On the 48 GB RTX 8000
    # this lands ~8M, so the scene multiplier (below) is the real limiter, not VRAM.
    try:
        free_vram, _ = torch.cuda.mem_get_info(device)
        gpu_ceiling = int(min(8_000_000, free_vram * 0.7 / 1500))
    except Exception:
        gpu_ceiling = 8_000_000

    if cap_max is None or cap_max == 'auto':
        # Scene-driven: sparse-point count folds in both coverage AND texture.
        # Clamp low so tiny scenes still get enough; clamp high by the GPU ceiling
        # so a huge scene can't OOM. NOTE: ADC mode ignores this entirely (it grows
        # organically, no cap) — this only bounds the MCMC strategy.
        cap_max = int(min(gpu_ceiling, max(500_000, num_sparse * 30)))
        log(f"Auto cap_max = {cap_max:,} ({num_sparse:,} sparse x30, GPU ceiling {gpu_ceiling:,})")
    else:
        cap_max = int(min(int(cap_max), gpu_ceiling))
        log(f"cap_max = {cap_max:,} (explicit, GPU ceiling {gpu_ceiling:,})")

    # Scene scale from camera centers (gsplat Parser convention):
    # max distance of any camera from the average camera position, * 1.1
    w2c_all = torch.stack(dataset['w2c_mats'])                     # (C, 4, 4)
    cam_centers = torch.linalg.inv(w2c_all)[:, :3, 3]              # c2w translation
    scene_scale = float((cam_centers - cam_centers.mean(0)).norm(dim=-1).max()) * 1.1
    if scene_scale <= 0:
        scene_scale = 1.0
    log(f"Scene scale: {scene_scale:.3f}")

    log("Initialising Gaussians...")
    raw = init_gaussians_from_sparse(dataset['sparse_xyz'], dataset['sparse_rgb'], sh_degree)
    param_dict = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in raw.items()}
    )

    # Reference learning rates from gsplat examples/simple_trainer.py
    lr_map = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3,
              'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    optimizers = {
        k: SelectiveAdam([{'params': [param_dict[k]], 'lr': lr_map[k], 'name': k}],
                         eps=1e-15, betas=(0.9, 0.999))
        for k in param_dict.keys()
    }
    # Decay means lr to 1% of initial over the run (reference scheduler)
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers['means'], gamma=0.01 ** (1.0 / total_steps)
    )

    is_adc = str(strategy_name).lower() in ('adc', 'default')
    if is_adc:
        # INRIA-style adaptive density control: clone/split Gaussians where the
        # view-space gradient says detail is missing, prune transparent ones.
        # No cap — grows organically to what the scene needs (the Postshot path).
        refine_stop = int(total_steps * 0.5)
        strategy = DefaultStrategy(
            refine_start_iter=500,
            refine_stop_iter=refine_stop,
            refine_every=100,
            reset_every=3000,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            prune_opa=0.005,
            verbose=False,
        )
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        log(f"Strategy: ADC (adaptive densification) — organic growth, no cap; densify until step {refine_stop}")
    else:
        strategy = MCMCStrategy(
            cap_max=cap_max,
            refine_start_iter=500,
            refine_stop_iter=int(total_steps * 0.85),  # reference ratio: 25K of 30K
            refine_every=100,
            min_opacity=0.005,
            verbose=False,
        )
        strategy_state = strategy.initialize_state()
        log(f"Strategy: MCMC — fills toward cap {cap_max:,}")
    strategy.check_sanity(param_dict, optimizers)

    # Gaussian-windowed SSIM (pytorch_msssim). A naive avg_pool SSIM has biased
    # border statistics and negative variances whose gradients wreck training —
    # verified: 32 dB (L1 only) vs 10.5 dB (L1 + avg_pool SSIM) on synthetic data.
    try:
        from pytorch_msssim import SSIM
        ssim_module = SSIM(data_range=1.0, channel=3).to(device)

        def ssim_loss(pred, target):
            p = pred.permute(2, 0, 1).unsqueeze(0)
            t = target.permute(2, 0, 1).unsqueeze(0)
            return 1 - ssim_module(p, t)
    except ImportError:
        log("pytorch-msssim not installed — run: pip install pytorch-msssim. Training with L1 only.", "WARNING")
        ssim_loss = None

    lpips_fn = None
    if use_lpips:
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
            lpips_fn.eval()
            log(f"LPIPS (VGG) enabled — PyTorch CUDA, no buffer limit")
        except ImportError:
            log("lpips package not installed — run: pip install lpips. Continuing without LPIPS.", "WARNING")

    # MCMC relies on these regularizers to drive Gaussians toward zero opacity
    # so relocation has dead Gaussians to teleport (3DGS-MCMC paper, Eq. 12)
    opacity_reg, scale_reg = 0.01, 0.01

    log(f"Starting {'ADC' if is_adc else 'MCMC'} training: {total_steps} steps, lpips={'on' if lpips_fn else 'off'}")

    for step in range(total_steps):
        idx = np.random.randint(num_images)
        cam = dataset['cameras'][idx]
        w2c = dataset['w2c_mats'][idx].to(device)
        gt = dataset['images'][idx].to(device)
        H, W = gt.shape[0], gt.shape[1]

        K = torch.tensor([[cam['fx'], 0, cam['cx']],
                          [0, cam['fy'], cam['cy']],
                          [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

        # SH warmup: start with DC only, unlock one band every 1000 steps
        sh_degree_to_use = min(step // 1000, sh_degree)

        colors_sh = torch.cat([param_dict['sh0'], param_dict['shN']], dim=1)
        rendered, _, info = rasterization(
            means=param_dict['means'],
            quats=param_dict['quats'],
            scales=torch.exp(param_dict['scales']),
            opacities=torch.sigmoid(param_dict['opacities']),
            colors=colors_sh,
            viewmats=w2c.unsqueeze(0),
            Ks=K, width=W, height=H,
            near_plane=0.01, far_plane=1000.0,
            sh_degree=sh_degree_to_use, render_mode="RGB",
            rasterize_mode="antialiased",  # low-pass filter — stops sub-pixel Gaussians shimmering as the camera moves
        )

        pred = rendered[0]
        # Sky mask: set the sky region of pred := gt so it contributes ZERO to
        # every loss (L1/SSIM/LPIPS). The trainer then never reconstructs the sky
        # -> no sky floaters, all capacity goes to real surfaces. pred (unmasked)
        # is kept for the PSNR log.
        if dataset.get('has_masks') and dataset['masks'][idx] is not None:
            mk3 = dataset['masks'][idx].to(device).unsqueeze(-1)   # [H,W,1] 1=keep,0=sky
            pred_l = pred * mk3 + gt * (1.0 - mk3)
        else:
            pred_l = pred
        if ssim_loss is not None:
            loss = 0.8 * F.l1_loss(pred_l, gt) + 0.2 * ssim_loss(pred_l, gt)
        else:
            loss = F.l1_loss(pred_l, gt)
        # Opacity/scale L1 only for MCMC — they create the dead Gaussians MCMC
        # relocates. ADC prunes on its own, so the regularizers would just fade detail.
        if not is_adc:
            loss = loss + opacity_reg * torch.sigmoid(param_dict['opacities']).abs().mean()
            loss = loss + scale_reg * torch.exp(param_dict['scales']).abs().mean()

        if lpips_fn is not None:
            # LPIPS on a random 512px crop — full-res VGG backward at 3200px+
            # is 10x slower and can exhaust VRAM
            ch, cw = min(512, H), min(512, W)
            y0 = np.random.randint(H - ch + 1)
            x0 = np.random.randint(W - cw + 1)
            p_c, g_c = pred_l[y0:y0+ch, x0:x0+cw], gt[y0:y0+ch, x0:x0+cw]
            # Normalize to [-1, 1] and convert to [N, C, H, W] for LPIPS
            p_lp = (p_c.clamp(0, 1) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            g_lp = (g_c.clamp(0, 1) * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            loss = loss + 0.05 * lpips_fn(p_lp, g_lp).mean()

        # ADC reads the means2d gradient to decide where to densify — it must
        # retain that grad before backward.
        if is_adc:
            strategy.step_pre_backward(param_dict, optimizers, strategy_state, step, info)

        loss.backward()

        # Visibility mask for SelectiveAdam (only visible Gaussians get updated).
        # rasterization() defaults to packed mode: info["gaussian_ids"] lists the
        # Gaussians that touched this view. Non-packed fallback: radii (C, N, 2).
        N = param_dict['means'].shape[0]
        if info.get("gaussian_ids") is not None:
            vis_mask = torch.zeros(N, device=device, dtype=torch.bool)
            vis_mask[info["gaussian_ids"]] = True
        else:
            vis_mask = (info["radii"] > 0).all(dim=-1).any(dim=0)

        for opt in optimizers.values():
            opt.step(visibility=vis_mask)
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        if is_adc:
            # Clone/split/prune. packed flag must match rasterization's mode.
            packed = info.get("gaussian_ids") is not None
            strategy.step_post_backward(param_dict, optimizers, strategy_state, step, info,
                                        packed=packed)
        else:
            # Relocation + noise injection; lr couples noise magnitude to means lr
            strategy.step_post_backward(param_dict, optimizers, strategy_state, step, info,
                                        lr=means_scheduler.get_last_lr()[0])

        if step % 500 == 0 or step == total_steps - 1:
            n = param_dict['means'].shape[0]
            with torch.no_grad():
                mse = F.mse_loss(pred.clamp(0, 1), gt)
                psnr = -10.0 * torch.log10(mse + 1e-10)
            log(f"[{'ADC' if is_adc else 'MCMC'}] Step {step}/{total_steps} | loss={loss.item():.4f} | psnr={psnr.item():.2f} | gaussians={n:,}")

    if output_ply_path is None:
        output_ply_path = os.path.join(parent_dir, "gaussian_splat.ply")

    # Clean up before export. MCMC leaves a large pool of near-transparent
    # "filler" Gaussians (needed for relocation during training) plus some
    # sub-pixel-tiny ones — at render time both only cause view-dependent
    # FLICKER, not detail. Drop/clamp them so the splat is stable when moving.
    with torch.no_grad():
        means = param_dict['means'].detach()
        scales = param_dict['scales'].detach()
        quats = param_dict['quats'].detach()
        opac = param_dict['opacities'].detach()
        sh0 = param_dict['sh0'].detach()
        shN = param_dict['shN'].detach()

        # min-scale floor (log space) tied to scene scale: no sub-pixel Gaussians
        min_log_scale = float(np.log(scene_scale * 5e-5 + 1e-12))
        scales = torch.clamp(scales, min=min_log_scale)

        op = torch.sigmoid(opac).reshape(-1)
        keep = op > export_opacity_min
        n0 = means.shape[0]
        n1 = int(keep.sum().item())
        if 0 < n1 < n0:
            means, scales, quats, opac, sh0, shN = (
                means[keep], scales[keep], quats[keep], opac[keep], sh0[keep], shN[keep])
        log(f"Export prune: {n0:,} -> {n1:,} Gaussians (dropped {n0-n1:,} with opacity<={export_opacity_min}); scales floored")

    log(f"Exporting {means.shape[0]:,} Gaussians to PLY...")
    try:
        from gsplat import export_splats
        # export_splats writes values verbatim; the 3DGS PLY format stores
        # log-scales and logit-opacities, so pass the RAW params — viewers
        # apply exp/sigmoid themselves (activated values render as giant blobs)
        export_splats(means=means, scales=scales, quats=quats, opacities=opac,
                      sh0=sh0, shN=shN, format="ply", save_to=output_ply_path)
    except ImportError:
        # Fallback: manual PLY export if export_splats not available
        _export_3dgs_ply({'means': means, 'scales': scales, 'quats': quats,
                          'opacities': opac, 'sh0': sh0, 'shN': shN},
                         output_ply_path, sh_degree)

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
    parser.add_argument("--cap-max", type=int, default=None,
                        help="Gaussian cap (MCMC only); omit for auto (scales to sparse-point count)")
    parser.add_argument("--strategy", choices=["mcmc", "adc"], default="mcmc",
                        help="mcmc = fixed cap; adc = adaptive densification, grows organically (no cap)")
    args = parser.parse_args()
    result = train_mcmc(args.input, total_steps=args.steps, cap_max=args.cap_max,
                        output_ply_path=args.output, strategy_name=args.strategy)
    print("Success:" if result else "Failed:", result)
