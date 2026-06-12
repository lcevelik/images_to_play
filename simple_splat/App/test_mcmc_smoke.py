"""
Smoke test for gsplat_mcmc_trainer: train on a synthetic scene and assert
the loss collapses. Catches broken gradient flow, optimizer wiring, and
MCMC strategy integration without needing a COLMAP reconstruction.

Run:  python test_mcmc_smoke.py
"""

import os
import tempfile
import numpy as np
import torch

import gsplat_mcmc_trainer as trainer_mod


def look_at_w2c(eye, target=np.zeros(3), up=np.array([0.0, 0.0, 1.0])):
    """Build a COLMAP-convention (x right, y down, z forward) w2c matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])            # rows: right, down, forward
    t = -R @ eye
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def build_synthetic_dataset(device, num_cams=24, res=160, num_gt=600, num_sparse=400):
    """Render GT views of a random Gaussian scene; return a dataset dict
    shaped exactly like load_colmap_dataset's output."""
    from gsplat import rasterization

    g = torch.Generator(device='cpu').manual_seed(42)
    means = (torch.rand(num_gt, 3, generator=g) - 0.5) * 1.6
    scales = torch.full((num_gt, 3), 0.09)
    quats = torch.randn(num_gt, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.full((num_gt,), 0.9)
    colors = torch.rand(num_gt, 3, generator=g)

    fx = fy = float(res) * 1.25
    cx = cy = res / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     device=device, dtype=torch.float32).unsqueeze(0)

    cameras, w2c_mats, images = [], [], []
    for i in range(num_cams):
        ang = 2 * np.pi * i / num_cams
        eye = np.array([3.0 * np.cos(ang), 3.0 * np.sin(ang), 1.2 * np.sin(ang * 2)])
        w2c = look_at_w2c(eye)
        w2c_t = torch.from_numpy(w2c).to(device)

        with torch.no_grad():
            rendered, _, _ = rasterization(
                means=means.to(device), quats=quats.to(device),
                scales=scales.to(device), opacities=opacities.to(device),
                colors=colors.to(device),
                viewmats=w2c_t.unsqueeze(0), Ks=K,
                width=res, height=res,
                near_plane=0.01, far_plane=1000.0, render_mode="RGB",
            )
        img = rendered[0].clamp(0, 1).cpu()
        assert img.mean() > 0.01, "GT render is black — camera setup is wrong"

        cameras.append({'width': res, 'height': res, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
        w2c_mats.append(torch.from_numpy(w2c))
        images.append(img)

    # Noisy subsample of GT points simulates a sparse SfM cloud
    idx = torch.randperm(num_gt, generator=g)[:num_sparse]
    sparse_xyz = (means[idx] + torch.randn(num_sparse, 3, generator=g) * 0.05).numpy().astype(np.float32)
    sparse_rgb = colors[idx].numpy().astype(np.float32)

    return {
        'cameras': cameras, 'w2c_mats': w2c_mats,
        'images': images, 'image_names': [f"synth_{i:03d}.png" for i in range(num_cams)],
        'sparse_xyz': sparse_xyz, 'sparse_rgb': sparse_rgb,
    }


def render_psnr_from_ply(ply_path, dataset, device):
    """Reload the exported PLY exactly as a viewer would (exp scales, sigmoid
    opacity, channel-major f_rest) and render view 0. Catches export bugs like
    writing activated values into raw-format fields."""
    import torch.nn.functional as F
    from plyfile import PlyData
    from gsplat import rasterization

    v = PlyData.read(ply_path)['vertex']
    N = len(v['x'])
    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], 1), device=device)
    scales = torch.exp(torch.tensor(np.stack([v[f'scale_{i}'] for i in range(3)], 1), device=device))
    quats = torch.tensor(np.stack([v[f'rot_{i}'] for i in range(4)], 1), device=device)
    ops = torch.sigmoid(torch.tensor(np.array(v['opacity']), device=device))
    n_rest = sum(1 for p in v.properties if p.name.startswith('f_rest_'))
    sh0 = torch.tensor(np.stack([v[f'f_dc_{i}'] for i in range(3)], 1), device=device).unsqueeze(1)
    shN = torch.tensor(np.stack([v[f'f_rest_{i}'] for i in range(n_rest)], 1),
                       device=device).reshape(N, 3, -1).transpose(1, 2).contiguous()
    colors = torch.cat([sh0, shN], 1)

    cam = dataset['cameras'][0]
    w2c = dataset['w2c_mats'][0].to(device)
    gt = dataset['images'][0].to(device)
    K = torch.tensor([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]],
                     device=device).unsqueeze(0)
    with torch.no_grad():
        rend, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=ops,
                                   colors=colors, viewmats=w2c.unsqueeze(0), Ks=K,
                                   width=cam['width'], height=cam['height'],
                                   sh_degree=3, render_mode="RGB")
    return float(-10 * torch.log10(F.mse_loss(rend[0].clamp(0, 1), gt) + 1e-10))


def main():
    assert torch.cuda.is_available(), "CUDA required for this smoke test"
    device = torch.device("cuda")

    dataset = build_synthetic_dataset(device)
    trainer_mod.load_colmap_dataset = lambda parent_dir: dataset

    losses, psnrs, counts = [], [], []

    def capture(msg, level="INFO"):
        import re
        m = re.search(r'loss=([\d.]+) \| psnr=([\d.]+) \| gaussians=([\d,]+)', msg)
        if m:
            losses.append(float(m.group(1)))
            psnrs.append(float(m.group(2)))
            counts.append(int(m.group(3).replace(',', '')))

    with tempfile.TemporaryDirectory() as tmp:
        out_ply = os.path.join(tmp, "out.ply")
        result = trainer_mod.train_mcmc(
            tmp, total_steps=5000, cap_max=20000, sh_degree=3,
            use_lpips=False, progress_callback=capture, output_ply_path=out_ply,
        )

        assert result is not None, "train_mcmc returned None"
        assert os.path.exists(out_ply), "no PLY exported"
        ply_mb = os.path.getsize(out_ply) / 1024 / 1024
        ply_psnr = render_psnr_from_ply(out_ply, dataset, device)

    assert len(losses) >= 3, f"too few loss samples: {losses}"
    initial, final = losses[0], losses[-1]
    print(f"\nloss: {initial:.4f} -> {final:.4f}  ({final/initial:.1%} of initial)")
    print(f"psnr: {psnrs[0]:.2f} -> {psnrs[-1]:.2f} dB")
    print(f"gaussians: {counts[0]:,} -> {counts[-1]:,}")
    print(f"ply: {ply_mb:.1f} MB, reloaded-PLY render psnr: {ply_psnr:.2f} dB")

    assert final < 0.5 * initial, f"loss did not collapse: {initial:.4f} -> {final:.4f}"
    assert psnrs[-1] > psnrs[0] + 6, f"PSNR gain too small: {psnrs[0]:.2f} -> {psnrs[-1]:.2f}"
    assert counts[-1] > counts[0], f"MCMC never grew Gaussians: {counts[0]} -> {counts[-1]}"
    assert ply_psnr > psnrs[-1] - 6, f"exported PLY renders much worse than training ({ply_psnr:.2f} vs {psnrs[-1]:.2f}) — export format bug"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
