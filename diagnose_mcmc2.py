"""Diagnostic v2: render with correct scaled intrinsics, compare w2c vs c2w."""
import os, sys
sys.path.insert(0, r"F:\Codebase\images_to_play\simple_splat\App")
os.environ["PATH"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" + ";" + os.environ.get("PATH", "")

import torch
import numpy as np
from PIL import Image

from gsplat_mcmc_trainer import load_colmap_dataset, init_gaussians_from_sparse
from gsplat import rasterization

WORK_DIR = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc"
OUT_DIR = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc\diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda")

print("Loading dataset...", flush=True)
dataset = load_colmap_dataset(WORK_DIR)
print(f"Loaded {len(dataset['images'])} images, {dataset['sparse_xyz'].shape[0]} sparse points")

# Scene scale
xyz = dataset['sparse_xyz']
extent = np.max(xyz.max(0) - xyz.min(0))
print(f"Scene max extent: {extent:.1f}")
print(f"Sparse centroid: {xyz.mean(0)}")

# Init Gaussians
splats = init_gaussians_from_sparse(xyz, dataset['sparse_rgb'], sh_degree=3)
for k, v in splats.items():
    splats[k] = v.to(device)

idx = 0
cam = dataset['cameras'][idx]
gt_image = dataset['images'][idx].to(device)
H, W = gt_image.shape[0], gt_image.shape[1]

# KEY FIX: Scale intrinsics to match actual image resolution
sx = W / cam['width']
sy = H / cam['height']
print(f"\nCamera calibration: {cam['width']}x{cam['height']}")
print(f"Actual image: {W}x{H}")
print(f"Scale factors: sx={sx:.4f}, sy={sy:.4f}")

fx_scaled = cam['fx'] * sx
fy_scaled = cam['fy'] * sy
cx_scaled = cam['cx'] * sx
cy_scaled = cam['cy'] * sy
print(f"Scaled intrinsics: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")

K = torch.tensor([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]], 
                  device=device, dtype=torch.float32).unsqueeze(0)

colors_sh = torch.cat([splats['sh0'], splats['shN']], dim=1)

# The dataset stores w2c in c2w_mats (naming bug from original code)
w2c = dataset['c2w_mats'][idx].to(device).unsqueeze(0)
c2w = torch.linalg.inv(w2c)

# Camera world position
cam_pos = c2w[0, :3, 3].cpu().numpy()
print(f"Camera world position: {cam_pos}")
print(f"Camera distance from centroid: {np.linalg.cam_pos - xyz.mean(0):.2f}" if False else "")

# Auto-compute near/far from scene
dist_to_center = np.linalg.norm(cam_pos - xyz.mean(0))
near = max(0.01, dist_to_center - extent)
far = dist_to_center + extent * 2
print(f"Auto near={near:.1f}, far={far:.1f}")

# Render with W2C (what the original code passed)
print(f"\n=== Render with W2C (original code, scaled intrinsics) ===")
rendered_w2c, alpha_w2c, info_w2c = rasterization(
    means=splats['means'], quats=splats['quats'],
    scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
    colors=colors_sh, viewmats=w2c, Ks=K,
    width=W, height=H, near_plane=near, far_plane=far,
    sh_degree=3, render_mode="RGB", absgrad=True,
)

# Render with C2W (inverted)
print(f"=== Render with C2W (inverted, scaled intrinsics) ===")
rendered_c2w, alpha_c2w, info_c2w = rasterization(
    means=splats['means'], quats=splats['quats'],
    scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
    colors=colors_sh, viewmats=c2w, Ks=K,
    width=W, height=H, near_plane=near, far_plane=far,
    sh_degree=3, render_mode="RGB", absgrad=True,
)

def save_img(tensor, name):
    img = tensor[0].detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUT_DIR, name))
    print(f"  {name}: mean={img.mean():.1f}, std={img.std():.1f}, min={img.min()}, max={img.max()}")

def save_alpha(tensor, name):
    img = tensor[0].detach().cpu().numpy().squeeze()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(os.path.join(OUT_DIR, name))
    coverage = np.mean(img > 10)
    print(f"  {name}: coverage={coverage:.1%}")

print("\nW2C render:")
save_img(rendered_w2c, "render_w2c_scaled.png")
save_alpha(alpha_w2c, "alpha_w2c_scaled.png")

print("\nC2W render:")
save_img(rendered_c2w, "render_c2w_scaled.png")
save_alpha(alpha_c2w, "alpha_c2w_scaled.png")

print("\nGround truth:")
save_img(gt_image.unsqueeze(0), "ground_truth.png")

print("\nDone! Compare the renders visually.")
