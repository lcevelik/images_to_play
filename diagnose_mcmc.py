"""Diagnostic: render one frame from the trained model and save as PNG."""
import os, sys
sys.path.insert(0, r"F:\Codebase\images_to_play\simple_splat\App")
os.environ["PATH"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" + ";" + os.environ.get("PATH", "")

import torch
import numpy as np
from PIL import Image

from gsplat_mcmc_trainer import load_colmap_dataset
from gsplat import rasterization

WORK_DIR = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc"
OUT_DIR = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc\diagnostics"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda")

# Load dataset
print("Loading dataset...", flush=True)
dataset = load_colmap_dataset(WORK_DIR)
print(f"Loaded {len(dataset['images'])} images")

# Print sparse point cloud scale
xyz = dataset['sparse_xyz']
print(f"\nSparse points: {xyz.shape[0]}")
print(f"  X range: [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]")
print(f"  Y range: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]")
print(f"  Z range: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")
print(f"  Centroid: [{xyz.mean(0)[0]:.3f}, {xyz.mean(0)[1]:.3f}, {xyz.mean(0)[2]:.3f}]")
print(f"  Max extent: {np.max(xyz.max(0) - xyz.min(0)):.3f}")

# Print first camera info
cam = dataset['cameras'][0]
print(f"\nCamera 0:")
print(f"  fx={cam['fx']:.2f}, fy={cam['fy']:.2f}")
print(f"  cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")
print(f"  W={cam['width']}, H={cam['height']}")

# Print first image dimensions
img0 = dataset['images'][0]
print(f"\nImage 0 tensor shape: {img0.shape} (H, W, C)")

# Print viewmat (w2c)
w2c = dataset['c2w_mats'][0].numpy()
print(f"\nw2c matrix (first camera):")
print(w2c)
print(f"  det(R): {np.linalg.det(w2c[:3,:3]):.4f}")

# Invert to get c2w
c2w = np.linalg.inv(w2c)
print(f"\nc2w matrix (inverted):")
print(c2w)

# Camera position in world space
cam_pos = c2w[:3, 3]
print(f"\nCamera position (c2w translation): {cam_pos}")
print(f"Distance from origin: {np.linalg.norm(cam_pos):.3f}")

# Initialize Gaussians from sparse points
from gsplat_mcmc_trainer import init_gaussians_from_sparse
splats = init_gaussians_from_sparse(dataset['sparse_xyz'], dataset['sparse_rgb'], sh_degree=3)

print(f"\nGaussians: {splats['means'].shape[0]}")
print(f"  Scale range: [{splats['scales'].min():.6f}, {splats['scales'].max():.6f}]")
print(f"  Opacity range: [{splats['opacities'].min():.6f}, {splats['opacities'].max():.6f}]")
print(f"  sh0 range: [{splats['sh0'].min():.4f}, {splats['sh0'].max():.4f}]")

# Move to device
for k, v in splats.items():
    splats[k] = v.to(device)

# Render one frame using W2C (which is what gsplat actually expects)
H, W = img0.shape[0], img0.shape[1]
K = torch.tensor([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]], 
                  device=device, dtype=torch.float32).unsqueeze(0)

viewmat = dataset['c2w_mats'][0].to(device).unsqueeze(0)  # This is w2c
colors_sh = torch.cat([splats['sh0'], splats['shN']], dim=1)

print(f"\nRendering {W}x{H} with viewmat (w2c)...")
rendered, alpha, info = rasterization(
    means=splats['means'], quats=splats['quats'],
    scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
    colors=colors_sh, viewmats=viewmat, Ks=K,
    width=W, height=H, near_plane=0.01, far_plane=1000.0,
    sh_degree=3, render_mode="RGB", absgrad=True,
)

# Also try inverted (c2w) for comparison
viewmat_inv = torch.linalg.inv(dataset['c2w_mats'][0].to(device)).unsqueeze(0)
rendered_inv, alpha_inv, _ = rasterization(
    means=splats['means'], quats=splats['quats'],
    scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
    colors=colors_sh, viewmats=viewmat_inv, Ks=K,
    width=W, height=H, near_plane=0.01, far_plane=1000.0,
    sh_degree=3, render_mode="RGB", absgrad=True,
)

# Save all images
def save_img(tensor, name):
    img = tensor[0].detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUT_DIR, name))
    print(f"  {name}: mean={img.mean():.1f}, std={img.std():.1f}, min={img.min()}, max={img.max()}")

print("\nRendered with w2c (original code):")
save_img(rendered, "render_w2c.png")

print("Rendered with c2w (inverted):")
save_img(rendered_inv, "render_c2w.png")

print("Ground truth:")
save_img(img0.unsqueeze(0), "ground_truth.png")

print("Alpha (w2c):")
alpha_img = alpha[0].detach().cpu().numpy()
alpha_img = np.clip(alpha_img * 255, 0, 255).astype(np.uint8)
Image.fromarray(alpha_img, mode='L').save(os.path.join(OUT_DIR, "alpha_w2c.png"))
print(f"  alpha_w2c: mean={alpha_img.mean():.1f}, coverage={np.mean(alpha_img > 0):.2%}")

print("\nDone! Check", OUT_DIR)
