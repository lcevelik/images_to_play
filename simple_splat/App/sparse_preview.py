"""Build a SuperSplat-viewable 3D preview of the COLMAP alignment.

Encodes the sparse point cloud (real colors) + every camera as a small frustum
wireframe — all as tiny Gaussians — into one 3DGS PLY. This lets the existing
SuperSplat viewer show the reconstruction the way RealityScan's alignment view
does, with no second 3D engine. Reads sparse/0 via pycolmap (bundled).
"""
import os
import numpy as np

# SH degree-0 basis: rendered_color = 0.5 + C0 * f_dc  →  f_dc = (color - 0.5)/C0
_C0 = 0.28209479177387814


def _rgb_to_f_dc(rgb01):
    return (np.clip(rgb01, 0.0, 1.0) - 0.5) / _C0


def build_alignment_preview(sparse_dir, output_ply, max_points=500_000):
    """Write a 3DGS PLY of sparse points + camera frustums.

    Returns (output_ply, num_points, num_cameras).
    """
    import pycolmap
    recon = pycolmap.Reconstruction(sparse_dir)

    # --- scene points (their real colors) ---
    xyz, rgb = [], []
    for _, pt in recon.points3D.items():
        xyz.append(pt.xyz)
        rgb.append(pt.color[:3])
    xyz = np.array(xyz, dtype=np.float32) if xyz else np.zeros((0, 3), np.float32)
    rgb = (np.array(rgb, dtype=np.float32) / 255.0) if len(rgb) else np.zeros((0, 3), np.float32)

    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    # Scene extent drives point/frustum sizing so it looks right at any scale.
    if len(xyz):
        extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0))) or 1.0
    else:
        extent = 1.0
    pt_scale = max(extent * 0.0015, 1e-4)

    # --- camera frustums (bright cyan, drawn as edge dots) ---
    cam_xyz, cam_rgb = [], []
    frustum_color = np.array([0.0, 0.9, 1.0], np.float32)  # cyan
    depth = extent * 0.04
    per_edge = 14
    for _, image in recon.images.items():
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = np.array(image.cam_from_world().matrix(), dtype=np.float32)
        c2w = np.linalg.inv(w2c)
        C = c2w[:3, 3]
        R = c2w[:3, :3]
        cam = recon.cameras[image.camera_id]
        fx = float(cam.focal_length_x) or 1.0
        fy = float(cam.focal_length_y) or fx
        W, H = cam.width, cam.height

        # image-plane corners projected to `depth` in camera space, then to world
        corners_cam = [
            np.array([(px) / fx * depth, (py) / fy * depth, depth], np.float32)
            for (px, py) in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]
        ]
        corners_w = [C + R @ c for c in corners_cam]
        edges = [(C, cw) for cw in corners_w]                       # apex → corners
        edges += [(corners_w[i], corners_w[(i + 1) % 4]) for i in range(4)]  # image rectangle
        for a, b in edges:
            for t in np.linspace(0.0, 1.0, per_edge):
                cam_xyz.append(a * (1 - t) + b * t)
                cam_rgb.append(frustum_color)

    cam_xyz = np.array(cam_xyz, np.float32) if cam_xyz else np.zeros((0, 3), np.float32)
    cam_rgb = np.array(cam_rgb, np.float32) if len(cam_rgb) else np.zeros((0, 3), np.float32)
    cam_scale = max(extent * 0.0012, 1e-4)

    all_xyz = np.concatenate([xyz, cam_xyz], 0)
    all_rgb = np.concatenate([rgb, cam_rgb], 0)
    scales = np.concatenate([
        np.full((len(xyz),), pt_scale, np.float32),
        np.full((len(cam_xyz),), cam_scale, np.float32),
    ])
    _write_splat_ply(output_ply, all_xyz, all_rgb, scales)
    return output_ply, len(xyz), len(recon.images)


def _write_splat_ply(path, xyz, rgb, scales):
    """Write a minimal binary 3DGS PLY (SH degree 0) the splat viewer can load."""
    N = len(xyz)
    f_dc = _rgb_to_f_dc(rgb)
    props = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz'] +
        ['f_dc_0', 'f_dc_1', 'f_dc_2'] +
        ['opacity'] +
        ['scale_0', 'scale_1', 'scale_2'] +
        ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    )
    dtype = np.dtype([(p, '<f4') for p in props])
    data = np.zeros(N, dtype=dtype)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    data['opacity'] = 8.0                                   # logit → effectively opaque
    log_scale = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)
    data['scale_0'] = data['scale_1'] = data['scale_2'] = log_scale
    data['rot_0'] = 1.0                                     # identity quaternion (isotropic)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n" +
        "".join(f"property float {p}\n" for p in props) +
        "end_header\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", "-s", required=True, help="path to sparse/0")
    ap.add_argument("--output", "-o", required=True, help="output .ply")
    args = ap.parse_args()
    out, npts, ncam = build_alignment_preview(args.sparse, args.output)
    print(f"Wrote {out}: {npts:,} points + {ncam} cameras")
