"""Build a SuperSplat-viewable 3D preview of the COLMAP alignment.

Parses COLMAP's binary model DIRECTLY (no pycolmap — the bundled Python doesn't
ship it). Sparse points keep their real colors; each registered camera becomes a
small cyan frustum wireframe; everything is written as tiny Gaussians so the
existing splat viewer shows it like RealityScan's alignment view.
"""
import os
import struct
import numpy as np

# SH degree-0 basis: rendered_color = 0.5 + C0 * f_dc  ->  f_dc = (color - 0.5)/C0
_C0 = 0.28209479177387814

# COLMAP camera model id -> number of intrinsic params (params[0] is always a focal)
_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read_points3D_bin(path):
    xyz, rgb = [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(8)                                   # point3D_id
            x, y, z = struct.unpack('<ddd', f.read(24))
            r, g, b = struct.unpack('<BBB', f.read(3))
            f.read(8)                                   # reprojection error
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.read(track_len * 8)                        # skip track (image_id+idx pairs)
            xyz.append((x, y, z)); rgb.append((r, g, b))
    xyz = np.array(xyz, np.float32) if xyz else np.zeros((0, 3), np.float32)
    rgb = (np.array(rgb, np.float32) / 255.0) if rgb else np.zeros((0, 3), np.float32)
    return xyz, rgb


def _read_cameras_bin(path):
    cams = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack('<iiQQ', f.read(24))
            k = _MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack('<' + 'd' * k, f.read(8 * k))
            cams[cam_id] = {'w': int(width), 'h': int(height), 'f': float(params[0])}
    return cams


def _read_images_bin(path):
    imgs = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            d = struct.unpack('<idddddddi', f.read(64))   # id, qw,qx,qy,qz, tx,ty,tz, cam_id
            qvec = np.array(d[1:5], np.float64)
            tvec = np.array(d[5:8], np.float64)
            cam_id = d[8]
            while True:                                    # image name (null-terminated)
                c = f.read(1)
                if c in (b'\x00', b''):
                    break
            num2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num2d * 24)                             # skip 2D points (x,y double + id uint64)
            imgs.append({'qvec': qvec, 'tvec': tvec, 'cam_id': cam_id})
    return imgs


def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], np.float64)


def _rgb_to_f_dc(rgb01):
    return (np.clip(rgb01, 0.0, 1.0) - 0.5) / _C0


def build_alignment_preview(sparse_dir, output_ply, max_points=500_000):
    """Write a 3DGS PLY of sparse points + camera frustums.

    Returns (output_ply, num_points, num_cameras).
    """
    pts_path = os.path.join(sparse_dir, 'points3D.bin')
    cams_path = os.path.join(sparse_dir, 'cameras.bin')
    imgs_path = os.path.join(sparse_dir, 'images.bin')
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"points3D.bin not found in {sparse_dir}")

    xyz, rgb = _read_points3D_bin(pts_path)
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0))) if len(xyz) else 1.0
    extent = extent or 1.0
    pt_scale = max(extent * 0.0015, 1e-4)

    # --- camera frustums (bright cyan, drawn as edge dots) ---
    cam_xyz, cam_rgb = [], []
    frustum_color = np.array([0.0, 0.9, 1.0], np.float32)
    if os.path.exists(cams_path) and os.path.exists(imgs_path):
        cams = _read_cameras_bin(cams_path)
        depth = extent * 0.04
        per_edge = 14
        for im in _read_images_bin(imgs_path):
            cam = cams.get(im['cam_id'])
            if not cam:
                continue
            R = _qvec2rotmat(im['qvec'])      # world-to-camera
            Rt = R.T                          # camera-to-world rotation
            C = -Rt @ im['tvec']              # camera center in world
            f = cam['f'] or 1.0
            W, H = cam['w'], cam['h']
            corners_cam = [
                np.array([px / f * depth, py / f * depth, depth], np.float64)
                for (px, py) in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]
            ]
            corners_w = [C + Rt @ c for c in corners_cam]
            edges = [(C, cw) for cw in corners_w] + \
                    [(corners_w[i], corners_w[(i + 1) % 4]) for i in range(4)]
            for a, b in edges:
                for t in np.linspace(0.0, 1.0, per_edge):
                    cam_xyz.append((a * (1 - t) + b * t).astype(np.float32))
                    cam_rgb.append(frustum_color)

    cam_xyz = np.array(cam_xyz, np.float32) if cam_xyz else np.zeros((0, 3), np.float32)
    cam_rgb = np.array(cam_rgb, np.float32) if len(cam_rgb) else np.zeros((0, 3), np.float32)
    cam_scale = max(extent * 0.0012, 1e-4)
    num_cams = len(cam_xyz) // (14 * 8) if len(cam_xyz) else 0

    all_xyz = np.concatenate([xyz, cam_xyz], 0)
    all_rgb = np.concatenate([rgb, cam_rgb], 0)
    scales = np.concatenate([
        np.full((len(xyz),), pt_scale, np.float32),
        np.full((len(cam_xyz),), cam_scale, np.float32),
    ])
    _write_splat_ply(output_ply, all_xyz, all_rgb, scales)
    return output_ply, len(xyz), num_cams


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
    data['opacity'] = 8.0                                   # logit -> effectively opaque
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


def build_alignment_json(sparse_dir, output_json, max_points=150_000):
    """Compact JSON for the Three.js alignment viewer (clean wireframe frustums).

    { points:[x,y,z,...], colors:[r,g,b,...], frustums:[[apex3, c0..c3 (12)], ...] }
    Returns (output_json, num_points, num_cameras).
    """
    import json
    pts_path = os.path.join(sparse_dir, 'points3D.bin')
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"points3D.bin not found in {sparse_dir}")

    xyz, rgb = _read_points3D_bin(pts_path)
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0))) if len(xyz) else 1.0
    extent = extent or 1.0
    depth = extent * 0.035

    frustums, n_cams = [], 0
    cams_path = os.path.join(sparse_dir, 'cameras.bin')
    imgs_path = os.path.join(sparse_dir, 'images.bin')
    if os.path.exists(cams_path) and os.path.exists(imgs_path):
        cams = _read_cameras_bin(cams_path)
        for im in _read_images_bin(imgs_path):
            cam = cams.get(im['cam_id'])
            if not cam:
                continue
            R = _qvec2rotmat(im['qvec']); Rt = R.T
            C = -Rt @ im['tvec']
            f = cam['f'] or 1.0
            W, H = cam['w'], cam['h']
            row = [round(float(C[0]), 5), round(float(C[1]), 5), round(float(C[2]), 5)]
            for (px, py) in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]:
                wc = C + Rt @ np.array([px / f * depth, py / f * depth, depth], np.float64)
                row += [round(float(wc[0]), 5), round(float(wc[1]), 5), round(float(wc[2]), 5)]
            frustums.append(row)
            n_cams += 1

    out = {
        'points': [round(float(x), 5) for x in xyz.reshape(-1)],
        'colors': [round(float(c), 4) for c in rgb.reshape(-1)] if len(rgb) else None,
        'frustums': frustums,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w') as fp:
        json.dump(out, fp)
    return output_json, len(xyz), n_cams


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", "-s", required=True, help="path to sparse/0")
    ap.add_argument("--output", "-o", required=True, help="output .ply or .json")
    args = ap.parse_args()
    if args.output.lower().endswith('.json'):
        out, npts, ncam = build_alignment_json(args.sparse, args.output)
    else:
        out, npts, ncam = build_alignment_preview(args.sparse, args.output)
    print(f"Wrote {out}: {npts:,} points + {ncam} cameras")
