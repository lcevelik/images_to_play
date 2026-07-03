"""Depth-seeded densification (Sony-inspired): back-project monocular depth into
dense 3D points on textureless surfaces where SfM finds nothing.

Per image: Depth-Anything-V2 gives affine-invariant inverse depth. We align its
scale+shift to the sparse COLMAP points visible in that image (the metric anchors),
back-project a pixel grid to 3D, accumulate across views, voxel-downsample, and
merge with the sparse seed. Sky/far points are dropped (Brush handles those).

Output: a points-only PLY (x,y,z + uchar rgb) usable as train_mcmc(init_ply=...).

CLI: python pipeline/depth_seed.py <colmap_parent_dir> <out.ply> [--stride 8] [--model small|base|large]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HF = {
    'small': 'depth-anything/Depth-Anything-V2-Small-hf',
    'base':  'depth-anything/Depth-Anything-V2-Base-hf',
    'large': 'depth-anything/Depth-Anything-V2-Large-hf',
}


def _fit_affine(pred, target):
    """Least-squares s,t for s*pred + t ~= target, with one outlier-rejection pass."""
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    s, t = np.linalg.lstsq(A, target, rcond=None)[0]
    resid = np.abs(s * pred + t - target)
    keep = resid < (np.median(resid) * 3 + 1e-9)
    if keep.sum() > 10:
        s, t = np.linalg.lstsq(A[keep], target[keep], rcond=None)[0]
    return float(s), float(t)


def _voxel_downsample(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def depth_seed(parent_dir, out_ply, stride=8, model='small', merge_sparse=True,
               far_mult=3.0, gap_mult=0.0, device='cuda', progress=print):
    import torch
    from PIL import Image
    import pycolmap
    from transformers import pipeline as hf_pipeline

    sparse_dir = os.path.join(parent_dir, 'sparse', '0')
    images_dir = os.path.join(parent_dir, 'images')
    recon = pycolmap.Reconstruction(sparse_dir)
    progress(f"[depth-seed] {len(recon.images)} images, {len(recon.points3D)} sparse points")

    # metric depth range of the sparse cloud, for sky/far rejection
    all_sp = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float64)

    progress(f"[depth-seed] loading Depth-Anything-V2-{model} ...")
    pipe = hf_pipeline('depth-estimation', model=_HF[model], device=0 if device == 'cuda' else -1)

    acc_xyz, acc_rgb = [], []
    img_lookup = {f.lower(): f for f in os.listdir(images_dir)}

    for ii, (_, image) in enumerate(recon.images.items()):
        fname = img_lookup.get(image.name.lower())
        if not fname:
            continue
        pil = Image.open(os.path.join(images_dir, fname)).convert('RGB')
        W, H = pil.size
        rgb_arr = np.asarray(pil)

        out = pipe(pil)
        pred = out['predicted_depth']
        if hasattr(pred, 'detach'):
            pred = pred.detach().float().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != (H, W):
            pt = torch.from_numpy(pred)[None, None]
            pred = torch.nn.functional.interpolate(pt, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()

        cam = recon.cameras[image.camera_id]
        fx, fy = cam.focal_length_x, cam.focal_length_y
        cx, cy = cam.principal_point_x, cam.principal_point_y
        w2c = np.array(image.cam_from_world().matrix(), dtype=np.float64)  # 3x4 [R|t]
        R, t = w2c[:, :3], w2c[:, 3]

        # 2D-3D anchors: project visible sparse points into this view
        us, vs, zs = [], [], []
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            X = recon.points3D[p2d.point3D_id].xyz
            Xc = R @ X + t
            if Xc[2] <= 0:
                continue
            u, v = p2d.xy
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                us.append(ui); vs.append(vi); zs.append(Xc[2])
        if len(zs) < 20:
            continue
        us, vs, zs = np.array(us), np.array(vs), np.array(zs)
        pred_at = pred[vs, us]
        # align inverse-depth: s*pred + t ~= 1/z_metric (disparity)
        s, b = _fit_affine(pred_at, 1.0 / zs)
        if not np.isfinite(s) or not np.isfinite(b) or s <= 0:
            continue   # bad fit (depth not positively correlated) -> skip frame

        # back-project a pixel grid
        gy, gx = np.mgrid[0:H:stride, 0:W:stride]
        gx = gx.ravel(); gy = gy.ravel()
        pv = pred[gy, gx]
        disp = s * pv + b
        z = 1.0 / disp
        # reject sky/far/non-finite: keep only finite, in front, within the metric band
        far = np.percentile(zs, 99) * far_mult
        near = np.percentile(zs, 1) * 0.5
        m = np.isfinite(pv) & np.isfinite(z) & (disp > 1e-4) & (z > near) & (z < far)
        gx, gy, z = gx[m], gy[m], z[m]

        x = (gx - cx) / fx * z
        y = (gy - cy) / fy * z
        Xc = np.stack([x, y, z], axis=1)
        Xw = (Xc - t) @ R  # R^T @ (Xc - t)  == (Xc - t) @ R
        acc_xyz.append(Xw)
        acc_rgb.append(rgb_arr[gy, gx])

        if (ii + 1) % 10 == 0:
            progress(f"[depth-seed]   {ii+1} views, {sum(len(a) for a in acc_xyz):,} raw pts")

    if not acc_xyz:
        # np.concatenate([]) raises a bare ValueError — name the actual problem
        raise RuntimeError("depth-seed produced no points: no view had enough "
                           "sparse anchors (>=20) to align its depth map")
    xyz = np.concatenate(acc_xyz).astype(np.float64)
    rgb = np.concatenate(acc_rgb).astype(np.uint8)
    progress(f"[depth-seed] {len(xyz):,} raw depth points")

    # drop non-finite and hard-crop to the sparse scene volume + 15% margin
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    lo, hi = np.percentile(all_sp, 1, axis=0), np.percentile(all_sp, 99, axis=0)
    margin = (hi - lo) * 0.15
    lo, hi = lo - margin, hi + margin
    inside = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    xyz, rgb = xyz[inside], rgb[inside]
    progress(f"[depth-seed] {len(xyz):,} after finite + scene-volume crop")

    # voxel downsample at a fraction of scene scale
    scene_scale = np.linalg.norm(all_sp.std(axis=0)) + 1e-9
    voxel = scene_scale * 0.004
    xyz, rgb = _voxel_downsample(xyz, rgb, voxel)
    progress(f"[depth-seed] {len(xyz):,} after voxel downsample (voxel={voxel:.4f})")

    # GAP-FILL: drop depth points near existing sparse SfM points (those surfaces
    # already have clean geometry — overwriting them with noisy monocular depth makes
    # them wavy). Keep ONLY depth points in textureless gaps far from any sparse point.
    if gap_mult and gap_mult > 0:
        from scipy.spatial import cKDTree
        stree = cKDTree(all_sp)
        dnn, _ = stree.query(all_sp, k=2)
        sp_spacing = float(np.median(dnn[:, 1]))
        thresh = sp_spacing * gap_mult
        d2sparse, _ = stree.query(xyz)
        keep = d2sparse > thresh
        progress(f"[depth-seed] gap-fill: sparse_spacing={sp_spacing:.3f}, keep depth dist>{thresh:.3f} "
                 f"-> {int(keep.sum()):,}/{len(xyz):,} depth points (dropped {int((~keep).sum()):,} on covered surfaces)")
        xyz, rgb = xyz[keep], rgb[keep]

    if merge_sparse:
        sp_rgb = np.array([p.color for p in recon.points3D.values()], dtype=np.uint8)
        xyz = np.concatenate([xyz, all_sp])
        rgb = np.concatenate([rgb, sp_rgb])
        progress(f"[depth-seed] {len(xyz):,} after merging sparse seed")

    _write_points_ply(out_ply, xyz, rgb)
    progress(f"[depth-seed] -> {out_ply}")
    return out_ply


def build_dense_seed_dir(parent_dir, out_dir, gap_mult=12.0, stride=8, model='small', progress=print):
    """Run gap-fill depth-seed on a COLMAP scene (parent_dir with sparse/0 + images)
    and write a new COLMAP dir (out_dir) with the SAME cameras/poses but a dense
    points3D.bin (sparse SfM + gap-filled depth). Ready for train_mcmc / Brush.

    Returns out_dir. Consolidates the manual seed -> points3D.bin steps.
    """
    import shutil
    import struct
    import pycolmap

    seed_ply = os.path.join(out_dir, '_depthseed.ply')
    os.makedirs(out_dir, exist_ok=True)
    depth_seed(parent_dir, seed_ply, stride=stride, model=model, gap_mult=gap_mult, progress=progress)

    # COLMAP dir: copy cameras/poses (everything but points3D), write dense points3D.bin
    src_sparse = os.path.join(parent_dir, 'sparse', '0')
    dst_sparse = os.path.join(out_dir, 'sparse', '0')
    os.makedirs(dst_sparse, exist_ok=True)
    for fn in os.listdir(src_sparse):
        if fn != 'points3D.bin':
            shutil.copy2(os.path.join(src_sparse, fn), os.path.join(dst_sparse, fn))

    with open(seed_ply, 'rb') as f:
        hdr = b''
        while b'end_header\n' not in hdr:
            hdr += f.read(1)
        n = int([l for l in hdr.decode().splitlines() if l.startswith('element vertex')][0].split()[-1])
        P = np.frombuffer(f.read(n * 15), dtype=np.dtype(
            [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]))
    dt = np.dtype([('id', '<u8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('err', '<f8'), ('tl', '<u8')])
    o = np.empty(n, dtype=dt)
    o['id'] = np.arange(1, n + 1)
    o['x'], o['y'], o['z'] = P['x'], P['y'], P['z']
    o['r'], o['g'], o['b'] = P['r'], P['g'], P['b']
    o['err'] = 1.0
    o['tl'] = 0
    with open(os.path.join(dst_sparse, 'points3D.bin'), 'wb') as f:
        f.write(struct.pack('<Q', n))
        f.write(o.tobytes())

    # images: point Brush/trainer at the undistorted images
    src_imgs = os.path.join(parent_dir, 'images')
    dst_imgs = os.path.join(out_dir, 'images')
    if os.path.isdir(src_imgs) and not os.path.isdir(dst_imgs):
        shutil.copytree(src_imgs, dst_imgs)
    progress(f"[depth-seed] dense COLMAP seed ready: {dst_sparse} ({n:,} points)")
    return out_dir


def _write_points_ply(path, xyz, rgb):
    n = len(xyz)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(n, dtype=dt)
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())


if __name__ == '__main__':
    parent, outp = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[sys.argv.index('--stride') + 1]) if '--stride' in sys.argv else 8
    model = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'small'
    gap_mult = float(sys.argv[sys.argv.index('--gap-mult') + 1]) if '--gap-mult' in sys.argv else 0.0
    depth_seed(parent, outp, stride=stride, model=model, gap_mult=gap_mult)
