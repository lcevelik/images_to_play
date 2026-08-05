"""Learned-feature SfM front-end: SuperPoint + LightGlue -> COLMAP sparse model.

COLMAP's SIFT can't match textureless surfaces (blank walls, asphalt, sky-lit
scenes), which starves the splat trainers of seed points — e.g. only ~3,325
sparse points on a 58-image parking-lot capture. SuperPoint + LightGlue are
learning-based and match far more of those regions.

Pipeline: extract SuperPoint keypoints per image -> match every pair with
LightGlue -> import keypoints + geometrically-verified matches into a pycolmap
database -> run pycolmap incremental mapping -> write the largest reconstruction
to <output_dir>/sparse/0 (cameras.bin / images.bin / points3D.bin), exactly the
layout the MCMC/Brush trainers already consume.

Runs on CPU by default so it doesn't fight a GPU training job for VRAM; pass
device='cuda' when the GPU is free for a big speed-up.
"""
import os
import glob
import shutil
import uuid
import numpy as np


def iter_candidate_pairs(num_images, pair_window=8):
    """Yield candidate image pairs in a bounded temporal window.

    Exhaustive matching over hundreds of images is extremely slow for this
    workflow and often times out before the mapper can start. We keep a small
    local window for temporal consistency and add a few wider-baseline anchor
    pairs so the mapper still has a chance to initialize on long sequences.
    """
    if num_images <= 1:
        return []
    max_window = max(1, int(pair_window))
    pairs = []
    seen = set()
    for a in range(num_images):
        for b in range(a + 1, min(num_images, a + max_window + 1)):
            if (a, b) not in seen:
                pairs.append((a, b))
                seen.add((a, b))
        for stride in (max_window, max_window * 2, max_window * 4):
            b = a + stride
            if b < num_images and (a, b) not in seen:
                pairs.append((a, b))
                seen.add((a, b))
    return pairs


def _sanitize_matches(matches):
    """Return deduplicated, valid match rows suitable for pycolmap."""
    if matches is None:
        return np.empty((0, 2), dtype=np.uint32)
    arr = np.asarray(matches)
    if arr.ndim != 2 or arr.shape[1] != 2:
        arr = arr.reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.uint32)
    valid = (arr[:, 0] >= 0) & (arr[:, 1] >= 0)
    arr = arr[valid]
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.uint32)
    arr = np.unique(arr, axis=0)
    return arr.astype(np.uint32)


def reset_output_dir(output_dir):
    """Remove stale SfM artifacts so a re-run starts from a clean database."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    for name in ('database.db', 'database.db-journal', 'database.db-wal', 'database.db-shm', 'images.bin', 'cameras.bin', 'points3D.bin'):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                try:
                    os.replace(path, path + '.stale')
                except Exception:
                    pass
    for path in glob.glob(os.path.join(output_dir, 'database*.db*')):
        try:
            os.remove(path)
        except PermissionError:
            try:
                os.replace(path, path + '.stale')
            except Exception:
                pass
    sparse_dir = os.path.join(output_dir, 'sparse')
    if os.path.isdir(sparse_dir):
        shutil.rmtree(sparse_dir, ignore_errors=True)
    os.makedirs(sparse_dir, exist_ok=True)


def run_learned_sfm(image_dir, output_dir, device='cpu', max_kpts=2048,
                    min_matches=15, progress=print, extractor='superpoint',
                    keep_two_view_tracks=False, mapper_min_angle=1.5,
                    mapper_max_reproj=4.0, mapper_min_matches=15,
                    pair_window=8):
    import torch
    from lightglue import LightGlue
    from lightglue.utils import load_image, rbd
    import pycolmap

    os.makedirs(output_dir, exist_ok=True)  # Database.open() needs the dir to exist
    reset_output_dir(output_dir)
    dev = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    feat = str(extractor).lower()
    if feat == 'aliked':
        from lightglue import ALIKED as Extractor
    elif feat == 'disk':
        from lightglue import DISK as Extractor
    else:
        feat = 'superpoint'
        from lightglue import SuperPoint as Extractor
    ext = Extractor(max_num_keypoints=max_kpts).eval().to(dev)
    matcher = LightGlue(features=feat).eval().to(dev)

    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    imgs = sorted([f for f in glob.glob(os.path.join(image_dir, '*')) if f.endswith(exts)])
    if len(imgs) < 2:
        progress(f"[learned-sfm] need >=2 images, found {len(imgs)}")
        return None
    names = [os.path.basename(f) for f in imgs]
    progress(f"[learned-sfm] {len(imgs)} images, device={dev}, max_kpts={max_kpts}")

    # --- extract SuperPoint features ---
    feats, kpts, sizes = [], [], []
    for i, f in enumerate(imgs):
        img = load_image(f).to(dev)                      # [3,H,W] float [0,1]
        with torch.no_grad():
            ft = ext.extract(img)                        # batched dict
        feats.append(ft)
        kpts.append(rbd(ft)['keypoints'].cpu().numpy())  # [N,2] pixel coords
        sizes.append((int(img.shape[-1]), int(img.shape[-2])))  # (W,H)
        if (i + 1) % 10 == 0 or i == len(imgs) - 1:
            progress(f"[learned-sfm] extracted {i+1}/{len(imgs)}")

    # --- database: shared camera per unique image size, images, keypoints ---
    # One camera for everything assumes a single-phone capture, but a mixed
    # portrait/landscape (or mixed-resolution) set would then get wrong
    # intrinsics for every image that differs from the first one. Share a
    # camera only among images with the same (W, H).
    db_path = os.path.join(output_dir, f'database_{uuid.uuid4().hex[:8]}.db')
    db = pycolmap.Database.open(db_path)

    def _make_camera(size):
        W, H = size
        focal = 1.2 * max(W, H)
        return pycolmap.Camera(model='SIMPLE_RADIAL', width=W, height=H,
                               params=[focal, W / 2.0, H / 2.0, 0.0])

    cam_ids = {}  # (W, H) -> camera_id
    cams_by_size = {}
    for size in sizes:
        if size not in cam_ids:
            cams_by_size[size] = _make_camera(size)
            cam_ids[size] = db.write_camera(cams_by_size[size])
    if len(cam_ids) > 1:
        progress(f"[learned-sfm] {len(cam_ids)} distinct image sizes -> one camera each")

    image_ids = []
    for i, name in enumerate(names):
        image_id = i + 1
        im = pycolmap.Image(name=name, camera_id=cam_ids[sizes[i]], image_id=image_id)
        db.write_image(im, True)
        db.write_keypoints(image_id, kpts[i].astype(np.float32))
        image_ids.append(image_id)

    # --- temporal-window LightGlue matching + per-pair geometric verification ---
    opts = pycolmap.TwoViewGeometryOptions()
    nverified = 0
    for a, b in iter_candidate_pairs(len(imgs), pair_window=pair_window):
        with torch.no_grad():
            m = matcher({'image0': feats[a], 'image1': feats[b]})
        matches = rbd(m)['matches'].cpu().numpy()    # [M,2] indices
        mu = _sanitize_matches(matches)
        if len(mu) < min_matches:
            continue
        db.write_matches(image_ids[a], image_ids[b], mu)
        tvg = pycolmap.estimate_two_view_geometry(
            cams_by_size[sizes[a]], kpts[a].astype(np.float64),
            cams_by_size[sizes[b]], kpts[b].astype(np.float64), mu, opts)
        if tvg is not None and np.asarray(tvg.inlier_matches).shape[0] >= min_matches:
            db.write_two_view_geometry(image_ids[a], image_ids[b], tvg)
            nverified += 1
        if b % 10 == 0 or b == len(imgs) - 1:
            progress(f"[learned-sfm] matched through image {b+1}/{len(imgs)} (verified pairs: {nverified})")
    db.close()
    progress(f"[learned-sfm] {nverified} verified pairs; running incremental mapping...")

    # --- incremental mapping -> largest reconstruction -> sparse/0 ---
    # Loosen the mapper to KEEP more points (denser seed for splatting). The
    # biggest lever: ignore_two_view_tracks defaults True, which discards every
    # point seen in only 2 views — huge on a limited-overlap pan. Plus a lower
    # min triangulation angle and higher reproj tolerance. Splat training prunes
    # any resulting noise, so trading a little precision for density is the right call.
    opt = pycolmap.IncrementalPipelineOptions()
    opt.min_num_matches = mapper_min_matches
    opt.triangulation.ignore_two_view_tracks = (not keep_two_view_tracks)
    opt.triangulation.min_angle = mapper_min_angle
    opt.triangulation.complete_max_reproj_error = mapper_max_reproj
    opt.triangulation.merge_max_reproj_error = mapper_max_reproj
    progress(f"[learned-sfm] mapper: keep_2view={keep_two_view_tracks}, min_angle={mapper_min_angle}, "
             f"max_reproj={mapper_max_reproj}, min_matches={mapper_min_matches}")
    recs = pycolmap.incremental_mapping(db_path, image_dir, output_dir, options=opt)
    if not recs:
        progress("[learned-sfm] mapping produced NO reconstruction")
        return None
    best = max(recs.values(), key=lambda r: r.num_points3D())
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)
    best.write(sparse_dir)
    progress(f"[learned-sfm] DONE: {best.num_reg_images()}/{len(imgs)} images registered, "
             f"{best.num_points3D():,} sparse points -> {sparse_dir}")
    return sparse_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--images", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-kpts", type=int, default=2048)
    p.add_argument("--extractor", default="superpoint", choices=["superpoint", "aliked", "disk"])
    p.add_argument("--pair-window", type=int, default=8)
    a = p.parse_args()
    import time
    t0 = time.time()
    r = run_learned_sfm(a.images, a.output, device=a.device, max_kpts=a.max_kpts,
                        extractor=a.extractor, pair_window=a.pair_window,
                        progress=lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True))
    print("RESULT:", r)
