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
import numpy as np


def run_learned_sfm(image_dir, output_dir, device='cpu', max_kpts=2048,
                    min_matches=15, progress=print):
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    import pycolmap

    os.makedirs(output_dir, exist_ok=True)  # Database.open() needs the dir to exist
    dev = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(dev)
    matcher = LightGlue(features='superpoint').eval().to(dev)

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
            ft = extractor.extract(img)                  # batched dict
        feats.append(ft)
        kpts.append(rbd(ft)['keypoints'].cpu().numpy())  # [N,2] pixel coords
        sizes.append((int(img.shape[-1]), int(img.shape[-2])))  # (W,H)
        if (i + 1) % 10 == 0 or i == len(imgs) - 1:
            progress(f"[learned-sfm] extracted {i+1}/{len(imgs)}")

    # --- database: one shared camera (single-phone capture), images, keypoints ---
    db_path = os.path.join(output_dir, 'database.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    db = pycolmap.Database.open(db_path)
    W, H = sizes[0]
    focal = 1.2 * max(W, H)
    cam = pycolmap.Camera(model='SIMPLE_RADIAL', width=W, height=H,
                          params=[focal, W / 2.0, H / 2.0, 0.0])
    cam_id = db.write_camera(cam)
    image_ids = []
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=cam_id, image_id=i + 1)
        db.write_image(im, True)
        db.write_keypoints(i + 1, kpts[i].astype(np.float32))
        image_ids.append(i + 1)

    # --- exhaustive LightGlue matching + per-pair geometric verification ---
    opts = pycolmap.TwoViewGeometryOptions()
    nverified = 0
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            with torch.no_grad():
                m = matcher({'image0': feats[a], 'image1': feats[b]})
            matches = rbd(m)['matches'].cpu().numpy()    # [M,2] indices
            if len(matches) < min_matches:
                continue
            mu = matches.astype(np.uint32)
            db.write_matches(image_ids[a], image_ids[b], mu)
            tvg = pycolmap.estimate_two_view_geometry(
                cam, kpts[a].astype(np.float64), cam, kpts[b].astype(np.float64), mu, opts)
            if tvg is not None and np.asarray(tvg.inlier_matches).shape[0] >= min_matches:
                db.write_two_view_geometry(image_ids[a], image_ids[b], tvg)
                nverified += 1
        progress(f"[learned-sfm] matched through image {a+1}/{len(imgs)} (verified pairs: {nverified})")
    db.close()
    progress(f"[learned-sfm] {nverified} verified pairs; running incremental mapping...")

    # --- incremental mapping -> largest reconstruction -> sparse/0 ---
    recs = pycolmap.incremental_mapping(db_path, image_dir, output_dir)
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
    a = p.parse_args()
    import time
    t0 = time.time()
    r = run_learned_sfm(a.images, a.output, device=a.device, max_kpts=a.max_kpts,
                        progress=lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True))
    print("RESULT:", r)
