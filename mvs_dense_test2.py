"""
MVS pipeline with upscaled images to match COLMAP reconstruction resolution.
COLMAP recon: 1957x1091, images: 979x546 (2x downscale).
"""
import os
import sys
import subprocess
import time
import struct
import shutil
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TRUCK_DIR = r"F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck"
BRUSH_PATH = r"F:\Codebase\images_to_play\Brush\brush_app.exe"
COLMAP_PATH = r"C:\COLMAP\bin\colmap.exe"

IMAGE_PATH = os.path.join(TRUCK_DIR, "images")
IMAGE_HIRES = os.path.join(TRUCK_DIR, "images_hires")
SPARSE_ORIG = os.path.join(TRUCK_DIR, "sparse", "0")
DENSE_DIR = os.path.join(TRUCK_DIR, "dense_mvs")
OUTPUT_DIR = os.path.join(TRUCK_DIR, "mvs_tests")

os.makedirs(DENSE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_HIRES, exist_ok=True)

LOG_FILE = r"F:\Codebase\images_to_play\mvs_log.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def count_points(points3d_path):
    if not os.path.exists(points3d_path):
        return 0
    with open(points3d_path, 'rb') as f:
        data = f.read()
    if len(data) < 8:
        return 0
    return struct.unpack('Q', data[0:8])[0]

def upscale_images():
    """Upscale images from 979x546 to 1957x1091 to match COLMAP reconstruction"""
    hires_files = os.listdir(IMAGE_HIRES)
    orig_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(('.jpg', '.png'))]
    
    if len(hires_files) >= len(orig_files):
        log(f"Already have {len(hires_files)} hi-res images — skipping upscale")
        return True

    log(f"Upscaling {len(orig_files)} images from 979→1957 (2x)...")
    
    try:
        from PIL import Image
    except ImportError:
        log("ERROR: PIL not installed!")
        return False
    
    start = time.time()
    for i, fname in enumerate(orig_files):
        src = os.path.join(IMAGE_PATH, fname)
        dst = os.path.join(IMAGE_HIRES, fname)
        if os.path.exists(dst):
            continue
        img = Image.open(src)
        img_hires = img.resize((1957, 1091), Image.LANCZOS)
        img_hires.save(dst, quality=95)
        if (i+1) % 50 == 0:
            log(f"  Upscaled {i+1}/{len(orig_files)}")
    
    log(f"  Done in {time.time()-start:.1f}s")
    return True


def run_mvs():
    """Run COLMAP MVS pipeline with hi-res images"""
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    
    if os.path.exists(fused_ply):
        size_mb = os.path.getsize(fused_ply) / (1024 * 1024)
        log(f"MVS fused.ply already exists ({size_mb:.1f} MB) — skipping")
        return True

    log("=" * 60)
    log("MVS Dense Reconstruction (hi-res images)")
    log("=" * 60)

    # Step 1: Image undistortion with hi-res images
    log("\n[1/3] Image undistortion...")
    undist_cmd = (
        f'"{COLMAP_PATH}" image_undistorter '
        f'--image_path "{IMAGE_HIRES}" '
        f'--input_path "{SPARSE_ORIG}" '
        f'--output_path "{DENSE_DIR}" '
        f'--output_type COLMAP '
        f'--max_image_size 1957'
    )
    start = time.time()
    result = subprocess.run(undist_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    log(f"  Time: {time.time()-start:.1f}s, Exit: {result.returncode}")
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[:500]}")
        return False

    # Step 2: Patch match stereo
    log("\n[2/3] Patch match stereo (GPU, ~30-60 min)...")
    pm_cmd = (
        f'"{COLMAP_PATH}" patch_match_stereo '
        f'--workspace_path "{DENSE_DIR}" '
        f'--workspace_format COLMAP '
        f'--PatchMatchStereo.geom_consistency true'
    )
    start = time.time()
    # Stream output for progress
    process = subprocess.Popen(pm_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line and ('Processing view' in line or 'Timer' in line or 'ERROR' in line):
            log(f"  COLMAP: {line}")
    process.wait()
    elapsed = time.time() - start
    log(f"  Time: {elapsed:.1f}s, Exit: {process.returncode}")
    
    if process.returncode != 0:
        log("  Retrying without geom_consistency...")
        pm_cmd2 = (
            f'"{COLMAP_PATH}" patch_match_stereo '
            f'--workspace_path "{DENSE_DIR}" '
            f'--workspace_format COLMAP'
        )
        process = subprocess.Popen(pm_cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line and ('Processing view' in line or 'ERROR' in line):
                log(f"  COLMAP: {line}")
        process.wait()
        if process.returncode != 0:
            log("  FATAL: Patch match failed")
            return False

    # Step 3: Stereo fusion
    log("\n[3/3] Stereo fusion...")
    fusion_cmd = (
        f'"{COLMAP_PATH}" stereo_fusion '
        f'--workspace_path "{DENSE_DIR}" '
        f'--workspace_format COLMAP '
        f'--input_type geometric '
        f'--output_path "{fused_ply}" '
        f'--StereoFusion.min_num_pixels 3 '
        f'--StereoFusion.max_reproj_error 4.0 '
        f'--StereoFusion.max_depth_error 0.02 '
        f'--StereoFusion.max_normal_error 15 '
        f'--StereoFusion.check_num_images 5'
    )
    start = time.time()
    result = subprocess.run(fusion_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    log(f"  Time: {time.time()-start:.1f}s, Exit: {result.returncode}")
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[:500]}")
        return False

    if os.path.exists(fused_ply):
        size_mb = os.path.getsize(fused_ply) / (1024 * 1024)
        log(f"\n✓ MVS complete: fused.ply ({size_mb:.1f} MB)")
        return True
    return False


def convert_and_train():
    """Convert fused.ply to points3D.bin and train Brush"""
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    
    # Read PLY
    log("\nReading fused.ply...")
    with open(fused_ply, 'rb') as f:
        header_lines = []
        num_vertices = 0
        is_binary = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line == 'end_header':
                is_binary = 'binary' in ' '.join(header_lines).lower()
                break
        
        if is_binary:
            data = f.read(num_vertices * 12)
            xyz = np.frombuffer(data, dtype=np.float32).reshape(-1, 3).astype(np.float64)
        else:
            xyz = []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii', errors='replace').strip().split()
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            xyz = np.array(xyz)

    log(f"  Read {len(xyz):,} MVS points")
    
    # Filter outliers
    centroid = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    median_dist = np.median(dists)
    mask = dists < median_dist * 10
    xyz = xyz[mask]
    log(f"  After outlier filter: {len(xyz):,} points")

    # Write points3D.bin
    pts_bin = os.path.join(DENSE_DIR, 'points3D_from_mvs.bin')
    with open(pts_bin, 'wb') as f:
        f.write(struct.pack('Q', len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack('ddd', xyz[i,0], xyz[i,1], xyz[i,2]))
            f.write(struct.pack('BBB', 128, 128, 128))
            f.write(struct.pack('d', 1.0))
            f.write(struct.pack('Q', 0))
    
    log(f"  Wrote points3D_from_mvs.bin: {len(xyz):,} points")

    # Swap and train
    orig_pts = os.path.join(SPARSE_ORIG, 'points3D.bin')
    backup_pts = os.path.join(SPARSE_ORIG, 'points3D_backup.bin')
    if not os.path.exists(backup_pts):
        shutil.copy2(orig_pts, backup_pts)
    
    shutil.copy2(pts_bin, orig_pts)
    log(f"  Swapped sparse/0/points3D.bin ({len(xyz):,} MVS points)")

    name = "mvs_dense_1920_30k"
    output_ply = os.path.join(OUTPUT_DIR, f"{name}.ply")
    
    if os.path.exists(output_ply):
        size_mb = os.path.getsize(output_ply) / (1024 * 1024)
        log(f"\n[PASS] {name} exists ({size_mb:.1f} MB)")
        result = {"status": "ok", "size_mb": size_mb}
    else:
        log(f"\n{'=' * 60}")
        log(f"[RUN] {name} — 30K steps, 1920 res, MVS seed ({len(xyz):,} pts)")
        log(f"{'=' * 60}")

        cmd = [BRUSH_PATH, TRUCK_DIR,
               "--total-steps", "30000",
               "--export-path", OUTPUT_DIR,
               "--export-name", f"{name}.ply",
               "--export-every", "30000",
               "--max-resolution", "1920"]

        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                log(f"  Brush: {line.strip()}")
        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0 and os.path.exists(output_ply):
            size_mb = os.path.getsize(output_ply) / (1024 * 1024)
            log(f"\n[DONE] {name}: {size_mb:.1f} MB, {elapsed:.0f}s")
            result = {"status": "ok", "size_mb": size_mb, "time": elapsed}
        else:
            log(f"\n[FAIL] exit code {process.returncode}")
            result = {"status": "failed"}

    # Restore
    if os.path.exists(backup_pts):
        shutil.copy2(backup_pts, orig_pts)
        log("Restored original points3D.bin")

    return result


def main():
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    log("MVS Dense Reconstruction + Brush Training")
    log("=" * 60)

    if not upscale_images():
        log("ERROR: Image upscale failed!")
        return

    if not run_mvs():
        log("ERROR: MVS failed!")
        return

    result = convert_and_train()

    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    mvs_pts = count_points(os.path.join(DENSE_DIR, 'points3D_from_mvs.bin'))
    log(f"  MVS points: {mvs_pts:,}")
    if result.get("status") == "ok":
        log(f"  Brush PLY: {result['size_mb']:.1f} MB")
    log("=" * 60)


if __name__ == "__main__":
    main()
