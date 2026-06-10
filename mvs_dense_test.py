"""
Phase 2: MVS Dense Reconstruction → Convert to points3D.bin → Brush training.
Uses COLMAP patch_match_stereo + stereo_fusion for millions of dense points.
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
SPARSE_ORIG = os.path.join(TRUCK_DIR, "sparse", "0")
DENSE_DIR = os.path.join(TRUCK_DIR, "dense_mvs")
OUTPUT_DIR = os.path.join(TRUCK_DIR, "mvs_tests")

os.makedirs(DENSE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def read_ply_xyz(ply_path):
    """Read XYZ coordinates from ASCII or binary PLY file"""
    with open(ply_path, 'rb') as f:
        header_lines = []
        num_vertices = 0
        is_binary = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            header_lines.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line == 'end_header':
                # Check if binary
                if any('binary' in h.lower() for h in header_lines):
                    is_binary = True
                break
        
        if is_binary:
            # Binary PLY: x,y,z as float32 (12 bytes per vertex)
            data = f.read(num_vertices * 12)
            xyz = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
        else:
            # ASCII PLY
            xyz = []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii', errors='replace').strip().split()
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            xyz = np.array(xyz, dtype=np.float64)
        
        return xyz


def write_points3d_bin(xyz, output_path, rgb=None):
    """Write points3D.bin in COLMAP binary format"""
    num_points = len(xyz)
    
    with open(output_path, 'wb') as f:
        # Header: num_points as uint64
        f.write(struct.pack('Q', num_points))
        
        for i in range(num_points):
            # xyz as 3x double
            f.write(struct.pack('ddd', float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            # rgb as 3x uint8
            if rgb is not None:
                f.write(struct.pack('BBB', int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))
            else:
                f.write(struct.pack('BBB', 128, 128, 128))  # gray
            # error as double
            f.write(struct.pack('d', 1.0))
            # track_length = 0 (no tracks for MVS points)
            f.write(struct.pack('Q', 0))


def run_mvs():
    """Run COLMAP MVS pipeline"""
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    
    if os.path.exists(fused_ply):
        size_mb = os.path.getsize(fused_ply) / (1024 * 1024)
        log(f"MVS fused.ply already exists ({size_mb:.1f} MB) — skipping MVS")
        return True

    log("=" * 60)
    log("PHASE 2: MVS Dense Reconstruction")
    log("=" * 60)

    # Step 1: Image undistortion
    log("\n[1/3] Image undistortion...")
    undist_cmd = (
        f'"{COLMAP_PATH}" image_undistorter '
        f'--image_path "{IMAGE_PATH}" '
        f'--input_path "{SPARSE_ORIG}" '
        f'--output_path "{DENSE_DIR}" '
        f'--output_type COLMAP '
        f'--max_image_size 1920'
    )
    start = time.time()
    result = subprocess.run(undist_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    log(f"  Time: {time.time()-start:.1f}s, Exit: {result.returncode}")
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr[:500]}")
        return False

    # Step 2: Patch match stereo
    log("\n[2/3] Patch match stereo (GPU, this takes a while)...")
    pm_cmd = (
        f'"{COLMAP_PATH}" patch_match_stereo '
        f'--workspace_path "{DENSE_DIR}" '
        f'--workspace_format COLMAP '
        f'--PatchMatchStereo.geom_consistency true '
        f'--PatchMatchStereo.gpu_index -1'
    )
    start = time.time()
    result = subprocess.run(pm_cmd, shell=True, capture_output=True, text=True, timeout=7200)
    log(f"  Time: {time.time()-start:.1f}s, Exit: {result.returncode}")
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr[:500]}")
        # Try without geom_consistency
        log("  Retrying without geom_consistency...")
        pm_cmd2 = (
            f'"{COLMAP_PATH}" patch_match_stereo '
            f'--workspace_path "{DENSE_DIR}" '
            f'--workspace_format COLMAP'
        )
        result = subprocess.run(pm_cmd2, shell=True, capture_output=True, text=True, timeout=7200)
        log(f"  Retry Time: {time.time()-start:.1f}s, Exit: {result.returncode}")
        if result.returncode != 0:
            log(f"  FATAL: {result.stderr[:500]}")
            return False

    # Step 3: Stereo fusion
    log("\n[3/3] Stereo fusion (generating dense point cloud)...")
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
        log(f"  ERROR: {result.stderr[:500]}")
        return False

    if os.path.exists(fused_ply):
        size_mb = os.path.getsize(fused_ply) / (1024 * 1024)
        log(f"\n✓ MVS complete: fused.ply ({size_mb:.1f} MB)")
        return True
    return False


def convert_mvs_to_points3d():
    """Convert fused.ply to points3D.bin for Brush"""
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    points3d_bin = os.path.join(DENSE_DIR, 'points3D_from_mvs.bin')
    
    if os.path.exists(points3d_bin):
        pts = count_points(points3d_bin)
        if pts > 0:
            log(f"points3D_from_mvs.bin already exists ({pts:,} points)")
            return True

    log("\nConverting fused.ply → points3D.bin...")
    
    xyz = read_ply_xyz(fused_ply)
    log(f"  Read {len(xyz):,} points from fused.ply")
    
    # Filter outliers: remove points too far from centroid
    centroid = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    median_dist = np.median(dists)
    mask = dists < median_dist * 10  # keep points within 10x median distance
    xyz_filtered = xyz[mask]
    log(f"  Filtered: {len(xyz):,} → {len(xyz_filtered):,} points (removed outliers)")
    
    write_points3d_bin(xyz_filtered, points3d_bin)
    size_mb = os.path.getsize(points3d_bin) / (1024 * 1024)
    log(f"  Wrote points3D_from_mvs.bin: {len(xyz_filtered):,} points, {size_mb:.1f} MB")
    return True


def train_brush_mvs():
    """Train Brush with MVS dense points"""
    # Swap sparse/0 with MVS version
    mvs_points = os.path.join(DENSE_DIR, 'points3D_from_mvs.bin')
    
    # Backup current points3D.bin
    orig_pts = os.path.join(SPARSE_ORIG, 'points3D.bin')
    backup_pts = os.path.join(SPARSE_ORIG, 'points3D_backup.bin')
    if not os.path.exists(backup_pts):
        shutil.copy2(orig_pts, backup_pts)
    
    # Swap
    shutil.copy2(mvs_points, orig_pts)
    pts = count_points(orig_pts)
    log(f"\nSwapped sparse/0/points3D.bin with MVS version ({pts:,} points)")

    # Train
    name = "mvs_dense_1920_30k"
    output_ply = os.path.join(OUTPUT_DIR, f"{name}.ply")
    
    if os.path.exists(output_ply):
        size_mb = os.path.getsize(output_ply) / (1024 * 1024)
        log(f"[PASS] {name} already exists ({size_mb:.1f} MB)")
        result = {"status": "skipped", "size_mb": size_mb}
    else:
        log(f"\n{'=' * 60}")
        log(f"[RUN]  {name} — 30K steps, max-res 1920, MVS dense seed")
        log(f"{'=' * 60}")

        cmd = [
            BRUSH_PATH, TRUCK_DIR,
            "--total-steps", "30000",
            "--export-path", OUTPUT_DIR,
            "--export-name", f"{name}.ply",
            "--export-every", "30000",
            "--max-resolution", "1920",
        ]

        start_time = time.time()
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    log(f"  Brush: {line}")
            process.wait()
            elapsed = time.time() - start_time

            if process.returncode == 0 and os.path.exists(output_ply):
                size_mb = os.path.getsize(output_ply) / (1024 * 1024)
                log(f"\n[DONE] {name}: {size_mb:.1f} MB, {elapsed:.0f}s")
                result = {"status": "ok", "size_mb": size_mb, "time": elapsed}
            else:
                log(f"\n[FAIL] {name}: exit code {process.returncode}")
                result = {"status": "failed"}
        except Exception as e:
            log(f"\n[ERR] {name}: {e}")
            result = {"status": "error"}

    # Restore original
    if os.path.exists(backup_pts):
        shutil.copy2(backup_pts, orig_pts)
        log("Restored original points3D.bin")

    return result


def main():
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    log("MVS Dense Reconstruction + Brush Training")
    log("=" * 60)

    # Step 1: Run MVS
    if not run_mvs():
        log("ERROR: MVS failed!")
        return

    # Step 2: Convert to points3D.bin
    if not convert_mvs_to_points3d():
        log("ERROR: Conversion failed!")
        return

    # Step 3: Train Brush
    result = train_brush_mvs()

    # Summary
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    mvs_pts = count_points(os.path.join(DENSE_DIR, 'points3D_from_mvs.bin'))
    log("\n" + "=" * 60)
    log("MVS RESULTS")
    log("=" * 60)
    log(f"  MVS points: {mvs_pts:,}")
    log(f"  Fused PLY: {os.path.getsize(fused_ply)/(1024*1024):.1f} MB" if os.path.exists(fused_ply) else "  Fused PLY: N/A")
    if result["status"] == "ok":
        log(f"  Brush PLY: {result['size_mb']:.1f} MB, {result['time']:.0f}s")
    else:
        log(f"  Brush: {result['status']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
