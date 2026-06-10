"""
Fix: Convert MVS fused.ply (14.96M points) to points3D.bin and retrain Brush.
Fixes: NaN/Inf handling, proper binary PLY parsing.
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

SPARSE_ORIG = os.path.join(TRUCK_DIR, "sparse", "0")
DENSE_DIR = os.path.join(TRUCK_DIR, "dense_mvs")
OUTPUT_DIR = os.path.join(TRUCK_DIR, "mvs_tests")
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
    """Read XYZ from binary PLY, handling NaN/Inf"""
    log(f"Reading {ply_path}...")
    
    with open(ply_path, 'rb') as f:
        num_vertices = 0
        is_binary = False
        props = []
        
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                props.append(line.split()[-1])
            elif line.startswith('format'):
                is_binary = 'binary' in line
            elif line == 'end_header':
                break
        
        log(f"  Format: {'binary' if is_binary else 'ascii'}, {num_vertices:,} vertices, props: {props[:6]}")
        
        # Find x,y,z property indices
        xyz_idx = [props.index(p) for p in ['x', 'y', 'z']]
        
        if is_binary:
            bytes_per_vertex = len(props) * 4  # float32 = 4 bytes
            data = f.read(num_vertices * bytes_per_vertex)
            all_data = np.frombuffer(data, dtype=np.float32).reshape(-1, len(props))
            xyz = all_data[:, xyz_idx].copy()
        else:
            xyz = []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii', errors='replace').strip().split()
                xyz.append([float(parts[i]) for i in xyz_idx])
            xyz = np.array(xyz, dtype=np.float64)
    
    log(f"  Raw: {len(xyz):,} points")
    return xyz


def filter_points(xyz):
    """Filter NaN, Inf, and outliers"""
    # Remove NaN/Inf
    valid = np.isfinite(xyz).all(axis=1)
    log(f"  After NaN/Inf filter: {valid.sum():,} / {len(xyz):,}")
    xyz = xyz[valid]
    
    if len(xyz) == 0:
        return xyz
    
    # Remove outliers (beyond 5x median distance from centroid)
    centroid = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    median_dist = np.median(dists)
    
    if median_dist > 0:
        mask = dists < median_dist * 5
        log(f"  After outlier filter: {mask.sum():,} / {len(xyz):,} (median_dist={median_dist:.2f})")
        xyz = xyz[mask]
    
    return xyz


def write_points3d_bin(xyz, output_path):
    """Write points3D.bin — no tracks (MVS points)"""
    num_points = len(xyz)
    log(f"Writing {output_path} ({num_points:,} points)...")
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', num_points))
        for i in range(num_points):
            f.write(struct.pack('ddd', float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))
            f.write(struct.pack('BBB', 128, 128, 128))  # gray
            f.write(struct.pack('d', 1.0))  # error
            f.write(struct.pack('Q', 0))    # track_length=0
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    log(f"  Done: {size_mb:.1f} MB")


def main():
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    log("Fix: MVS Conversion + Brush Retrain")
    log("=" * 60)
    
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    if not os.path.exists(fused_ply):
        log(f"ERROR: {fused_ply} not found!")
        return
    
    log(f"Fused PLY: {os.path.getsize(fused_ply)/(1024*1024):.1f} MB")
    
    # Step 1: Read PLY
    xyz = read_ply_xyz(fused_ply)
    
    # Step 2: Filter
    xyz = filter_points(xyz)
    if len(xyz) == 0:
        log("ERROR: All points filtered out!")
        return
    
    # Step 3: Subsample if too many (>3M points may slow Brush significantly)
    MAX_POINTS = 3000000
    if len(xyz) > MAX_POINTS:
        log(f"  Subsampling {len(xyz):,} → {MAX_POINTS:,} points")
        indices = np.random.RandomState(42).choice(len(xyz), MAX_POINTS, replace=False)
        xyz = xyz[indices]
    
    # Step 4: Write points3D.bin
    pts_bin = os.path.join(DENSE_DIR, 'points3D_from_mvs.bin')
    write_points3d_bin(xyz, pts_bin)
    
    # Step 5: Swap and train
    orig_pts = os.path.join(SPARSE_ORIG, 'points3D.bin')
    backup_pts = os.path.join(SPARSE_ORIG, 'points3D_backup.bin')
    if not os.path.exists(backup_pts):
        shutil.copy2(orig_pts, backup_pts)
    
    shutil.copy2(pts_bin, orig_pts)
    log(f"\nSwapped sparse/0 with MVS ({len(xyz):,} points)")
    
    name = "mvs_dense_1920_30k_v2"
    output_ply = os.path.join(OUTPUT_DIR, f"{name}.ply")
    
    if os.path.exists(output_ply):
        size_mb = os.path.getsize(output_ply) / (1024*1024)
        log(f"[PASS] {name} exists ({size_mb:.1f} MB)")
    else:
        log(f"\n{'=' * 60}")
        log(f"[RUN] {name} — 30K steps, 1920 res, {len(xyz):,} MVS seed points")
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
            size_mb = os.path.getsize(output_ply) / (1024*1024)
            log(f"\n[DONE] {name}: {size_mb:.1f} MB, {elapsed:.0f}s")
        else:
            log(f"\n[FAIL] exit code {process.returncode}")
    
    # Restore
    if os.path.exists(backup_pts):
        shutil.copy2(backup_pts, orig_pts)
        log("Restored original points3D.bin")
    
    # Summary
    log("\n" + "=" * 60)
    log("FINAL COMPARISON")
    log("=" * 60)
    
    results = {
        "original_high_1280": 301.8,
        "baseline_1920_30k": 366.6,
        "enhanced_1920_30k (170K pts)": 286.1,
        "mvs_dense_1920_30k_v2 (MVS)": None,
    }
    
    if os.path.exists(output_ply):
        results["mvs_dense_1920_30k_v2 (MVS)"] = os.path.getsize(output_ply) / (1024*1024)
    
    for name, size in results.items():
        if size:
            log(f"  {name}: {size:.1f} MB")
        else:
            log(f"  {name}: N/A")
    log("=" * 60)


if __name__ == "__main__":
    main()
