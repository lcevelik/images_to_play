"""
Fix v2: Convert MVS fused.ply with proper subsampling (500K max).
Brush can't handle 3M initial points (capacity overflow).
"""
import os, sys, subprocess, time, struct, shutil
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TRUCK_DIR = r"F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck"
BRUSH_PATH = r"F:\Codebase\images_to_play\Brush\brush_app.exe"
SPARSE_ORIG = os.path.join(TRUCK_DIR, "sparse", "0")
DENSE_DIR = os.path.join(TRUCK_DIR, "dense_mvs")
OUTPUT_DIR = os.path.join(TRUCK_DIR, "mvs_tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = r"F:\Codebase\images_to_play\mvs_log.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def main():
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    log("Fix v2: MVS Conversion (500K cap) + Brush Retrain")
    log("=" * 60)
    
    fused_ply = os.path.join(DENSE_DIR, 'fused.ply')
    
    # Read PLY
    with open(fused_ply, 'rb') as f:
        num_vertices = 0
        props = []
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        
        bytes_per_vertex = len(props) * 4
        data = f.read(num_vertices * bytes_per_vertex)
        all_data = np.frombuffer(data, dtype=np.float32).reshape(-1, len(props))
        xyz_idx = [props.index(p) for p in ['x', 'y', 'z']]
        xyz = all_data[:, xyz_idx].copy()
    
    log(f"Read {len(xyz):,} points")
    
    # Remove NaN/Inf only (no outlier filter — coordinates are huge)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    log(f"After NaN filter: {len(xyz):,}")
    
    # The MVS coordinates are in COLMAP's coordinate space (possibly huge values)
    # Just subsample randomly — outlier filtering doesn't work at this scale
    MAX_POINTS = 500000
    if len(xyz) > MAX_POINTS:
        indices = np.random.RandomState(42).choice(len(xyz), MAX_POINTS, replace=False)
        xyz = xyz[indices]
        log(f"Subsampled to {len(xyz):,}")
    
    # Write points3D.bin
    pts_bin = os.path.join(DENSE_DIR, 'points3D_from_mvs.bin')
    with open(pts_bin, 'wb') as f:
        f.write(struct.pack('Q', len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack('ddd', float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))
            f.write(struct.pack('BBB', 128, 128, 128))
            f.write(struct.pack('d', 1.0))
            f.write(struct.pack('Q', 0))
    log(f"Wrote {pts_bin}: {os.path.getsize(pts_bin)/(1024*1024):.1f} MB")
    
    # Swap
    orig_pts = os.path.join(SPARSE_ORIG, 'points3D.bin')
    backup_pts = os.path.join(SPARSE_ORIG, 'points3D_backup.bin')
    if not os.path.exists(backup_pts):
        shutil.copy2(orig_pts, backup_pts)
    shutil.copy2(pts_bin, orig_pts)
    
    # Train
    name = "mvs_500k_1920_30k"
    output_ply = os.path.join(OUTPUT_DIR, f"{name}.ply")
    
    if os.path.exists(output_ply):
        log(f"[PASS] {name} exists ({os.path.getsize(output_ply)/(1024*1024):.1f} MB)")
    else:
        log(f"\n[RUN] {name}")
        cmd = [BRUSH_PATH, TRUCK_DIR,
               "--total-steps", "30000",
               "--export-path", OUTPUT_DIR,
               "--export-name", f"{name}.ply",
               "--export-every", "30000",
               "--max-resolution", "1920"]
        
        start = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(p.stdout.readline, ''):
            if line.strip():
                log(f"  Brush: {line.strip()}")
        p.wait()
        elapsed = time.time() - start
        
        if p.returncode == 0 and os.path.exists(output_ply):
            size_mb = os.path.getsize(output_ply) / (1024*1024)
            log(f"\n[DONE] {name}: {size_mb:.1f} MB, {elapsed:.0f}s")
        else:
            log(f"\n[FAIL] exit {p.returncode}")
    
    # Restore
    if os.path.exists(backup_pts):
        shutil.copy2(backup_pts, orig_pts)
        log("Restored original points3D.bin")
    
    # Summary
    log("\n" + "=" * 60)
    log("ALL RESULTS")
    log("=" * 60)
    for test_name, ply_file in [
        ("baseline (136K pts, 1920)", "baseline_1920_30k.ply"),
        ("enhanced (170K pts, 1920)", "enhanced_1920_30k.ply"),
        ("mvs_500k (500K MVS pts, 1920)", "mvs_500k_1920_30k.ply"),
    ]:
        path = os.path.join(OUTPUT_DIR, ply_file)
        if os.path.exists(path):
            log(f"  {test_name}: {os.path.getsize(path)/(1024*1024):.1f} MB")
        else:
            log(f"  {test_name}: N/A")
    log("=" * 60)

if __name__ == "__main__":
    main()
