"""
Enhanced quality test: COLMAP global_mapper (100K features) + Brush at 1920.
"""
import os
import sys
import subprocess
import time
import struct
import shutil

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TRUCK_DIR = r"F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck"
BRUSH_PATH = r"F:\Codebase\images_to_play\Brush\brush_app.exe"
COLMAP_PATH = r"C:\COLMAP\bin\colmap.exe"

IMAGE_PATH = os.path.join(TRUCK_DIR, "images")
SPARSE_ORIG = os.path.join(TRUCK_DIR, "sparse", "0")
SPARSE_BACKUP = os.path.join(TRUCK_DIR, "sparse_original", "0")
SPARSE_100K = os.path.join(TRUCK_DIR, "sparse_100k", "0")
OUTPUT_DIR = os.path.join(TRUCK_DIR, "enhanced_tests")

os.makedirs(SPARSE_BACKUP, exist_ok=True)
os.makedirs(SPARSE_100K, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(TRUCK_DIR, "..", "..", "..", "..", "enhanced_log.txt")
LOG_FILE = os.path.normpath(LOG_FILE)

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

def run_colmap_100k():
    pts = count_points(os.path.join(SPARSE_100K, 'points3D.bin'))
    if pts > 500000:
        log(f"Already have {pts:,} points in sparse_100k — skipping COLMAP")
        return True

    log("=" * 60)
    log("PHASE 1: COLMAP global_mapper with 100K features")
    log("=" * 60)

    db_path = os.path.join(TRUCK_DIR, "database_100k.db")
    if os.path.exists(db_path):
        for attempt in range(3):
            try:
                os.remove(db_path)
                log(f"  Removed old database")
                break
            except PermissionError:
                log(f"  Database locked, waiting... (attempt {attempt+1}/3)")
                time.sleep(5)

    sparse_100k_parent = os.path.join(TRUCK_DIR, "sparse_100k")
    for f in os.listdir(SPARSE_100K):
        os.remove(os.path.join(SPARSE_100K, f))

    steps = [
        ("Feature Extraction", (
            f'"{COLMAP_PATH}" feature_extractor '
            f'--database_path "{db_path}" '
            f'--image_path "{IMAGE_PATH}" '
            f'--SiftExtraction.max_num_features 100000 '
            f'--FeatureExtraction.use_gpu 1 '
            f'--SiftExtraction.first_octave -1 '
            f'--SiftExtraction.peak_threshold 0.0005 '
            f'--SiftExtraction.num_octaves 7 '
            f'--SiftExtraction.edge_threshold 5 '
            f'--SiftExtraction.domain_size_pooling 1 '
            f'--SiftExtraction.estimate_affine_shape 1'
        )),
        ("Exhaustive Matching", (
            f'"{COLMAP_PATH}" exhaustive_matcher '
            f'--database_path "{db_path}" '
            f'--FeatureMatching.use_gpu 1 '
            f'--FeatureMatching.max_num_matches 32768'
        )),
        ("Global Mapper (fast SfM)", (
            f'"{COLMAP_PATH}" global_mapper '
            f'--database_path "{db_path}" '
            f'--image_path "{IMAGE_PATH}" '
            f'--output_path "{sparse_100k_parent}" '
            f'--GlobalMapper.track_min_num_views_per_track 2'
        )),
    ]

    for name, cmd in steps:
        log(f"\n--- {name} ---")
        start = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - start
        log(f"  Time: {elapsed:.1f}s, Exit: {result.returncode}")
        if result.returncode != 0:
            log(f"  STDERR: {result.stderr[:500]}")
            if "global_mapper" in name.lower():
                log("  Global mapper failed, falling back to incremental mapper...")
                return run_colmap_incremental()
            return False
        for line in result.stdout.strip().split('\n')[-3:]:
            if line.strip():
                log(f"  {line.strip()}")

    pts = count_points(os.path.join(SPARSE_100K, 'points3D.bin'))
    log(f"\n✓ COLMAP done: {pts:,} sparse points (100K features)")
    return pts > 0


def run_colmap_incremental():
    """Fallback: incremental mapper"""
    log("Running incremental mapper as fallback...")
    sparse_100k_parent = os.path.join(TRUCK_DIR, "sparse_100k")
    db_path = os.path.join(TRUCK_DIR, "database_100k.db")
    
    mapper_cmd = (
        f'"{COLMAP_PATH}" mapper '
        f'--database_path "{db_path}" '
        f'--image_path "{IMAGE_PATH}" '
        f'--output_path "{sparse_100k_parent}" '
        f'--Mapper.filter_min_tri_angle 0.5 '
        f'--Mapper.filter_max_reproj_error 4'
    )
    
    log(f"\n--- Incremental Mapper ---")
    start = time.time()
    result = subprocess.run(mapper_cmd, shell=True, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start
    log(f"  Time: {elapsed:.1f}s, Exit: {result.returncode}")
    
    pts = count_points(os.path.join(SPARSE_100K, 'points3D.bin'))
    log(f"  Points: {pts:,}")
    return pts > 0


def swap_sparse(source_dir):
    for fname in ['cameras.bin', 'images.bin', 'points3D.bin', 'project.ini', 'frames.bin', 'rigs.bin']:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(SPARSE_ORIG, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

def backup_original():
    if os.path.exists(os.path.join(SPARSE_BACKUP, 'points3D.bin')):
        log("Backup already exists")
        return
    for fname in ['cameras.bin', 'images.bin', 'points3D.bin', 'project.ini', 'frames.bin', 'rigs.bin']:
        src = os.path.join(SPARSE_ORIG, fname)
        dst = os.path.join(SPARSE_BACKUP, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

def restore_original():
    log("Restoring original sparse/0...")
    for fname in ['cameras.bin', 'images.bin', 'points3D.bin', 'project.ini', 'frames.bin', 'rigs.bin']:
        src = os.path.join(SPARSE_BACKUP, fname)
        dst = os.path.join(SPARSE_ORIG, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

def train_brush(name, steps, max_res):
    output_ply = os.path.join(OUTPUT_DIR, f"{name}.ply")
    if os.path.exists(output_ply):
        size_mb = os.path.getsize(output_ply) / (1024 * 1024)
        log(f"\n[PASS] {name} already exists ({size_mb:.1f} MB) — skipping")
        return {"status": "skipped", "size_mb": size_mb}

    sparse_pts = count_points(os.path.join(SPARSE_ORIG, 'points3D.bin'))
    log(f"\n{'=' * 60}")
    log(f"[RUN]  {name} — {steps:,} steps, max-res {max_res}")
    log(f"  Sparse points: {sparse_pts:,}")
    log(f"{'=' * 60}")

    cmd = [
        BRUSH_PATH, TRUCK_DIR,
        "--total-steps", str(steps),
        "--export-path", OUTPUT_DIR,
        "--export-name", f"{name}.ply",
        "--export-every", str(steps),
        "--max-resolution", str(max_res),
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
            return {"status": "ok", "size_mb": size_mb, "time": elapsed}
        else:
            log(f"\n[FAIL] {name}: exit code {process.returncode}")
            return {"status": "failed"}
    except Exception as e:
        log(f"\n[ERR] {name}: {e}")
        return {"status": "error"}


def main():
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    log("Enhanced Quality Test: 100K Features + 1920 Resolution")
    log("=" * 60)

    orig_pts = count_points(os.path.join(SPARSE_ORIG, 'points3D.bin'))
    log(f"Original sparse points: {orig_pts:,}")

    backup_original()

    if not run_colmap_100k():
        log("ERROR: COLMAP failed!")
        return

    pts_100k = count_points(os.path.join(SPARSE_100K, 'points3D.bin'))
    log(f"\nSparse: {orig_pts:,} → {pts_100k:,} ({pts_100k/orig_pts:.1f}x)")

    # Train with enhanced sparse
    log("\nSwapping sparse/0 with 100K version...")
    swap_sparse(SPARSE_100K)

    results = {}
    results["enhanced_1920_30k"] = train_brush("enhanced_1920_30k", 30000, 1920)

    # Restore and train baseline
    restore_original()
    results["baseline_1920_30k"] = train_brush("baseline_1920_30k", 30000, 1920)

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"  Original sparse: {orig_pts:,} points")
    log(f"  Enhanced sparse: {pts_100k:,} points ({pts_100k/orig_pts:.1f}x)")
    for name, r in results.items():
        if r["status"] == "ok":
            log(f"  {name}: {r['size_mb']:.1f} MB, {r['time']:.0f}s")
        elif r["status"] == "skipped":
            log(f"  {name}: {r['size_mb']:.1f} MB (skipped)")
        else:
            log(f"  {name}: {r['status']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
