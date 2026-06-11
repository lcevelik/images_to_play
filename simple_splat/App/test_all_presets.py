"""
Test runner: submits all presets to the Flask API sequentially,
waits for completion, copies PLY to named result folders.

Results are viewable at:
  http://localhost:5000/static/supersplat/index.html?load=/ply/result_<preset>
"""

import os
import sys
import time
import shutil
import requests

BASE_URL = "http://localhost:5000"
IMAGE_DIR = r"C:\Users\LuxMC\Downloads\iCloud Photos\iCloud Photos"
PROCESSING_DIR = os.path.join(os.path.dirname(__file__), "processing")

# All test configurations: (label, preset, trainer, quality_scale)
TESTS = [
    # low_brush already done manually — PLY copied to result_low_brush/
    ("medium_brush",  "medium", "brush", "standard"),
    ("high_brush",    "high",   "brush", "standard"),
    ("quality_brush", "quality","brush", "standard"),
    ("expert_brush",  "expert", "brush", "standard"),
    ("medium_mcmc",   "medium", "mcmc",  "standard"),
]

POLL_INTERVAL = 15  # seconds between status checks


def upload_and_start(label, preset, trainer, quality_scale):
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}  (preset={preset}, trainer={trainer})")
    print(f"{'='*60}")

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"  Uploading {len(image_files)} images...")

    data = {
        "method":        "traditional",
        "preset":        preset,
        "matcher_type":  "auto",
        "interval":      "1",
        "quality_scale": quality_scale,
        "trainer":       trainer,
        "enable_dense":  "false",
        "mvs_quality_mode": "balanced",
    }

    # Retry upload until a slot opens (handles 429 from concurrent job limit)
    for attempt in range(30):
        files = [("files", (f, open(os.path.join(IMAGE_DIR, f), "rb"), "image/jpeg"))
                 for f in image_files]
        try:
            resp = requests.post(f"{BASE_URL}/upload", files=files, data=data, timeout=120)
        except Exception as e:
            for _, (_, fh, _) in files:
                fh.close()
            return None, f"Upload failed: {e}"
        for _, (_, fh, _) in files:
            fh.close()

        if resp.status_code == 429:
            active = resp.json().get("active_jobs", "?")
            print(f"  Server busy ({active} active jobs), waiting 30s... (attempt {attempt+1}/30)")
            time.sleep(30)
            continue

        if resp.status_code != 200:
            return None, f"Upload HTTP {resp.status_code}: {resp.text[:200]}"

        job_id = resp.json().get("job_id")
        if not job_id:
            return None, f"No job_id in response: {resp.text[:200]}"

        print(f"  Job ID: {job_id}")
        return job_id, None

    return None, "Timed out waiting for a job slot after 15 minutes"


def wait_for_job(job_id, label):
    start = time.time()
    last_step = ""

    while True:
        try:
            resp = requests.get(f"{BASE_URL}/status/{job_id}", timeout=30)
            status = resp.json()
        except Exception as e:
            print(f"  [poll error] {e}")
            time.sleep(POLL_INTERVAL)
            continue

        stage = status.get("stage", "")
        step = status.get("step", "")
        progress = status.get("progress", 0)
        elapsed = time.time() - start

        display = step if step else stage
        if display != last_step:
            print(f"  [{elapsed:>5.0f}s | {progress:>3}%] {display}")
            last_step = display

        if status.get("status") in ("completed", "complete", "error", "failed"):
            return status, time.time() - start

        time.sleep(POLL_INTERVAL)


def copy_result(job_id, label):
    src_ply = os.path.join(PROCESSING_DIR, job_id, "gaussian_splat.ply")
    fallback = os.path.join(PROCESSING_DIR, job_id, "point_cloud.ply")
    result_dir = os.path.join(PROCESSING_DIR, f"result_{label}")
    os.makedirs(result_dir, exist_ok=True)

    if os.path.exists(src_ply):
        size_mb = os.path.getsize(src_ply) / 1024 / 1024
        shutil.copy2(src_ply, os.path.join(result_dir, "gaussian_splat.ply"))
        return "gaussian_splat.ply", size_mb
    elif os.path.exists(fallback):
        size_mb = os.path.getsize(fallback) / 1024 / 1024
        shutil.copy2(fallback, os.path.join(result_dir, "gaussian_splat.ply"))
        return "point_cloud.ply (fallback)", size_mb
    else:
        return "MISSING", 0


def main():
    print(f"\nTest runner starting. {len(TESTS)} configurations to test.")
    print(f"Images: {IMAGE_DIR}")
    print(f"Results: {PROCESSING_DIR}/result_<label>/gaussian_splat.ply")
    print(f"\nViewer URLs (open after each completes):")
    for label, *_ in TESTS:
        print(f"  http://localhost:5000/static/supersplat/index.html?load=/ply/result_{label}")

    summary = []
    total_start = time.time()

    for label, preset, trainer, quality_scale in TESTS:
        job_id, err = upload_and_start(label, preset, trainer, quality_scale)
        if err:
            summary.append((label, "UPLOAD_FAILED", 0, 0, err))
            print(f"  ERROR: {err}")
            continue

        status, elapsed = wait_for_job(job_id, label)
        final_status = status.get("status", "unknown")
        ply_generated = status.get("ply_generated", False)

        src_label, size_mb = copy_result(job_id, label)

        viewer_url = f"http://localhost:5000/static/supersplat/index.html?load=/ply/result_{label}"
        summary.append((label, final_status, elapsed, size_mb, src_label, viewer_url))

        mins = elapsed / 60
        print(f"\n  DONE: {label} | status={final_status} | {mins:.1f}min | {size_mb:.1f}MB | output={src_label}")

    # Final summary
    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  (total: {total_elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"{'Label':<20} {'Status':<12} {'Time':>7} {'Size':>8}  Output")
    print(f"{'-'*70}")
    for row in summary:
        if len(row) == 5:
            label, st, elapsed, size_mb, note = row
            url = ""
        else:
            label, st, elapsed, size_mb, note, url = row
        mins = elapsed / 60 if elapsed else 0
        flag = "OK" if st in ("complete", "completed") and "fallback" not in note and "MISSING" not in note else "WARN" if st in ("complete", "completed") else "FAIL"
        print(f"{label:<20} {st:<12} {mins:>6.1f}m {size_mb:>7.1f}MB  [{flag}] {note}")

    print(f"\nViewer URLs:")
    for row in summary:
        if len(row) == 6:
            label, st, *_, url = row
            flag = "OK " if st in ("complete", "completed") else "ERR"
            print(f"  [{flag}] {url}")

    print("\nDone.")


if __name__ == "__main__":
    main()
