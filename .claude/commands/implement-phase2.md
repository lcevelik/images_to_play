# implement-phase2

Implement Phase 2 medium-effort improvements. Complete Phase 1 first.

**Read CLAUDE.md "Improvement Roadmap" section before starting.**

## Items

### 1. Quality scaler

File: `simple_splat/App/run_glomap.py`

Replace per-preset hardcoded training step counts with a single quality multiplier (0.3x / 1x / 2x).
Base steps: 30,000. Low=9k, medium=30k, high=60k.
Expose as a dropdown in advanced settings.

### 2. Brush streaming (live step progress)

File: `simple_splat/App/app.py`

In `process_images_async()`, change `subprocess.run(brush_cmd)` to `subprocess.Popen` with `stdout=PIPE`.
Stream output lines, parse step numbers, write to `processing_status[job_id]["brush_step"]`.
Frontend polling already calls `/status/<job_id>` — add `brush_step` to the response.

### 3. Fix Brush image folder

File: `simple_splat/App/app.py` or `run_glomap.py`

Brush must be given `images/` (undistorted, output of COLMAP image_undistorter), not `source/`.
Check the current `brush_cmd` construction — if it passes `source/`, change to `images/`.

### 4. Stage timings

File: `simple_splat/App/app.py`

Record `time.time()` at start/end of each stage.
Store as `processing_status[job_id]["timings"] = {"colmap": seconds, "mvs": seconds, "training": seconds}`.
Include in `/status` response so the frontend can display elapsed time.

## Verify
After implementing streaming: submit a job and confirm step progress appears in the UI without full page reload.
