# Python Modules — Architecture & Reference

This document explains what each `.py` file does, what functions it exposes, and how all the modules interact during a processing job.

---

## Module Overview

```
app.py                  <- Web server + job orchestrator (entry point)
  |
  +-- run_glomap.py     <- COLMAP/GLOMAP pipeline (feature extraction, matching, reconstruction, MVS)
  |
  +-- dense_reconstruction.py  <- Standalone dense MVS fallback (used only if run_glomap skips MVS)
  |
  +-- gaussian_splat_utils.py  <- PLY fallback (used only if Brush training fails)
```

`app.py` is the only module that imports from the others. The three helper modules do not import from each other.

---

## `app.py` — Web Server & Job Orchestrator

**Role:** Entry point. Runs the Flask web server, handles HTTP requests, manages the full job lifecycle from upload to download.

### Startup

When `START_SERVER.bat` runs `python app.py`:
1. Checks if ML-Sharp (`sharp`) is available in PATH
2. Starts a background cleanup thread (deletes old job folders every 24 hours)
3. Starts Flask on `0.0.0.0:5000` (accessible from any network interface)

### Job Lifecycle

Every uploaded job goes through the same sequence:

```
/upload (POST)
  -> Saves files to processing/<uuid>/images/
  -> Starts a background daemon thread
     -> process_images_async()    (traditional pipeline)
     -> process_mlsharp_async()   (ML-Sharp single-image)
  -> Returns { job_id } immediately

/status/<job_id> (GET)    <- frontend polls this every 2 seconds
  -> Returns { status, progress, step, error, logs }

/ply/<job_id> (GET)       <- SuperSplat viewer calls this to load the splat
  -> Serves the .ply file as binary stream

/download/<job_id>/ply    <- user-initiated download
/download/<job_id>/sparse <- downloads sparse reconstruction as .zip
```

### `process_images_async(job_id, image_path, preset, matcher_type, interval, advanced_settings)`

The main processing function, runs in a background thread. Orchestrates the full pipeline:

```
Step 1 → run_colmap()           (from run_glomap.py)
          Sparse reconstruction + optional MVS inside run_glomap

Step 2 → run_dense_reconstruction()   (from dense_reconstruction.py)
          ONLY called when:
          - preset is 'low' (no MVS in run_glomap)
          - AND enable_dense=True was requested via advanced settings
          Skipped when run_glomap already ran MVS (medium/high/sharpness presets)

Step 3 → subprocess: brush_app.exe    (external binary)
          Gaussian splat training on the COLMAP output
          Exports gaussian_splat.ply

Step 4 → generate_ply_from_colmap()   (from gaussian_splat_utils.py)
          ONLY called as fallback if Brush fails or is not found
          Exports basic point cloud from sparse reconstruction
```

Progress and logs from all steps are collected into `processing_status[job_id]` and streamed to the browser via Server-Sent Events (`/logs/stream`).

### `process_mlsharp_async(job_id, image_path, render_views, device)`

Alternative processing path for single-image ML-Sharp jobs. Does not use COLMAP or Brush — calls `sharp predict` directly as a subprocess and moves the output `.ply` to the standard location.

### Key Global State

| Variable | Type | Purpose |
|----------|------|---------|
| `processing_status` | dict | Job status, progress, step, logs, output paths |
| `job_start_times` | dict | Job creation timestamps (for retention cleanup) |
| `log_buffer` | deque(500) | Global rolling log (last 500 entries across all jobs) |
| `mlsharp_available` | bool | Set at startup, controls ML-Sharp UI |
| `MAX_CONCURRENT_JOBS` | int (3) | Limits simultaneous traditional pipeline jobs |

### Thread Safety

- All log writes go through `log_lock` (threading.Lock)
- Progress callbacks use `threading.local()` so each job thread has its own callback reference
- `set_current_job(job_id)` / `get_current_job()` use thread-local storage so `add_log()` routes messages to the right job

### HTTP Routes Summary

| Route | Method | What it does |
|-------|--------|-------------|
| `/` | GET | Renders `templates/index.html` |
| `/upload` | POST | Accepts files, creates job, starts processing thread |
| `/status/<job_id>` | GET | Returns job status dict as JSON |
| `/ply/<job_id>` | GET | Streams PLY file binary (for viewer) |
| `/download/<job_id>/<type>` | GET | Downloads PLY or sparse ZIP |
| `/view/<job_id>` | GET | Redirects to SuperSplat viewer URL |
| `/upload-for-view` | POST | Accepts .ply/.splat, serves it via viewer |
| `/logs` | GET | Renders `templates/logs.html` |
| `/logs/stream` | GET | SSE stream of log entries |
| `/logs/current` | GET | Current log buffer as JSON |
| `/logs/json/<job_id>` | GET | Per-job log as JSON array |
| `/gpu-info` | GET | Detects NVIDIA GPU via nvidia-smi |
| `/mlsharp-info` | GET | ML-Sharp availability and version |
| `/kill` | POST | Kills COLMAP/Brush processes via psutil |
| `/cleanup` | POST | Deletes all job folders |
| `/cleanup-old` | POST | Deletes folders older than 24h |

---

## `run_glomap.py` — COLMAP / GLOMAP Pipeline

**Role:** Runs the complete Structure-from-Motion reconstruction pipeline by calling COLMAP (and optionally GLOMAP) as external processes. Also handles optional dense MVS reconstruction inline for medium/high/sharpness presets.

### Main Function

```python
run_colmap(image_path, matcher_type, interval, model_type,
           detail_level='medium', quality_mode=False, ultra_sharpness_mode=False)
```

Internally runs these COLMAP stages in order:

| Stage | COLMAP Command | What it does |
|-------|----------------|--------------|
| 1 | `feature_extractor` | Detects SIFT keypoints in every image (GPU) |
| 2 | `exhaustive_matcher` or `sequential_matcher` | Matches features between image pairs (GPU) |
| 3 | `mapper` (COLMAP) or `glomap mapper` | Builds 3D sparse point cloud, estimates camera poses |
| 4 | `image_undistorter` | Removes lens distortion, prepares images for Brush |
| 5 | `patch_match_stereo` | Computes per-image depth maps (MVS, GPU) — only for dense presets |
| 6 | `stereo_fusion` | Fuses depth maps into a dense point cloud (`dense/fused.ply`) — only for dense presets |

### Preset Settings

The `detail_settings` dict maps preset names to COLMAP parameters:

| Parameter | What it controls |
|-----------|-----------------|
| `features` | Max SIFT keypoints per image (0 = unlimited) |
| `peak` | SIFT peak threshold — lower keeps more keypoints |
| `octaves` | Number of scale-space octaves for SIFT |
| `match_ratio` | Lowe's ratio test threshold — higher keeps more matches |
| `tri_angle` | Minimum triangulation angle — lower keeps more 3D points |
| `reproj_error` | Max reprojection error — higher keeps more 3D points |
| `dense` | Whether to run MVS stages (5 & 6) above |
| `mvs_window_radius` | Patch match window size (larger = more context) |
| `mvs_iterations` | Patch match iterations (more = better convergence) |
| `mvs_samples` | Patch match random samples (more = better accuracy) |

The three user-visible presets map to:

| Preset | `features` | `dense` | MVS window | Approx. points |
|--------|-----------|---------|------------|---------------|
| `low` | 16384 | No | — | 50K–200K |
| `medium` | 32768 | Yes | 5px | 500K–2M |
| `high` | unlimited | Yes (quality) | 7px | 5M–50M+ |

Legacy presets (`ultra`, `extreme`, `maximum`, `insane`, `unlimited`, `dense`, `expert`, `sharpness`) are kept for backward compatibility with advanced settings.

### GLOMAP vs COLMAP Mapper

At startup the function probes for `glomap.exe`. If found, it replaces the COLMAP `mapper` step with GLOMAP, which uses global rotation averaging instead of incremental registration — usually faster and equally accurate. If GLOMAP is not found, it falls back to COLMAP mapper transparently.

GLOMAP outputs to `distorted/sparse/0/`. The function copies those files to `sparse/0/` afterward so the rest of the pipeline sees a consistent layout.

### Progress Reporting

COLMAP outputs are captured line-by-line via `subprocess.Popen` + `readline()`. Regular expressions parse the output for:
- Feature extraction: `Processing image X / Y [name: ...]`
- Matching: `Matching X / Y`
- Registration: `Registering image #N`
- Bundle adjustment: `Bundle adjustment iteration N`
- Triangulation: `Triangulated N points`
- MVS: `Processing view X/Y`

Parsed events are forwarded through a thread-local progress callback set by `app.py` → routed to `add_log()` → stored in the job's log buffer → streamed to the browser.

### Helper Functions

| Function | Purpose |
|----------|---------|
| `set_progress_callback(cb)` | Register a log callback for the current thread |
| `get_progress_callback()` | Retrieve the current thread's callback |
| `log_progress(msg, level)` | Call callback or print |
| `rename_image_folder_if_needed(path)` | Renames `input/` or `images/` → `source/` before processing |
| `filter_images(path, interval)` | Copies every Nth image to `input/` for sub-sampling |

### Standalone CLI Usage

```bash
python run_glomap.py \
  --image_path processing/<uuid>/images \
  --matcher_type exhaustive_matcher \
  --detail_level medium \
  --model_type 3dgs
```

---

## `dense_reconstruction.py` — Standalone Dense MVS

**Role:** A self-contained dense reconstruction module. Used by `app.py` as a **fallback** when `run_glomap.py` did not run MVS (e.g., the `low` preset was selected but the user enabled dense via advanced settings).

> **Important:** For `medium`, `high`, and `sharpness` presets, `run_glomap.py` handles MVS internally. `app.py` explicitly skips calling this module in those cases to avoid double-processing and workspace conflicts.

### Main Function

```python
run_dense_reconstruction(parent_dir, image_path, sparse_path,
                         enable_dense=True, max_image_size=3200,
                         quality_mode=False, ultra_sharpness_mode=False,
                         colmap_path='colmap')
```

Runs three COLMAP stages sequentially:

1. **`image_undistorter`** — undistorts images and writes them to `dense/images/`, produces `dense/sparse/` with camera models
2. **`patch_match_stereo`** — computes per-view depth maps in `dense/stereo/depth_maps/`
3. **`stereo_fusion`** — fuses depth maps into `dense/fused.ply`

Returns the path to `fused.ply` on success, or `None` on failure.

### Quality Modes

| Mode | Window | Samples | Iterations | Fusion min_pixels |
|------|--------|---------|------------|-------------------|
| Balanced | 5 | 15 | 5 | 5 |
| Quality | 7 | 25 | 10 | 3 |
| Ultra Sharpness | 11 | 40 | 20 | 2 |

### Other Functions

| Function | Purpose |
|----------|---------|
| `run_poisson_reconstruction(dense_ply, output, depth)` | Optional: turns dense point cloud into a mesh via `colmap poisson_mesher`. Not called in the current pipeline but available for standalone use. |

### Standalone CLI Usage

```bash
python dense_reconstruction.py \
  --parent_dir processing/<uuid> \
  --image_path processing/<uuid>/images \
  --max_image_size 3200
```

---

## `gaussian_splat_utils.py` — PLY Fallback Utilities

**Role:** Last-resort fallback. If Brush training fails or Brush is not found, `app.py` calls `generate_ply_from_colmap()` to produce a basic point cloud PLY from the COLMAP sparse reconstruction. This is not a trained Gaussian splat — it is a simple XYZ+RGB point cloud.

### Functions

#### `generate_ply_from_colmap(colmap_path, output_ply_path, center_at_origin=True)`

Reads the COLMAP sparse reconstruction using `pycolmap`, extracts all 3D points and their colors, optionally centers the point cloud at the origin, and writes a PLY file.

- Returns `True` on success, `False` on failure
- Requires `pycolmap` and `numpy` (both bundled)
- Output is ASCII PLY format with `x y z r g b` per vertex

#### `write_ply_file(output_path, points, colors=None)`

Low-level PLY writer. Writes an ASCII PLY file from a list of `[x, y, z]` points and optional `[r, g, b]` colors (0–255). Called internally by `generate_ply_from_colmap`.

#### Other Functions (not used in the current pipeline)

| Function | Status | Purpose |
|----------|--------|---------|
| `prepare_for_gaussian_splatting(colmap_path, images, output)` | Unused | Copies COLMAP output into expected 3DGS folder layout |
| `check_gaussian_splatting_available()` | Unused | Checks for `gsplat` / `diff_gaussian_rasterization` |
| `get_training_command(data, output)` | Unused | Returns a template training command string |

---

## How the Modules Work Together — Full Processing Flow

```
Browser: POST /upload
         |
         v
app.py: Creates job folder   processing/<uuid>/images/
        Saves uploaded files
        Starts daemon thread → process_images_async()

─────────────────────────────────────────────────────────────────────
Thread: process_images_async()
─────────────────────────────────────────────────────────────────────

  [1] Calls run_glomap.run_colmap()
      │
      ├─ COLMAP feature_extractor  (GPU SIFT detection)
      ├─ COLMAP exhaustive_matcher (GPU feature matching)
      ├─ COLMAP/GLOMAP mapper      (3D reconstruction → sparse/0/)
      ├─ COLMAP image_undistorter  (→ images/ + sparse/ in parent_dir)
      │
      └─ If preset is medium/high/sharpness:
         ├─ COLMAP patch_match_stereo  (GPU depth maps → dense/stereo/)
         └─ COLMAP stereo_fusion       (fused point cloud → dense/fused.ply)

  [2] If preset is low AND enable_dense=True:
      Calls dense_reconstruction.run_dense_reconstruction()
      │
      ├─ COLMAP image_undistorter
      ├─ COLMAP patch_match_stereo
      └─ COLMAP stereo_fusion → dense/fused.ply

  [3] Calls Brush as subprocess:
      brush_app.exe <parent_dir> --total-steps N --export-path <parent_dir>
      │
      └─ Trains Gaussian splat → gaussian_splat.ply

  [4] FALLBACK — only if step 3 fails or Brush not found:
      Calls gaussian_splat_utils.generate_ply_from_colmap()
      │
      └─ Reads sparse/0/ via pycolmap → point_cloud.ply

─────────────────────────────────────────────────────────────────────
Result:
  processing/<uuid>/gaussian_splat.ply   <- trained splat (Brush)
  processing/<uuid>/point_cloud.ply      <- fallback point cloud
  processing/<uuid>/dense/fused.ply      <- dense MVS cloud (if enabled)
  processing/<uuid>/sparse/0/            <- COLMAP cameras + points

Browser: GET /ply/<uuid>   <- SuperSplat viewer fetches this
         Returns gaussian_splat.ply (or point_cloud.ply as fallback)
```

### Progress / Logging Flow

```
run_glomap.log_progress()
    │
    └─► thread-local callback (set by app.py via set_progress_callback)
            │
            └─► app.add_log(message, level)
                    │
                    ├─► log_buffer (global deque, last 500 entries)
                    └─► processing_status[job_id]['logs'] (per-job deque)
                                │
                                └─► GET /logs/stream (SSE)
                                    GET /logs/json/<job_id>
                                    GET /status/<job_id>
                                        │
                                        └─► Browser live log panel
```

---

## Output Folder Structure (per job)

After processing completes, a job folder looks like this:

```
processing/<uuid>/
├── images/                    <- Undistorted images (prepared by COLMAP for Brush)
├── source/                    <- Original uploaded images (renamed from images/ by run_glomap)
├── sparse/
│   └── 0/
│       ├── cameras.bin        <- Camera intrinsics and poses
│       ├── images.bin         <- Per-image camera registration
│       └── points3D.bin       <- Sparse 3D point cloud
├── dense/
│   ├── images/                <- Undistorted images for MVS
│   ├── stereo/
│   │   └── depth_maps/        <- Per-image depth maps (.geometric.bin)
│   └── fused.ply              <- Dense point cloud (millions of points)
├── gaussian_splat.ply         <- Final trained Gaussian splat (Brush output)
├── point_cloud.ply            <- Fallback sparse PLY (gaussian_splat_utils)
├── point_cloud_dense.ply      <- Copy of dense/fused.ply (if created)
└── colmap_run.log             <- Full COLMAP command log
```
