# images_to_play — Feature Roadmap & Code Review

> Generated: 2026-05-27 | Updated: 2026-05-28
> Status: **Plan only** — no code changes

---

## Implementation Status Audit (2026-05-28)

Cross-referenced the ROADMAP against the actual codebase. Here's what's real:

### Already Done (ROADMAP was wrong or outdated)
| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1.13 | Symlinks instead of copies for interval filter | **DONE** | `run_glomap.py:188` uses `os.link()` with `copy2()` fallback |
| 2.1 | Batch processing module | **CODE EXISTS, NOT WIRED UP** | `batch_processing.py` has full `BatchQueue` + Flask routes, but `register_batch_routes` is imported in `app.py:18` and **never called** |
| — | Test scripts | **DONE** | `test_batch.py` and `test_presets.py` in project root |

### Confirmed Missing / Broken
| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1.1 | Brush stdout streaming | **BROKEN** | `app.py:908-913` — `stdout=subprocess.DEVNULL` |
| 1.2 | COLMAP path caching | **MISSING** | Probed every job in `app.py:588-596` AND `run_glomap.py:271-279` |
| 1.3 | Parallel image resize | **MISSING** | `app.py:28-47` — single-threaded PIL loop |
| 1.4 | Blur filter efficiency | **INEFFICIENT** | `app.py:49-70` — full-res image loads |
| 1.5 | Combined resize+blur pass | **MISSING** | Two separate passes over all images |
| 1.6 | Multi-camera auto-detect | **MISSING** | `run_glomap.py:365` — hardcoded `single_camera 1` |
| 1.7 | gsplat GPU memory | **INEFFICIENT** | `gsplat_mcmc_trainer.py:229-230` — all images on GPU at once |
| 1.8 | CPU KDTree init | **INEFFICIENT** | `gsplat_mcmc_trainer.py:109` — full scipy cKDTree |
| 1.9 | MVS cache_size auto-detection | **MISSING** | `run_glomap.py:751` — hardcoded 44000 (44GB) |
| 1.10 | shell=True in subprocesses | **PRESENT** | `run_glomap.py:606,796,887,981,1025` + `dense_reconstruction.py:68,128,183` |
| 1.11 | Duplicate COLMAP detection | **PRESENT** | Both `app.py:544-601` and `run_glomap.py:240-282` |
| 1.12 | In-memory job status | **PRESENT** | `app.py:107` — dict only, no persistence |
| 1.16 | mlsharpDevice JS bug | **CONFIRMED** | `index.html:1078` references `getElementById('mlsharpDevice')` — element doesn't exist |

### Batch Module Design Issue
`batch_processing.py:286-313` — the `_upload` function calls `http://127.0.0.1:5000/upload` via `requests.post()`, re-uploading files over HTTP to itself. This is:
- Wasteful (double network I/O for local files)
- Fragile (hardcoded localhost:5000)
- Breaks if the server binds to a different port

Should call `process_images_async()` directly instead of going through HTTP.

---

## Table of Contents

1. [Code Review: Speed & Quality Issues](#1-code-review-speed--quality-issues)
2. [Feature Roadmap](#2-feature-roadmap)
3. [Video Camera Tracking & Matchmoving](#3-video-camera-tracking--matchmoving)
4. [Batch Processing System](#4-batch-processing-system)
5. [Competitive Landscape](#5-competitive-landscape)
6. [Priority Matrix](#6-priority-matrix)

---

## 1. Code Review: Speed & Quality Issues

### CRITICAL — Major Performance Bottlenecks

#### 1.1 Brush subprocess has no stdout streaming
**File:** `app.py:908-929`
**Issue:** Brush is launched with `stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL`. The progress loop estimates steps by elapsed time (`elapsed / 0.5`), which is wildly inaccurate. Brush does emit progress to stdout — it's being thrown away.
**Impact:** User sees fake progress; no way to know if Brush is stuck or working.
**Fix:** Pipe stdout, parse actual step numbers from Brush output (same pattern used for gsplat MCMC at line 783).

#### 1.2 COLMAP PATH probing runs `--help` on every job start
**File:** `app.py:588-596` AND `run_glomap.py:271-279`
**Issue:** Every single job runs `subprocess.run([path, "--help"])` for up to 4 candidate paths (10s timeout each). This adds 1-40 seconds of startup overhead per job. COLMAP doesn't move between jobs.
**Impact:** Wastes 1-40s per job for no reason.
**Fix:** Probe once at startup, cache the result in a module-level variable.

#### 1.3 `resize_images_for_colmap` is single-threaded with Pillow
**File:** `app.py:28-47`
**Issue:** Each image is opened, resized, and saved sequentially using PIL. For 100+ images at 4K, this can take 30-60 seconds.
**Impact:** Adds significant delay before COLMAP even starts.
**Fix:** Use `concurrent.futures.ThreadPoolExecutor` to parallelize across CPU cores. PIL releases the GIL during I/O, so threading helps. Or use `pillow-simd` for 2-4x faster resize.

#### 1.4 `filter_blurry_images` loads every image into memory
**File:** `app.py:49-70`
**Issue:** Reads full-resolution images with OpenCV just to compute Laplacian variance. For 100 images at 4K, this loads ~3GB into memory sequentially.
**Impact:** Memory spike, slow startup. And the function only warns — never acts.
**Fix:** Downscale to ~640px before computing blur score (the variance is scale-invariant enough for filtering). Use multiprocessing pool.

#### 1.5 No image caching between resize and blur filter
**File:** `app.py:625-633`
**Issue:** `resize_images_for_colmap` modifies files in-place, then `filter_blurry_images` re-reads them from disk. The images were just in memory.
**Impact:** Double I/O for every image.
**Fix:** Combine into a single pass, or cache the loaded arrays.

#### 1.6 COLMAP feature extraction uses `single_camera 1`
**File:** `run_glomap.py:365`
**Issue:** `--ImageReader.single_camera 1` assumes all images come from the same camera. If the user mixes phone + drone photos, or different phones, the reconstruction fails silently with poor results.
**Impact:** Silent quality degradation for mixed-camera inputs.
**Fix:** Auto-detect by checking EXIF focal lengths; set `single_camera 0` when variance is high.

### HIGH — Quality Issues

#### 1.7 gsplat MCMC trainer loads ALL images into GPU memory at once
**File:** `gsplat_mcmc_trainer.py:229-230`
**Issue:** `c2w_mats = [m.to(device) for m in dataset['c2w_mats']]` and `img_tensors = [img.to(device) for img in dataset['images']]` move everything to GPU upfront. For 200 images at 4K, this is ~12GB of GPU memory just for data, before any Gaussians.
**Impact:** OOM on GPUs with <16GB VRAM. Even on 24GB cards, leaves little room for training.
**Fix:** Keep images on CPU, move only the current batch image to GPU in the training loop.

#### 1.8 gsplat MCMC trainer uses CPU KDTree for initialization
**File:** `gsplat_mcmc_trainer.py:109`
**Issue:** `from scipy.spatial import cKDTree` on potentially millions of sparse points. This is slow and uses lots of RAM.
**Impact:** 5-30 seconds of CPU computation that could be avoided.
**Fix:** Use a fixed initial scale based on the scene bounding box / average nearest-neighbor from a random subsample.

#### 1.9 MVS PatchMatch cache_size is hardcoded to 44GB
**File:** `run_glomap.py:751`
**Issue:** `--PatchMatchStereo.cache_size 44000` (44GB). On systems with 16-32GB RAM, this causes swapping or crashes.
**Impact:** System freeze, OOM, or very slow MVS on mid-range machines.
**Fix:** Detect available RAM and set cache_size to ~70% of available memory.

#### 1.10 COLMAP commands use `shell=True`
**File:** `run_glomap.py:606,796,887,981,1025`, `dense_reconstruction.py:68,128,183`
**Issue:** All COLMAP subprocess calls use `shell=True` with string commands. This is fragile (paths with spaces, special characters), slower (extra shell process), and a security concern.
**Impact:** Potential injection if filenames contain shell metacharacters. Extra process overhead.
**Fix:** Use list-based args without shell=True. COLMAP paths are already resolved — just split the command string into a list.

### MEDIUM — Architecture & Reliability

#### 1.11 Duplicate COLMAP path detection logic
**File:** `app.py:544-601` AND `run_glomap.py:240-282`
**Issue:** Both files independently probe for COLMAP, add paths to `os.environ["PATH"]`, and check for global_mapper. This runs twice per job.
**Fix:** Centralize into a single `get_colmap_path()` utility, called once at startup.

#### 1.12 Job status stored in process memory only
**File:** `app.py:107`
**Issue:** `processing_status = {}` is an in-memory dict. Server restart loses all job state. No persistence, no recovery.
**Impact:** If the Flask process crashes or restarts during a long COLMAP run, the job is orphaned (still running) but the UI shows nothing.
**Fix:** Write status to a SQLite DB or JSON file per job in the processing folder.

#### 1.13 ~~`filter_images` copies files instead of using symlinks~~ — FIXED
**File:** `run_glomap.py:184-190`
**Status:** Already uses `os.link()` (hardlink) with `copy2()` fallback for cross-volume. No action needed.

#### 1.14 No cleanup of intermediate files
**Issue:** Each job creates: source/, images/, sparse/, dense/, distorted/, input/, plus the original upload. A single job can use 5-15GB. The cleanup thread only runs every 24 hours.
**Fix:** Clean up distorted/ and input/ after COLMAP completes. Offer "aggressive cleanup" that removes everything except the final PLY.

#### 1.15 Frontend polls status every 1.5s even when idle
**File:** `index.html:1232`
**Issue:** `setTimeout(pollStatus, 1500)` polls continuously, including when the job is complete (before the response arrives). Multiple tabs multiply this.
**Fix:** Use exponential backoff when status hasn't changed, or switch to SSE/WebSocket.

#### 1.16 Frontend references non-existent `mlsharpDevice` element
**File:** `index.html:1078`
**Issue:** `document.getElementById('mlsharpDevice').value` — there is no element with id `mlsharpDevice` in the HTML. This would throw a JS error when processing with ML-Sharp.
**Fix:** Either add the device selector to the HTML, or remove the reference.

### LOW — Minor Issues

#### 1.17 `write_ply_file` writes one struct.pack per vertex
**File:** `gaussian_splat_utils.py:78-86`
**Issue:** For 1M points, this does 1M individual `f.write()` calls with `struct.pack()`.
**Fix:** Batch into numpy arrays and use `tobytes()` for ~10x faster PLY writing.

#### 1.18 Excessive log output for large datasets
**File:** `run_glomap.py:505-632`
**Issue:** Every COLMAP output line triggers regex matching against ~10 patterns. For 500 images, COLMAP produces thousands of output lines.
**Impact:** Minor CPU overhead, but log buffer fills up fast (deque maxlen=500).
**Fix:** Sample log lines (every Nth) for high-frequency messages like per-image processing.

#### 1.19 `gsplat_mcmc_trainer.py` doesn't support multi-GPU
**Issue:** The trainer always uses `torch.device("cuda")` which defaults to GPU 0. The `CUDA_VISIBLE_DEVICES` env var set by `app.py` handles this at the subprocess level, but the trainer itself has no awareness.
**Impact:** If called directly (not through app.py), it ignores GPU assignment.

---

## 2. Feature Roadmap

### Tier 1 — High Impact, Relatively Easy

#### 2.1 Batch Processing Queue — MODULE EXISTS, NOT WIRED UP
**Description:** Upload multiple sets of images, each as a separate job. Process them sequentially or in parallel (up to MAX_CONCURRENT_JOBS). Show a job queue dashboard.
**Current state:** `batch_processing.py` has a complete `BatchQueue` class with `BatchJob`, Flask routes (`/batch/create`, `/batch/<id>/add`, `/batch/<id>/start`, `/batch/<id>/status`, `/batch/<id>/cancel`, `/batch/list`, `/batch/<id>/remove`, `/batch/upload-folder`), and ETA calculation. `test_batch.py` exercises the API. **But `register_batch_routes()` is never called in `app.py`** — the routes are dead code.
**What's needed:**
- Call `register_batch_routes(app, ...)` in `app.py` with proper callbacks
- Fix the self-HTTP-call design (`batch_processing.py:286-313` calls `localhost:5000/upload` via requests — should call `process_images_async()` directly)
- Add batch dashboard UI to `index.html`
**Complexity:** Low (mostly wiring)
**Value:** Very high — the #1 requested feature for production workflows

#### 2.2 Export Formats (FBX, OBJ, GLTF, USDZ)
**Description:** Export the trained Gaussian Splat as mesh-based formats for use in Blender, Unity, Unreal, etc.
**Implementation:**
- **PLY → mesh:** Use Poisson surface reconstruction (Open3D) or Delaunay triangulation on the dense point cloud
- **FBX export:** Use Autodesk FBX SDK (Python bindings) or `trimesh` + `pyfbx`
- **GLTF/GLB:** Use `trimesh` which has native GLTF export
- **USDZ:** Use `pxr` (Pixar USD) Python package
- **OBJ:** Trivial — `trimesh` or custom writer
**Complexity:** Medium-High (mesh reconstruction is the hard part)
**Value:** Very high — bridges the gap to traditional 3D workflows

#### 2.3 Point Cloud Cleanup & Filtering Tools
**Description:** Before training, let users remove outlier points, crop the bounding box, or filter by confidence.
**Implementation:**
- Statistical outlier removal (SOR) from Open3D
- Bounding box crop UI in the viewer
- Confidence-based filtering (remove low-confidence 3D points from COLMAP)
**Complexity:** Medium
**Value:** High — dramatically improves output quality for noisy reconstructions

#### 2.4 Auto Scene Detection for Videos
**Description:** When processing a video, automatically detect scene changes and create separate reconstruction jobs per scene.
**Implementation:**
- FFmpeg `select='gt(scene,0.3)'` filter for scene detection
- Split video at scene boundaries
- Queue each segment as a separate job
**Complexity:** Medium
**Value:** High — makes video input practical for real-world footage

### Tier 2 — High Impact, More Complex

#### 2.5 Video Camera Tracking & FBX Export
**Description:** Extract camera movement from video footage and export as FBX for matchmoving in Blender/Maya/Unreal. (See Section 3 for full details.)
**Complexity:** High
**Value:** Very high — unique feature, no easy open-source alternative

#### 2.6 Real-Time Preview During Training
**Description:** Show a live preview of the Gaussian Splat as it trains, updating every N steps.
**Implementation:**
- gsplat MCMC trainer: render every 100 steps, save a small preview PLY
- Brush: capture intermediate exports
- WebSocket push to frontend
- Small preview canvas in the UI
**Complexity:** High
**Value:** High — huge UX improvement, early quality feedback

#### 2.7 3DGS Compression & LOD
**Description:** Compress the output PLY for faster loading and smaller files. Generate LODs for streaming.
**Implementation:**
- **SH coefficient quantization:** Reduce from float32 to uint8 with lookup tables
- **Gaussian pruning:** Remove low-opacity Gaussians post-training
- **LOD generation:** Train at multiple resolutions, merge
- **Compressed PLY format:** Use the gsplat compressed format or custom binary
- **Web streaming:** Progressive loading in the viewer
**Complexity:** High
**Value:** High — critical for web/mobile viewing

#### 2.8 AI-Powered Image Enhancement (Pre-processing)
**Description:** Auto-enhance input images before COLMAP: denoise, sharpen, correct exposure, remove lens distortion.
**Implementation:**
- **Denoising:** OpenCV fastNlMeansDenoising or a lightweight neural denoiser
- **Sharpening:** Unsharp mask or Real-ESRGAN for super-resolution
- **Exposure correction:** Histogram equalization or CLAHE
- **Lens correction:** Auto-detect from EXIF, apply undistortion
**Complexity:** Medium
**Value:** Medium-High — improves results for suboptimal input

#### 2.9 Mesh Extraction from Gaussians
**Description:** Convert the trained Gaussian Splat into a textured mesh for traditional 3D workflows.
**Implementation:**
- **SuGaR (Surface-Aligned Gaussian Splatting):** Extract mesh directly from Gaussians
- **2DGS (2D Gaussian Splatting):** Flatten Gaussians to surfels, extract mesh via TSDF fusion
- **NeuS2 / nvdiffrec:** Neural mesh extraction
- **Simple approach:** Render depth maps from multiple views, fuse with TSDF
**Complexity:** Very High
**Value:** Very high — the missing link between 3DGS and traditional 3D

### Tier 3 — Differentiating Features

#### 2.10 Text-to-3D Generation
**Description:** Generate 3D scenes from text prompts using score distillation (DreamFusion-style).
**Implementation:**
- Integrate `threestudio` or `GaussianDreamer`
- Use Stable Diffusion for score distillation
- Output as Gaussian Splat
**Complexity:** Very High
**Value:** High — trendy, attracts users, but quality is still limited

#### 2.11 Image-to-3D (Single/Multi-view)
**Description:** Beyond ML-Sharp: use recent multi-view diffusion models to generate 3D from 1-4 images.
**Implementation:**
- **Zero-1-to-3++** or **Wonder3D** for novel view synthesis
- **InstantMesh** or **LGM** for fast reconstruction
- Feed generated views into the COLMAP pipeline
**Complexity:** Very High
**Value:** High — much better quality than ML-Sharp alone

#### 2.12 Collaborative / Multi-User Processing
**Description:** Allow multiple users to submit jobs to the same server. User authentication, job ownership, quotas.
**Implementation:**
- Simple API key auth
- Per-user job limits
- Job sharing with shareable links
**Complexity:** Medium-High
**Value:** Medium — useful for teams, studios

#### 2.13 Cloud Processing / Remote GPU
**Description:** Offload heavy processing (COLMAP MVS, Brush training) to cloud GPU instances.
**Implementation:**
- Spin up spot instances on AWS/GCP/Lambda Labs
- Transfer images + reconstruction data
- Stream results back
- Cost estimation before processing
**Complexity:** Very High
**Value:** High — enables processing on any device

#### 2.14 Custom Training Schedules
**Description:** Let users define training step budgets per stage: e.g., "1000 steps at 480p, 5000 at 720p, 20000 at 1080p" (progressive resolution training).
**Implementation:**
- Training schedule JSON format
- Modify Brush/gsplat commands to support multi-phase training
- UI for creating/editing schedules
**Complexity:** Medium
**Value:** Medium — power users love this

#### 2.15 Splat Editing in Viewer
**Description:** Interactive tools in SuperSplat to crop, color-adjust, remove Gaussians, merge splats.
**Implementation:**
- Fork SuperSplat viewer with editing tools
- Selection tools (box, lasso, brush)
- Gaussian deletion, color grading, opacity adjustment
- Save edited splat
**Complexity:** Very High (viewer-side)
**Value:** Very high — huge differentiator

#### 2.16 PLY Comparison & Diff
**Description:** Load two splats side-by-side and visualize differences (quality comparison between presets, before/after editing).
**Complexity:** Medium
**Value:** Medium — useful for quality tuning

#### 2.17 Automatic Quality Assessment
**Description:** After training, automatically render novel views and compute quality metrics (PSNR, SSIM, LPIPS) against held-out images.
**Implementation:**
- Hold out 10-20% of input images
- Render from their camera poses
- Compute metrics
- Display quality score in the UI
**Complexity:** Medium
**Value:** Medium-High — objective quality feedback

---

## 3. Video Camera Tracking & Matchmoving

### Overview

This is the feature inspired by the Polyfjord COLMAP + Blender workflow. The goal: take a video, extract camera movement, export as FBX for use in Blender/Maya/Unreal/Unity.

### How It Works (Technical)

COLMAP doesn't just reconstruct 3D — it also estimates the **6-DOF pose** (position + rotation) of every camera/image. This is exactly what a matchmover needs.

The pipeline:
1. **Video → Frames:** FFmpeg extracts frames at a configurable rate
2. **Frames → COLMAP:** Feature extraction, matching, sparse reconstruction
3. **COLMAP → Camera Poses:** Extract camera positions and orientations from `images.bin`
4. **Poses → FBX:** Write camera animation data into an FBX file

### Implementation Plan

#### 3.1 Camera Pose Extraction Module
**File:** New `camera_tracking.py`

```python
# Pseudocode structure
def extract_camera_poses(colmap_reconstruction_path):
    """Extract camera poses from COLMAP reconstruction.
    
    Returns list of dicts:
    [
        {
            'frame': 0,
            'position': [x, y, z],
            'rotation': [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]],
            'focal_length': fx,
            'frame_time': 0.033,  # seconds
        },
        ...
    ]
    """
    # Use pycolmap to read reconstruction
    # Extract image.cam_from_world() for each registered image
    # Invert to get world_from_camera (camera position in world space)
    # Sort by frame number
    # Return structured data
```

#### 3.2 FBX Export Module
**File:** New `fbx_export.py`

**Library options for FBX export:**
- **Autodesk FBX SDK** (official, but C++ with Python bindings, large binary)
- **trimesh** (supports GLTF/OBJ but not FBX natively)
- **pyfbx** (pure Python, limited but works for cameras)
- **Custom binary writer** (FBX format is documented)
- **Best approach:** Export as **GLTF** first (trivial with trimesh), then offer FBX as optional via FBX SDK

**What the FBX/GLTF contains:**
- A camera object with animated position/rotation keyframes
- Frame rate matching the source video
- Optional: the sparse point cloud as a reference mesh
- Optional: the dense point cloud for scene context

#### 3.3 Frame Extraction Optimization for Tracking
**Current:** `extract_frames_from_video` in `app.py:316-380` uses OpenCV frame-by-frame reading.

**Improvements for camera tracking:**
- **Adaptive frame rate:** Extract more frames for fast camera motion, fewer for slow
- **Scene detection:** Only process frames within a continuous shot
- **FFmpeg-based extraction:** Much faster than OpenCV for large videos
- **Blur detection per frame:** Skip blurry frames (camera shake, motion blur)

#### 3.4 COLMAP Optimizations for Video Tracking
**Key differences from 3D reconstruction:**
- **Sequential matcher is mandatory** (not exhaustive) — video frames are ordered
- **Loop closure detection** helps with long clips that revisit scenes
- **Lower feature count is OK** — we need camera poses, not a dense model
- **Can skip MVS entirely** — sparse reconstruction gives us all the camera data we need
- **Can use lower resolution** — 720p is often sufficient for tracking

**Optimized preset:**
```json
{
    "name": "camera_tracking",
    "features": 8192,
    "peak": 0.008,
    "octaves": 4,
    "match_ratio": 0.85,
    "tri_angle": 0.5,
    "reproj_error": 8.0,
    "dense": false,
    "matcher": "sequential_matcher",
    "description": "Fast camera tracking — poses only, no dense reconstruction"
}
```

#### 3.5 Blender Integration
**Blender can import COLMAP data via:**
- **Photogrammetry Importer** addon (mentioned in Polyfjord video)
- **Custom import script** using Blender's Python API
- **COLMAP-to-Blender converter** scripts (several exist on GitHub)

**What to generate:**
1. A `.json` or `.txt` file with camera poses (Blender-compatible format)
2. A `.ply` file with the sparse point cloud (for reference)
3. An `.fbx` file with animated camera (for any 3D software)
4. A `.py` Blender import script that sets up the scene automatically

#### 3.6 UI for Camera Tracking
**New workflow in the frontend:**
- "Camera Tracking" mode selector (alongside Multi-Image and ML-Sharp)
- Video upload with frame rate slider
- Preview: show extracted frames in a timeline
- Output format selector: FBX / GLTF / Blender script / JSON
- "Track Camera" button
- Results: camera path visualization, download options

#### 3.7 Advanced Tracking Features
- **Stabilization data export:** Track points that stay stable across frames
- **Ground plane detection:** Auto-detect the floor/ground plane from the point cloud
- **Scale estimation:** If known objects are in the scene, estimate real-world scale
- **Lens distortion export:** Include radial/tangential distortion coefficients
- **Multi-video tracking:** Track cameras across multiple video clips of the same scene

### Comparison with Commercial Tools

| Feature | images_to_play (planned) | PFTrack | SynthEyes | Boujou | Mocha Pro |
|---------|-------------------------|--------|-----------|--------|-----------|
| Price | Free | $1,500/yr | $500 | $3,000 | $1,500/yr |
| Auto tracking | Yes (COLMAP) | Yes | Yes | Yes | Yes |
| Manual refinement | No | Yes | Yes | Yes | Yes |
| 3D reconstruction | Yes | No | No | No | No |
| Gaussian Splat | Yes | No | No | No | No |
| FBX export | Yes | Yes | Yes | Yes | Yes |
| Blender integration | Yes | Plugin | Plugin | No | Plugin |

**Key advantage:** The combination of camera tracking + 3D reconstruction + Gaussian Splatting in one tool is unique. Commercial tools do tracking only; they don't generate 3D content.

---

## 4. Batch Processing System

### Architecture

#### 4.1 Job Queue Manager
```python
# Pseudocode
class BatchQueue:
    def __init__(self, max_concurrent=3):
        self.queue = []  # List of (priority, job_spec)
        self.running = {}  # job_id -> job_info
        self.completed = []
    
    def add_job(self, job_spec, priority=0):
        """Add a job to the queue. Lower priority number = runs first."""
        
    def process_next(self):
        """Pick the highest-priority pending job and start it."""
        
    def get_status(self):
        """Return queue status: pending, running, completed, failed."""
```

#### 4.2 Batch Upload API
```
POST /batch/upload
  - Accept multiple file groups (each group = one reconstruction job)
  - Each group has its own preset, settings
  - Returns batch_id with list of job_ids

GET /batch/<batch_id>/status
  - Returns status of all jobs in the batch
  - Overall progress percentage
  - Estimated time remaining

POST /batch/<batch_id>/cancel
  - Cancel all pending jobs in the batch
```

#### 4.3 Folder-Based Batch Input
Allow users to drop a folder structure where each subfolder is a separate job:
```
batch_input/
├── scene_1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── scene_2/
│   ├── img_001.jpg
│   └── ...
└── scene_3/
    └── video.mp4
```

#### 4.4 Batch Dashboard UI
- Table view: job name, status, progress, elapsed time, ETA
- Bulk actions: cancel all, retry failed, download all results
- Drag-and-drop reordering of pending jobs
- Per-job settings override
- ZIP download of all completed results

### Multi-GPU Batch Processing
The existing GPU assignment system (`assign_gpu`, `release_gpu`) already supports multi-GPU. Extend it for batch:
- Assign different jobs to different GPUs automatically
- Show GPU utilization in the dashboard
- Allow manual GPU assignment per job

---

## 5. Competitive Landscape

### Consumer Apps

| Tool | Key Features | What We Can Learn |
|------|-------------|-------------------|
| **Luma AI** | Cloud-based, web viewer, NeRF + 3DGS, share links | Cloud processing, social sharing, embed links |
| **Polycam** | Mobile capture, LiDAR support, room scanning, export formats | Mobile-first capture, guided shooting UX |
| **Scaniverse** | Mobile, auto-scanning, real-time preview, social feed | Real-time feedback, community features |
| **KIRI Engine** | Cloud, photo/phone capture, free tier, web viewer | Free tier model, web-only workflow |
| **Meshroom** | Open source, node-based pipeline, AliceVision | Node-based customization, academic-grade algorithms |

### Professional Tools

| Tool | Key Features | What We Can Learn |
|------|-------------|-------------------|
| **RealityCapture** | Fastest reconstruction, mesh+texture, PPI licensing | Speed optimizations, mesh output, enterprise features |
| **Nerfstudio** | Modular, many methods (3DGS, NeRF, etc.), CLI+GUI | Plugin architecture, method selection |
| **Postshot** | Desktop app, 3DGS training, real-time preview | Live preview, simple UX |
| **Agisoft Metashape** | Mature, orthomosaics, DEM, GIS integration | GIS/ortho features for survey use cases |
| **3DF Zephyr** | User-friendly, batch processing, cloud option | Batch processing UX, cloud offloading |

### Open Source / Research

| Tool | Key Features | What We Can Learn |
|------|-------------|-------------------|
| **gsplat** | MCMC strategy, CUDA kernels, fast rasterization | Already integrated; keep up with updates |
| **3D Gaussian Splatting** (original) | Reference implementation | Benchmark against |
| **SuGaR** | Mesh extraction from Gaussians | Mesh export pipeline |
| **2DGS** | 2D Gaussians for better surfaces | Alternative representation |
| **GaussianEditor** | Edit Gaussians with text/brush | Editing tools |
| **DreamGaussian** | Text/image to 3DGS | AI generation features |
| **4DGS** | Dynamic/temporal Gaussians | Video-to-4D feature |
| **COLMAP** | SfM, MVS, camera poses | Already the backbone |
| **OpenMVS** | Open-source MVS alternative | Alternative dense reconstruction |

### Key Differentiators We Should Build

1. **Camera tracking + 3DGS in one tool** — nobody else does this
2. **Batch processing with queue** — most tools are single-job
3. **Multiple trainers** (Brush + gsplat) — user choice
4. **Preset system** — easy for beginners, powerful for experts
5. **Self-hosted** — no cloud dependency, no data leaves the machine
6. **Open source** — fully customizable

---

## 6. Priority Matrix

### P0 — Do First (High impact, moderate effort)
| # | Feature | Effort | Impact | Status |
|---|---------|--------|--------|--------|
| 1 | Fix Brush stdout streaming (1.1) | Low | High | **Broken** |
| 2 | Cache COLMAP path at startup (1.2) | Low | Medium | **Missing** |
| 3 | Parallel image resize (1.3) | Low | Medium | **Missing** |
| 4 | Fix mlsharpDevice JS error (1.16) | Trivial | High | **Bug** |
| 5 | Wire up batch processing (2.1) | Low | Very High | **Code exists, not connected** |
| 6 | Fix batch self-HTTP call | Low | High | **Design flaw** |
| 7 | Camera tracking module (3.1-3.4) | Medium | Very High | **Not started** |

### P1 — Do Next (High impact, higher effort)
| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 7 | FBX/GLTF export (2.2) | Medium-High | Very High |
| 8 | Auto scene detection for video (2.4) | Medium | High |
| 9 | MVS cache_size auto-detection (1.9) | Low | High |
| 10 | Point cloud cleanup tools (2.3) | Medium | High |
| 11 | Batch dashboard UI (4.4) | Medium | High |
| 12 | gsplat GPU memory optimization (1.7) | Medium | High |

### P2 — Plan For (Differentiating features)
| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 13 | Mesh extraction from Gaussians (2.9) | Very High | Very High |
| 14 | Real-time preview during training (2.6) | High | High |
| 15 | Compression & LOD (2.7) | High | High |
| 16 | AI image enhancement pre-processing (2.8) | Medium | Medium-High |
| 17 | Splat editing in viewer (2.15) | Very High | Very High |
| 18 | Text-to-3D generation (2.10) | Very High | High |

### P3 — Future Vision
| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 19 | Multi-view image-to-3D (2.11) | Very High | High |
| 20 | Cloud processing (2.13) | Very High | High |
| 21 | Collaborative multi-user (2.12) | Medium-High | Medium |
| 22 | Dynamic 4D Gaussians | Very High | High |

---

## Appendix A: FFmpeg Camera Tracking Workflow (Polyfjord Method)

### Folder Structure
```
project/
├── 01_colmap/          # COLMAP database and reconstruction
├── 02_videos/          # Source video files
├── 03_ffmpeg/          # FFmpeg binary
├── 04_scenes/          # Extracted frames per scene
└── 05_script/          # Automation scripts
```

### Steps
1. `ffmpeg -i video.mp4 -qscale:v 2 frames/%06d.jpg` — extract frames
2. `colmap feature_extractor --database_path db.db --image_path frames/`
3. `colmap sequential_matcher --database_path db.db`
4. `colmap mapper --database_path db.db --image_path frames/ --output_path sparse/`
5. Import into Blender via Photogrammetry Importer addon

### Key Tips from Polyfjord
- **Disable image stabilization** on the camera — it ruins reconstruction
- Use **Blackmagic Camera app** on phones (disable stabilization, enable lens correction)
- COLMAP handles heavy camera shake surprisingly well
- Works even with vertical (portrait) video
- Works with moving objects in the scene (they become part of the point cloud)
- Use **Movie Clip Editor** proxy files in Blender for smooth playback of long clips

### Libraries for FBX Export in Python
| Library | FBX Support | Notes |
|---------|------------|-------|
| Autodesk FBX SDK | Full | Official, large binary, Python 3 bindings |
| pyfbx | Basic | Pure Python, camera export works |
| trimesh | GLTF/OBJ only | Excellent for meshes, no FBX |
| assimp (pyassimp) | Limited FBX | Can export basic scenes |
| Blender Python API | Full | Requires Blender installed |

**Recommended approach:** Export as GLTF (widely supported, trimesh handles it), and optionally offer FBX via FBX SDK for users who need it.

---

## Appendix B: Speed Optimization Quick Wins

| Optimization | Time Saved | Effort |
|-------------|-----------|--------|
| Cache COLMAP path | 1-40s/job | 10 min |
| Parallel image resize | 10-60s (100+ images) | 30 min |
| Symlinks instead of copies for interval filter | 5-30s | 15 min |
| Auto-detect RAM for MVS cache_size | Prevents crashes | 20 min |
| Blur filter on downscaled images | 5-20s | 20 min |
| Combined resize+blur pass | 5-30s | 45 min |
| Shell=False for subprocess calls | 1-2s per command | 1 hr |
| Batch struct.pack in PLY writer | 1-5s for large PLYs | 15 min |
| Progressive image loading for gsplat MCMC | Prevents OOM | 1 hr |
| Use FFmpeg for frame extraction (instead of OpenCV) | 2-5x faster | 30 min |
