# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

Three independent sub-projects in one repo:

```
simple_splat/App/     — Flask web app: image → 3D Gaussian Splat pipeline
supersplat-src/       — SuperSplat viewer source (TypeScript/Rollup, builds to simple_splat/App/static/supersplat/)
ml-sharp/             — Apple ML-Sharp single-image Gaussian splat model (Python package)
```

---

## simple_splat (Flask App)

### Running the server

```bash
cd simple_splat/App
python app.py
# Server starts at http://localhost:5000
```

The server runs with `debug=True, use_reloader=False, use_debugger=False`. The reloader must stay disabled — when it is enabled, ML-Sharp detection runs only in the parent process but requests are served by a child process where `mlsharp_available` is always `False`.

ML-Sharp detection (`sharp --help`) has a 30-second timeout because PyTorch takes ~6s to import on first run.

### COLMAP flag compatibility

The installed COLMAP is **4.1.0** (from the 4.0.4 release tag). In 4.x the namespaces split:
- GPU/thread/general options → `FeatureExtraction.*` and `FeatureMatching.*`
- SIFT algorithm tuning → `SiftExtraction.*` and `SiftMatching.*`

Always use:
- `--FeatureExtraction.use_gpu`, `--FeatureExtraction.gpu_index -1`, `--FeatureExtraction.num_threads`, `--FeatureExtraction.max_image_size`
- `--SiftExtraction.max_num_features`, `--SiftExtraction.peak_threshold`, `--SiftExtraction.num_octaves`, `--SiftExtraction.first_octave`, `--SiftExtraction.edge_threshold`
- `--FeatureMatching.use_gpu`, `--FeatureMatching.gpu_index -1`, `--FeatureMatching.num_threads`, `--FeatureMatching.max_num_matches`, `--FeatureMatching.guided_matching`
- `--SiftMatching.max_ratio`

GLOMAP is now built into COLMAP 4.x as `colmap global_mapper` — no separate binary needed. Flags use `GlobalMapper.*` namespace.

COLMAP lives at `C:\COLMAP\bin\colmap.exe`. All DLLs are bundled in `bin/` (4.x no longer needs a separate `lib/`). The app adds `C:\COLMAP\bin` to `os.environ["PATH"]` at startup.

### Processing pipeline architecture

`app.py` orchestrates everything; the three helper modules do not import each other:

| Module | Role |
|--------|------|
| `run_glomap.py` | COLMAP/GLOMAP pipeline — feature extraction, matching, sparse reconstruction, optional MVS |
| `dense_reconstruction.py` | Fallback MVS — only called when `run_glomap` did NOT run MVS (i.e., `low` preset + dense requested via advanced settings) |
| `gaussian_splat_utils.py` | Last-resort fallback — generates a basic PLY from sparse reconstruction if Brush fails |

**Do not call `dense_reconstruction.py` for medium/high/sharpness presets** — those presets run MVS inside `run_glomap.py` and calling it again causes workspace conflicts.

### Preset system

Presets are defined as JSON files in `simple_splat/App/presets/`. The inline dict in `run_glomap.py` is a fallback only used if the `presets/` directory is missing.

Available presets: `low`, `medium`, `high`, `ultra`, `extreme`, `maximum`, `insane`, `unlimited`, `dense`, `expert`, `sharpness`

To add or tune a preset: edit the corresponding `.json` file — no Python changes needed.

Key parameters in each preset:
- `features` — max SIFT features per image (0 = unlimited)
- `peak` — SIFT peak threshold (lower = more keypoints)
- `octaves` — SIFT octave count (more = multi-scale)
- `match_ratio` — Lowe's ratio test threshold (higher = keep more matches)
- `tri_angle` — min triangulation angle in degrees (lower = more points kept)
- `reproj_error` — max reprojection error in pixels (higher = more points kept)
- `dense` — whether to run MVS (patch_match_stereo + stereo_fusion)
- `mvs_window_radius`, `mvs_iterations`, `mvs_samples` — MVS quality settings

### Job output layout

```
simple_splat/App/processing/<uuid>/
├── source/              — original uploaded images
├── images/              — undistorted images (for Brush training)
├── sparse/0/            — cameras.bin, images.bin, points3D.bin
├── dense/fused.ply      — dense MVS point cloud (if enabled)
├── gaussian_splat.ply   — Brush-trained splat (primary output)
├── point_cloud.ply      — fallback sparse PLY
└── colmap_run.log       — full COLMAP command transcript
```

### External binaries

| Binary | Location | Required |
|--------|----------|----------|
| COLMAP | `C:\COLMAP\bin\colmap.exe` | Yes |
| Brush | `simple_splat\Brush\brush_app.exe` | No (falls back to sparse PLY) |
| GLOMAP | built into COLMAP 4.x as `colmap global_mapper` | No (falls back to incremental mapper) |
| sharp (ML-Sharp) | in PATH via pip install | No (feature disabled if absent) |

### Python dependencies

Installed in the system Python (3.14). Key packages and their non-obvious requirements:
- `torch` must be the **CUDA build** (`torch==2.x+cu126`), not `+cpu`. Install with `--index-url https://download.pytorch.org/whl/cu126 --force-reinstall`.
- `pycolmap` — used only to read reconstruction stats; not required for the pipeline itself.
- `gsplat` — required by ml-sharp at import time even if rendering is not used.

### Viewer integration

SuperSplat is served from `simple_splat/App/static/supersplat/`. The viewer loads splats via the `?load=` URL parameter:
```
/static/supersplat/index.html?load=/ply/<job_id>.ply
```
The `index.js` bundle is **not committed** — it must be built from `supersplat-src/` (see below). The `index.js.map` source map is committed; replace it when rebuilding.

---

## supersplat-src (Viewer Source)

### Building

```bash
cd supersplat-src
npm install
npm run build        # outputs to supersplat-src/dist/
```

After building, copy artifacts to the app's static folder:
```bash
cp supersplat-src/dist/index.js   simple_splat/App/static/supersplat/index.js
cp supersplat-src/dist/index.js.map simple_splat/App/static/supersplat/index.js.map
cp supersplat-src/dist/index.css  simple_splat/App/static/supersplat/index.css
cp supersplat-src/dist/sw.js      simple_splat/App/static/supersplat/sw.js
```

### Development server (watch mode)

```bash
cd supersplat-src
npm run develop      # watch + serve at http://localhost:3000
```

The viewer reads splats from the `?load=` query param (see `supersplat-src/src/main.ts` line ~249). Other supported params: `?focal=x,y,z`, `?angles=az,elev`, `?distance=n`.

---

## ml-sharp (Apple ML-Sharp)

### Installation

The package lives at `ml-sharp/` and is installed in editable mode:
```bash
cd ml-sharp
pip install -e . --no-deps
pip install timm imageio imageio-ffmpeg matplotlib pillow-heif plyfile scipy gsplat
pip install torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### CLI usage

```bash
sharp predict -i /path/to/image/folder -o /path/to/output
sharp predict -i input/ -o output/ --render   # also render novel views (CUDA only)
```

Model checkpoint (~500MB) auto-downloads to `~/.cache/torch/hub/checkpoints/` on first run.

### How the app calls it

`app.py:process_mlsharp_async()` runs `sharp predict -i <images_folder> -o <output_dir> --device cuda` as a subprocess and streams stdout for progress. The output `.ply` is copied to `processing/<uuid>/gaussian_splat.ply` for viewer compatibility.

---

## Improvement Roadmap

Derived from analysis of LichtFeld-Studio (https://github.com/MrNeRF/LichtFeld-Studio) and the current pipeline's bottlenecks. **Nothing below is implemented yet** — this is the plan for the next phase of work.

### Phase 1 — Quick Wins

#### 1.1 Auto-resize images before COLMAP
**File:** `app.py` — new `resize_images_for_colmap(image_folder, scene_type)` function  
**Why:** LichtFeld caps images at 3840px. Outdoor scenes benefit from 4x downsample (→1920px), indoor from 2x (→2560px). 4K input images slow COLMAP by 16x vs 2K.  
**What to add:**
- `resize_images_for_colmap(image_folder, scene_type='auto')` using PIL
- `scene_type`: `'auto'` = cap 3840px, `'outdoor'` = cap 1920px, `'indoor'` = cap 2560px
- Call it in `process_images_async()` after image count check, before `run_colmap()`
- Add `scene_type` select to `index.html` UI (Auto / Outdoor / Indoor)
- Pass `scene_type` from form in `upload_files()`

#### 1.2 Skip MVS toggle
**Files:** `run_glomap.py`, `app.py`, `index.html`  
**Why:** LichtFeld skips MVS entirely — Gaussian training densifies from sparse init. Saves 20–60 min per job.  
**What to add:**
- Add `enable_dense_override=None` parameter to `run_colmap()` in `run_glomap.py`
- After `settings = detail_settings.get(...)`, add: `if enable_dense_override is not None: settings = dict(settings); settings['dense'] = enable_dense_override`
- In `process_images_async()` in `app.py`: pass `enable_dense_override` to `run_colmap()` based on config
- Add visible "Skip Dense Reconstruction (faster, sparse init)" checkbox to main UI in `index.html`

#### 1.3 Named phase labels in progress stream
**Files:** `app.py`, `run_glomap.py`  
**Why:** Users see a frozen progress bar. Adding a `stage` field to `processing_status` lets the UI highlight active pipeline stage accurately.  
**What to add:**
- Add `stage` key to `processing_status[job_id]`: one of `feature_extraction`, `feature_matching`, `mapping`, `dense_mvs`, `training`
- Update it at each transition in `process_images_async()` and via the COLMAP progress callback
- UI already has stage elements (`#stage1`–`#stage5`) and `updateStagesFromStep()` — just needs the `stage` field from backend

#### 1.4 JSON preset files ✅ DONE
JSON files created at `simple_splat/App/presets/*.json`. The `run_glomap.py` inline dict is the fallback if the directory is missing.  
**Remaining:** Wire JSON loading into `run_glomap.py` — replace the inline `detail_settings` dict with `_load_presets()` call.

```python
# Add at top of run_glomap.py (after imports):
def _load_presets():
    preset_dir = pathlib.Path(__file__).parent / "presets"
    if not preset_dir.exists():
        return None
    presets = {}
    for f in sorted(preset_dir.glob("*.json")):
        with open(f) as fp:
            presets[f.stem] = json.load(fp)
    return presets if presets else None

# In run_colmap(), replace the inline dict assignment:
_json_presets = _load_presets()
detail_settings = _json_presets if _json_presets else { ... existing inline dict ... }
```

#### 1.5 Blur filter before COLMAP
**File:** `app.py` — new `filter_blurry_images(image_folder, blur_threshold, remove_blurry)` function  
**Why:** Blurry frames from video extraction or shaky shots cause COLMAP to fail. OpenCV Laplacian variance detects them cheaply.  
**What to add:**
```python
def filter_blurry_images(image_folder, blur_threshold=100.0, remove_blurry=False):
    import cv2, numpy as np
    blurry = []
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
        img = cv2.imread(os.path.join(image_folder, fname), cv2.IMREAD_GRAYSCALE)
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        if score < blur_threshold:
            blurry.append((fname, score))
    if blurry:
        add_log(f"Blur filter: {len(blurry)} blurry images", "WARNING")
        if remove_blurry:
            for fname, _ in blurry: os.remove(os.path.join(image_folder, fname))
    return blurry
```
- Call in `process_images_async()` after resize, before `run_colmap()`
- Default `warn_only=True` (safe — just logs, doesn't delete)
- Add blur threshold to Advanced Settings in `index.html`

---

### Phase 2 — Medium Improvements

#### 2.1 Quality scaler (steps multiplier)
**Files:** `app.py`, `index.html`  
Replace hardcoded step counts per preset with a single multiplier applied to base config:
- `draft` = 0.3× (quick preview)
- `standard` = 1.0× (preset default)
- `cinematic` = 2.0× (max quality)

Add a "Quality Scale" select to `index.html` main UI. In `process_images_async()`: `training_steps = int(config['training_steps'] * scale)`.

#### 2.2 Brush progress streaming
**File:** `app.py`  
**Why:** Brush is called with `subprocess.run` (blocking). Users see a frozen bar during the longest phase.  
Change to `subprocess.Popen` + stdout streaming (same pattern as COLMAP in `run_glomap.py`). Parse Brush output lines for step numbers. Update `processing_status[job_id]['progress']` and `step` in real time.

The current Brush call is in `process_images_async()` around line 743. The timeout logic (`timeout_seconds = max(7200, ...)`) must be preserved — implement as a deadline on the Popen loop.

#### 2.3 Fix Brush image folder resolution
**File:** `app.py`  
The current logic comparing image counts across `images/`, `source/`, `input/` can overwrite undistorted images with originals. Fix: after `run_colmap()`, use `images/` directly if it exists (created by `image_undistorter`). Only fall back to `source/` if `images/` is empty.

#### 2.4 Stage timings in job status
**File:** `app.py`  
Add `stages` dict to `processing_status[job_id]`:
```python
'stages': {
    'feature_extraction': {'status': 'pending', 'elapsed': None},
    'feature_matching': {'status': 'pending', 'elapsed': None},
    'mapping': {'status': 'pending', 'elapsed': None},
    'dense_mvs': {'status': 'pending', 'elapsed': None},
    'training': {'status': 'pending', 'elapsed': None},
}
```
Expose in `/status/<job_id>`. Show elapsed time per stage in `index.html` next to each stage label.

---

### Phase 3 — gsplat MCMC Trainer

LichtFeld uses MCMC densification (better quality than standard 3DGS adaptive densification). The `gsplat` library (already installed for ml-sharp) supports `MCMCStrategy`.

#### 3.1 New file: `simple_splat/App/gsplat_mcmc_trainer.py`

```python
# Key structure:
def load_colmap_dataset(parent_dir):
    # Uses pycolmap.Reconstruction to read sparse/0/
    # Returns cameras (intrinsics), c2w matrices, images (PIL), sparse_xyz, sparse_rgb
    ...

def train_mcmc(parent_dir, total_steps=10000, cap_max=1_500_000,
               progress_callback=None, output_ply_path=None):
    # 1. Load COLMAP dataset (use undistorted images/  from image_undistorter)
    # 2. Init Gaussians from sparse points
    # 3. Build Adam optimizers per parameter group
    # 4. Init MCMCStrategy(cap_max=cap_max)
    # 5. Training loop with L1 + SSIM loss
    # 6. Export PLY via gsplat.exporter.export_splats
    ...
```

**Critical coordinate system note:** gsplat `rasterization()` takes `viewmats` as world-to-camera 4×4 matrices. pycolmap's `image.cam_from_world` is already world-to-camera — use it directly. Do NOT invert it.

**Use undistorted images:** Always load from `<parent_dir>/images/` (created by COLMAP `image_undistorter`), not `source/`. These are already pinhole — no distortion handling needed.

**Hardware:** RTX 8000 (48GB VRAM) can handle `cap_max=1_500_000` comfortably.

#### 3.2 Integration into `app.py`

Add `trainer` field to upload form: `brush` (default) or `gsplat_mcmc`.  
In `process_images_async()`: after COLMAP completes, branch on `config.get('trainer', 'brush')`.  
Add "Trainer" select to `index.html`.

---

## Point Cloud Size Reference

| Preset | Sparse points | Dense points (MVS) | Brush Gaussians |
|--------|--------------|-------------------|-----------------|
| low    | 10K–100K     | — (disabled)      | 100K–500K       |
| medium | 50K–500K     | 500K–5M           | 200K–2M         |
| high   | 200K–2M      | 5M–30M            | 1M–10M          |
| sharpness | 500K–5M  | 10M–150M          | 2M–30M          |

**Maximizing point count:** Use sharpness preset + 100–200 images with 80% overlap. With RTX 8000 (48GB VRAM), `patch_match_stereo` cache can be raised to 44GB (`--PatchMatchStereo.cache_size 44`) and `fusion_min_pixels` lowered to 1 for maximum density.

**Image count guide:**
- <20 images → use `low` preset (high/medium waste time and produce fewer points due to strict filters)
- 20–50 images → `medium`
- 50–100 images → `high`
- 100+ images → `high` or `sharpness`
