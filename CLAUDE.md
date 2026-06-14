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

**Do not call `dense_reconstruction.py` when `run_glomap` already ran MVS** (preset JSON has `dense: true`, or `enable_dense_override` forced it) — running MVS twice causes workspace conflicts. Note: all current presets ship `dense: false`, so MVS only ever runs when explicitly requested.

### Preset system

Presets are defined as JSON files in `simple_splat/App/presets/`. The inline dict in `run_glomap.py` is a fallback only used if the `presets/` directory is missing. `_load_presets()` loads JSON files at runtime.

Available presets: `low`, `medium`, `high`, `quality`, `expert`

Note: `dense`, `maximum`, `ultra`, `extreme`, `insane`, `unlimited`, and `sharpness` were removed in the preset cleanup — check `presets/*.json` for the current list.

To add or tune a preset: edit the corresponding `.json` file — no Python changes needed.

Key parameters in each preset:
- `features` — max SIFT features per image (0 = unlimited)
- `peak` — SIFT peak threshold (lower = more keypoints)
- `octaves` — SIFT octave count (more = multi-scale, capped at 8)
- `match_ratio` — Lowe's ratio test threshold (capped at 0.99)
- `tri_angle` — min triangulation angle in degrees (capped at 0.01)
- `reproj_error` — max reprojection error in pixels (capped at 128)
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

### MCMC auto-cap

When `--cap-max` is omitted, the trainer auto-scales the Gaussian cap to the sparse-point count:
- `cap = sparse_pts × 30`, clamped to **500k–2M**
- The app (`app.py`) also uses auto by default (no more hardcoded 1M)
- Explicit `--cap-max N` still works for manual control

### External binaries
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
- `gsplat` — required by ml-sharp at import time even if rendering is not used. **JIT-compiles its CUDA kernels on first use** — needs MSVC `cl.exe` on PATH (app.py discovers and adds it at startup; for standalone scripts, add `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\<ver>\bin\Hostx64\x64` to PATH). First compile takes ~10 min; cached afterwards.
- `pytorch-msssim` — Gaussian-windowed SSIM for the MCMC trainer loss. Do NOT replace with a hand-rolled avg_pool SSIM: its biased border statistics destroy training (verified: 32 dB vs 10.5 dB PSNR on synthetic data).

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

Derived from analysis of LichtFeld-Studio and the current pipeline's bottlenecks. Phases 1-4 implemented 2026-05-25.

### Phase 1 — Quick Wins ✅ DONE

#### 1.1 Auto-resize images before COLMAP ✅
`resize_images_for_colmap(image_folder, max_size)` in `app.py`. Caps images at `max_image_size` from config before COLMAP. Called in `process_images_async()`.

#### 1.2 Skip MVS toggle ✅
`enable_dense_override` parameter added to `run_colmap()`. "Enable Dense Reconstruction" checkbox in Advanced Settings.

#### 1.3 Named phase labels in progress stream ✅
`stage` field added to `processing_status[job_id]`. Frontend `updateStagesFromStage()` uses it for accurate stage highlighting.

#### 1.4 JSON preset files ✅
`_load_presets()` in `run_glomap.py` loads from `presets/` directory. Inline dict is fallback.

#### 1.5 Blur filter before COLMAP ✅
`filter_blurry_images(image_folder, blur_threshold)` in `app.py`. Warn-only (doesn't delete). Called before `run_colmap()`.

### Phase 2 — Medium Improvements ✅ DONE

#### 2.1 Quality scaler ✅
Draft 0.3x / Standard 1.0x / Cinematic 2.0x multiplier. `quality_scale` select in UI. Applied to training_steps in `process_images_async()`.

#### 2.2 Brush progress streaming ✅
Brush uses `Popen` with stdout streaming. Step numbers parsed from output for real-time progress.

#### 2.3 Fix Brush image folder resolution ✅
Now prefers undistorted `images/` folder. Only falls back to `source/` if empty.

#### 2.4 Stage timings ✅
Elapsed time per stage tracked in `processing_status[job_id]['stages']`. Frontend displays live timing next to each stage label via `updateStageTimings()`.

### Phase 3 — gsplat MCMC Trainer ✅ DONE

#### 3.1 `simple_splat/App/gsplat_mcmc_trainer.py` ✅
Implements `load_colmap_dataset()` and `train_mcmc()` using gsplat's `MCMCStrategy` and `rasterization()`. Uses `SelectiveAdam` optimizer per parameter group. Hyperparameters follow gsplat's reference `examples/simple_trainer.py` (mcmc preset): per-param LRs, means lr × scene_scale with exponential decay to 1%, opacity/scale L1 regularization (0.01 each — required for MCMC relocation to work), SH degree warmup (one band per 1000 steps).

**Critical:** pycolmap's `image.cam_from_world` is already world-to-camera (w2c) and gsplat's `rasterization(viewmats=...)` expects w2c — pass it through directly, do NOT invert.

**Critical:** optimizers must be built over the same `ParameterDict` entries used in the render graph — building them over pre-copy tensors leaves `grad=None` and `SelectiveAdam` silently skips the update.

Smoke test: `python test_mcmc_smoke.py` (needs CUDA + MSVC on PATH) — trains a synthetic scene and asserts loss collapse, PSNR gain, and MCMC growth.

#### 3.2 Integration into `app.py` ✅
Trainer selector in UI: "Brush" (default) / "gsplat MCMC". Branches in `process_images_async()` after COLMAP completes. Falls back to Brush on error.

---

## Point Cloud Size Reference

| Preset | Sparse points | Brush Gaussians | MCMC Gaussians |
|--------|--------------|-----------------|----------------|
| low     | 10K–100K    | 100K–500K       | up to cap |
| medium  | 50K–500K    | 200K–2M         | up to cap (1M default) |
| high    | 200K–2M     | 1M–10M          | up to cap |
| quality / expert | 500K–5M | 2M–30M     | up to cap |

MCMC always converges to `mcmc_cap` (default 1,000,000, `config.get('mcmc_cap')` in app.py — no UI knob yet). On the RTX 8000 (48 GB) the cap can safely go to 2–4M.

**Dense MVS:** all presets ship `dense: false` (MVS adds nothing for splat training). If enabled via `enable_dense`, with the RTX 8000 the `patch_match_stereo` cache can be raised to 44GB (`--PatchMatchStereo.cache_size 44`) and `fusion_min_pixels` lowered to 1 for maximum density.

**Image count guide:**
- <20 images → use `low` preset (high/medium waste time and produce fewer points due to strict filters)
- 20–50 images → `medium`
- 50–100 images → `high`
- 100+ images → `high` or `quality`

---

## Status Log

**2026-06-12 — MCMC trainer fixed and verified.** The trainer had been broken since inception: (1) c2w passed where gsplat expects w2c, (2) optimizers built over dead tensor copies (`grad=None`, silent no-op), (3) hand-rolled avg-pool SSIM with corrupt border gradients, (4) `export_splats` given activated values where the PLY format stores raw log/logit (splat rendered as giant blobs in viewers). All fixed; hyperparameters aligned to gsplat's reference trainer; LPIPS computed on a 512px crop. Verified: synthetic smoke test 13.4 → 39 dB PSNR, real 74-photo job 25.5 dB mean PSNR, 1M Gaussians, ~13 min training.

**2026-06-12 — GUI redesigned + rebrand.** App renamed **FonixFlow Splat** (directory name `simple_splat/` intentionally unchanged). UI is now tabbed (Create / Settings / Process / Results), no scrolling, auto-switches tabs as the job advances. All element IDs the JS depends on were preserved. Dev preview config in `.claude/launch.json` (name `fonixflow-splat`, port 5000, `autoPort: false` because app.py hardcodes the port).

**Known next steps** (see PROJECT.md): expose `mcmc_cap` in UI, antialiased rasterization + bilateral grid options, VGGT/MASt3R fallback when COLMAP fails, Brush-vs-MCMC benchmark on identical input.
