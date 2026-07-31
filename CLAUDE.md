# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

Three independent sub-projects in one repo:

```
simple_splat/App/     — Flask web app: image → 3D Gaussian Splat pipeline
supersplat-src/       — SuperSplat viewer source (TypeScript/Rollup, builds to simple_splat/App/static/supersplat/)
ml-sharp/             — Apple ML-Sharp single-image Gaussian splat model (Python package)
docs/                 — reference docs (APP_GUIDE, SETUP, MODULES, FULL_EDITION_PLAN)
```

**App layout (reorganized 2026-06-14):** `app.py` (entry point) + `batch_processing.py` stay at `simple_splat/App/`; the processing modules live in `simple_splat/App/pipeline/` (`run_glomap`, `dense_reconstruction`, `gaussian_splat_utils`, `gsplat_mcmc_trainer`, `sparse_preview`, `camera_tracking`); tests in `simple_splat/App/tests/`. `app.py` imports them as `from pipeline.X import …`. Standalone CLIs run as `python pipeline/run_glomap.py …`, `python pipeline/gsplat_mcmc_trainer.py …`, etc. `CLAUDE.md` + `PROJECT.md` stay at repo root.

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

The server binds **`127.0.0.1`** by default (it has no auth and executes uploads). Set `SPLAT_HOST=0.0.0.0` to expose it on the LAN deliberately.

### Tests

```bash
cd simple_splat/App
python -m pytest tests/test_cpu_regression.py -q     # ~0.5s, needs only numpy + pytest
```

`tests/test_cpu_regression.py` is the **CPU-only regression suite** — no GPU, COLMAP, Brush, server, or network. It pins the things that have silently regressed before: preset loading, the pure-`struct` COLMAP readers, the pycolmap-free PLY/pose fallbacks, `compress_ply`'s channel-major SH truncation (the green-tint bug), and the binary-FBX conventions (object-name separator, metre scale, declared fps, key grid). It runs in CI (`.github/workflows/tests.yml`, Python 3.11 + 3.14).

The other two test files need real resources and are **not** in CI: `tests/test_mcmc_smoke.py` (CUDA + MSVC) and `tests/test_all_presets.py` (a live server + an image folder).

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

### COLMAP high-quality flags (high / quality / expert)

`run_glomap.py` applies extra SfM quality flags via an `hq = detail_level in ('high','quality','expert')` gate (low/medium stay fast):
- `SiftExtraction.estimate_affine_shape 1` + `domain_size_pooling 1` — far more robust keypoints (CPU-heavy; the main speed cost)
- `FeatureExtraction.max_image_size` raised (`settings.get('max_image_size', 6400)`) — COLMAP's 3200 default was silently downscaling
- `FeatureMatching.guided_matching 1` — re-check matches against epipolar geometry
- `GlobalMapper.ba_refine_principal_point 1` — tighter intrinsics

These close part of the gap to RealityScan (~54k sparse pts vs COLMAP's ~31k on the same 74-photo scene). The deeper gap (textureless surfaces) needs LiDAR or a **learning-based matcher** (LightGlue / MASt3R-SfM / VGGT) — see PROJECT.md.

### 3D alignment viewer (Process tab)

A **dedicated Three.js viewer** (`static/align/viewer.html`, with `three.min.js` + `OrbitControls.js` bundled locally for offline use) shows the COLMAP alignment RealityScan-style: grid floor, sparse point cloud, and **clean wireframe camera frustums** (NOT the old SuperSplat-Gaussian-dots — that read as a confusing blob and was replaced). SuperSplat is still used for the final trained splat in the Results tab.

Data flow: `pipeline/sparse_preview.py:build_alignment_json(sparse_dir, out_json)` parses COLMAP `*.bin` directly and writes `processing/<job>/alignment.json` (`{points, colors, frustums:[apex+4corners]}`). Served at **`/align/<job_id>.json`**; the Process-tab `<iframe id="alignPreview">` loads `static/align/viewer.html?job=<id>` once `preview_ready` is set.

Key details:
- **No pycolmap** — parse the binaries with `struct`. The bundled Python 3.11 has no pycolmap (the old "detailed stats" code also fails with `No module named 'pycolmap'`). Do NOT reintroduce it in the bundled path.
- **Auto-level:** the scene is rotated so the cameras' average up-vector → +Z (COLMAP world has no canonical up), so the subject stands upright.
- **Frustum size** is a fraction (~4.5%) of the **camera spread**, not the point extent (outlier points would inflate it).
- Route `send_file` paths must be **absolute** (`os.path.abspath`) — `os.path.exists` resolves vs cwd but Flask `send_file` resolves vs `app.root_path`; they disagree when the server is launched from a different cwd (caused a 500).
- The Process tab is **full-width + responsive**: side-by-side (stages | preview) on wide windows, **stacked** (stages on top, big preview below) on portrait/narrow. The live log moved to the Logs page.

### Brush trainer (v0.3.0 specifics)

The bundled `brush_app.exe` is **Brush v0.3.0** (ArthurBrussee/brush). Hard-won facts:
- It is a **GUI-subsystem binary** — it writes **nothing to a parent stdout pipe**, so step progress cannot be parsed from stdout. Progress is tracked from the **`export_<iter>.ply` checkpoint files on disk** instead, and the ETA is computed from the measured export rate.
- **`--with-viewer` is a presence flag, not a value flag.** `--with-viewer false` is REJECTED (`unexpected argument 'false'`). The v0.3.0 source auto-disables the viewer when a source path is present, but the bundled build still opens a window — do **not** pass `--with-viewer false`.
- **`--export-name` supports an `{iter}` token** → use `export_{iter}.ply` so each periodic export is numbered and the real step is readable (a fixed name overwrites one unnumbered file).
- The app exports **periodically** (`--export-every = steps//10`) so a timeout/crash still leaves a usable splat; on timeout it **salvages the most-progressed `export_<iter>.ply`** instead of reporting total failure.

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
- `pycolmap` — **optional**, and absent from the Lite bundle (`LITE_REQUIREMENTS` omits it; only `MCMC_REQUIREMENTS`/Full ships it). Everything on the Lite path reads COLMAP `*.bin` with the pure-`struct` readers in `pipeline/sparse_preview.py` (`_read_points3D_bin` / `_read_cameras_bin` / `_read_images_bin`, verified bit-identical to pycolmap on the Mip-360 garden model). pycolmap is only reached for `.txt` reconstructions, the "detailed stats" log line, and the Full-only modules (`learned_sfm`, `depth_seed`, `combine_splats`, `gsplat_mcmc_trainer`). **When adding a Lite-path code path, use the struct readers — do not add a hard pycolmap import.**
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

## Packaging (distributable builds)

`packaging/build_lite_package.py` produces a self-contained Windows bundle in `packaging/dist/`.

- **Lite** (default, ~1.3 GB): bundled **Python 3.11 embed + COLMAP + Brush + offline wheels**. Brush-only — the embedded Python has **no torch**, so selecting MCMC import-fails and falls back to Brush. Launch: `START_SERVER.bat`.
- **Full / MCMC** (`--with-mcmc`, ~4 GB): also downloads **torch (cu126) + gsplat (prebuilt wheel) + MCMC deps** into `App/wheels/`, so the bundled Python runs MCMC natively. `--only-binary :all:` makes the build **fail loudly** if gsplat has no prebuilt wheel for the chosen Python (fix: pin Python 3.10). **MCMC is NVIDIA-only** — even bundled it will not run on AMD/Intel/Mac (gsplat is CUDA). torch's wheel carries the CUDA runtime, so targets need only a recent NVIDIA driver, not the CUDA toolkit.
- **`START_SERVER_MCMC.bat`** (written into the Lite bundle): runs the same app on port 5000 but with the **system CUDA Python** (`C:\Program Files\Python314\python.exe`, torch 2.12.0+cu126), so MCMC works without rebuilding — for a machine that already has torch+gsplat.

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

#### 3.1 `simple_splat/App/pipeline/gsplat_mcmc_trainer.py` ✅
Implements `load_colmap_dataset()` and `train_mcmc()` using gsplat's `MCMCStrategy` and `rasterization()`. Uses `SelectiveAdam` optimizer per parameter group. Hyperparameters follow gsplat's reference `examples/simple_trainer.py` (mcmc preset): per-param LRs, means lr × scene_scale with exponential decay to 1%, opacity/scale L1 regularization (0.01 each — required for MCMC relocation to work), SH degree warmup (one band per 1000 steps).

**Critical:** pycolmap's `image.cam_from_world` is already world-to-camera (w2c) and gsplat's `rasterization(viewmats=...)` expects w2c — pass it through directly, do NOT invert.

**Critical:** optimizers must be built over the same `ParameterDict` entries used in the render graph — building them over pre-copy tensors leaves `grad=None` and `SelectiveAdam` silently skips the update.

Smoke test: `python tests/test_mcmc_smoke.py` (needs CUDA + MSVC on PATH) — trains a synthetic scene and asserts loss collapse, PSNR gain, and MCMC growth.

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

MCMC converges to `cap_max`. When omitted it **auto-scales** to `sparse_pts × 30` clamped to **500k–2M** (`gsplat_mcmc_trainer.py`); app.py passes `config.get('mcmc_cap')` (None → auto). Explicit values still work; on the RTX 8000 (48 GB) the cap can safely go to 2–4M. No UI knob yet (PROJECT.md TODO). Note: the cap is a **ceiling, not a target** — MCMC fills toward it gradually, so a high cap needs enough steps (this 74-photo scene reaches its ~940k auto-cap; 4M only filled by step ~10.5k of a 50k run).

**Dense MVS:** all presets ship `dense: false` (MVS adds nothing for splat training). If enabled via `enable_dense`, with the RTX 8000 the `patch_match_stereo` cache can be raised to 44GB (`--PatchMatchStereo.cache_size 44`) and `fusion_min_pixels` lowered to 1 for maximum density.

**Image count guide:**
- <20 images → use `low` preset (high/medium waste time and produce fewer points due to strict filters)
- 20–50 images → `medium`
- 50–100 images → `high`
- 100+ images → `high` or `quality`

---

## Status Log

**2026-07-03 — Full codebase audit: 11 bugs fixed, ~600 lines of dead code removed (4 commits, b0f4614..ad9dd96).**
**Two long-standing "KNOWN TODOs" are now CLOSED:** (1) the `compress_ply` SH1/SH2 channel-major bug (green tint) — fixed with per-channel band selection R[0:c]/G[in:in+c]/B[2*in:2*in+c], verified with a synthetic R=1/G=2/B=3 splat round-trip; (2) `_load_presets()` had been loading from `pipeline/presets/` (nonexistent) since the 2026-06-14 reorg — **the JSON preset system was silently dead** and every job used the stale inline dict (`quality` preset didn't even exist there → ran as medium). Fixed to `parent.parent / "presets"`, verified all 5 presets load.
Other fixes: batch-cancel poison (queue-global `_stop_event` cancelled every future batch → per-job status check); batch route forced `enable_dense=true`/3200px overrides defeating preset defaults (now None-sentinel); `PatchMatchStereo.cache_size` passed MB where COLMAP expects **GB** (~90,000 GB request); hardcoded MVS `depth_min/max 0-100` truncated big scenes (COLMAP now auto-derives); `/download` + `/ply` could serve an arbitrary rglob hit (Brush checkpoint) instead of the final splat (`_pick_result_ply` prefers gaussian_splat/combined/point_cloud); `recipes._run` timeout never fired on a SILENT hang (clock only checked per output line → detached job stuck forever; now reader-thread + `p.wait(timeout)`); MCMC fallback `_export_3dgs_ply` flattened shN coeff-major (needs channel-major transpose like gsplat's export_splats); camera poses sorted by plain string (frame_10 < frame_2; now natural sort); `export_point_cloud_ply` centered the cloud while poses/splat stay in COLMAP frame (center now defaults False); learned_sfm forced ONE camera from `sizes[0]` onto mixed portrait/landscape sets (now one camera per unique size); resource leaks (PIL handles, cv2 VideoCapture, recipe spawn-log handle).
Dead code removed: unreachable `detail_level_includes_mvs` branch in app.py; duplicate inline COLMAP probe in `run_colmap` (now uses cached `get_colmap_path()`); dead ASCII FBX writer + helpers in camera_tracking (~135 lines); dead `unlimited` mapper/triangulation branches; 15 root `run_*.py` experiment scripts; unused imports/params throughout. Verified: all files byte-compile, app.py imports with 25 routes, FBX smoke round-trip OK, preset loading OK.

**2026-06-27 — Production VERIFIED end-to-end through the UI + FBX matchmove export actually fixed & Blender-verified (it never imported before).**
**PRODUCTION ran clean end-to-end through the frontend** (job `2f3883c2`): learned-SfM → undistort → MCMC + Brush in PARALLEL → `combine_fade(σ=1.7)` → `combined.ply` 624 MB + `gaussian_splat.ply` 162 MB, status `completed`, ~3 hr. Closes the "Production unverified end-to-end" TODO (Draft was already verified).
**FBX matchmove export was BROKEN the whole time** — the prior "tested 58-frame FBX" only ever validated our *own* re-parser; it had never been imported into a real DCC, and Blender's importer **crashed** on it. Fixed four real bugs in `pipeline/fbx_binary.py` + added source-fps matching (all **Blender 4.3 headless-verified**: pos err 0.0000, look·=1.000, up·=1.000, keys on integer frames):
1. **Binary object names** were written ASCII-style `Class::Name` (e.g. `"NodeAttribute::"`) instead of binary `Name\x00\x01Class` → Blender `ValueError: not enough values to unpack`. Fixed via `_objname()` (all 6 object types funnel through it).
2. **Scale 100× too small** — FBX is centimetre-native, Blender is metres → it divided positions by 100 (camera 100× smaller than the splat). Fixed with `UnitScaleFactor=100` (declares metres) → camera imports 1:1 with the splat.
3. **Orientation ~90° off** — we wrote COLMAP-convention poses raw AND declared the file Y-up (Blender rotates 90° on import). Also **FBX cameras aim down local +X**, not −Z. Fixed: declare Blender-native Z-up axes (no reorientation, stays in splat frame) + `_CAM_LOOK_FLIP` quaternion for the **FBX +X-aim** convention (universal across Blender/Maya/Max/Nuke).
4. **Keyframe timing** — keys were authored at 30 fps but the file never declared a rate, so Blender resampled them to its 25 fps grid (fractional frames, mid-clip drift). Fixed via `TimeMode`+`CustomFrameRate` (`_fps_to_timemode()`).
5. **Match the SOURCE VIDEO fps** (user ask): `extract_frames_from_video` now persists `video_info.json` (`source_fps`, `frame_interval`); `camera_tracking.video_frame_timing()` feeds the FBX export the true source fps with keys on **original source-frame numbers** (`i×interval`) → frame-accurate to the plate (degrades to consecutive when interval=1). Used by both the recipe path (`_export_fbx_camera`) and the legacy `/api/camera-tracking` route. Non-video jobs fall back to 30 fps consecutive.
**Files touched (uncommitted):** `pipeline/fbx_binary.py`, `pipeline/camera_tracking.py`, `pipeline/recipes.py`, `app.py`. The FBX is now genuinely binary-FBX-7.4, metre-scale, correctly oriented, source-fps-matched. **GOTCHA learned: validate FBX by importing into a real DCC headless (`blender --background --python-expr ...`), not by re-parsing our own bytes.**

**2026-06-23 (end) — Full productization: Draft VERIFIED end-to-end through the UI + all UI integration fixed.**
**DRAFT ran clean end-to-end through the frontend** (learned-SfM → undistort → gap-fill depth-seed → fast Brush → 830 MB splat, viewable in Results). It even survived **LM Studio GPU contention** (ran ~6× slower but the detached job finished) — proving the detached architecture. Fixes this round (all committed + pushed, origin/master `b760e6d`):
- **Production MCMC + Brush now run in PARALLEL** (threads over two GPU subprocesses, ~28 GB < 48 GB) → ~4 hr → ~2.5 hr. SH3-only compress (no SH1 — its channel bug is moot).
- **Recipe-path UI integration restored** (it bypassed the legacy COLMAP flow): stage timers (recipe reports `feature_extraction/feature_matching/mapping/training` by parsing learned-SfM output; `/status` drives `transition_stage`), **sparse point-cloud preview** (`_build_preview` writes `alignment.json` after undistort; `/status` sets `preview_ready`), and **recipe logs streamed to `/logs`** (the `/status` overlay ingests `recipe.log` incrementally — detached process can't touch the server's in-memory log).
- **FBX matchmove camera WIRED** (was cosmetic): video + toggle → `app.py` appends `--fbx` to the spawn → recipe runs `_export_fbx_camera` (extract poses → `export_camera_fbx`) → `job/camera_tracking/cameras.fbx` → "Download FBX Camera" button in Results. Tested: 58-frame FBX.
- **No more pop-up windows** — all recipe subprocesses spawn with `CREATE_NO_WINDOW` + hidden STARTUPINFO (Brush GUI hidden via SW_HIDE).
- **Hamburger menu z-index** fixed (was behind the preview iframe): `.header` gets a stacking context.
- **`recipes.py` adds APP_DIR to sys.path** so `import pipeline.X` works when run as a detached script (this was silently breaking the preview build).
**GOTCHAS:** preview/test server is EPHEMERAL (tied to Claude tooling) — run real jobs via `python app.py` DIRECTLY. NEVER run a manual GPU job while the user is on the live server (single GPU → two Brush = collision + server death). LM Studio (not Ollama this time) was hogging the GPU — unload its model for fast drafts.
**REMAINING TODO:** (1) `compress_ply` SH1/SH2 channel-major bug (green tint; fix = R[0:c],G[15:15+c],B[30:30+c]; SH3/SH0 fine). (2) Full **Production** run unverified end-to-end (Draft is; Production components individually proven). (3) Brush's GUI-hide (SW_HIDE) untested on a real run — if a draft's Brush stalls, un-hide just Brush.

**2026-06-23 (later) — Recipes wired into backend (Draft/Production) + detached jobs + gap-fill depth-seed.**
**Verdict (user's eye):** PRODUCTION = the champion `COMBINE_gauss_sigma17` (learned-SfM → undistort → MCMC + Brush + combine σ=1.7 + compress, ~4 hr). The gap-fill dense seed is sharper on asphalt + keeps a clean building, but "too sharp" — kept for exploration. DRAFT = **gap-fill dense seed → ONE fast Brush pass** (1280 res, 8k steps, no combine, ~10 min; learned-SfM ~5 min is the bottleneck).
**`pipeline/recipes.py`** — `run_draft()` / `run_production()` orchestrators driving the validated pipelines via SUBPROCESSES (torch/transformers load in children, not Flask). `app.py` `process_images_async` early-branches on `advanced_settings['quality']` ∈ {draft, production} → spawns recipes.py as a **DETACHED process** (Windows `DETACHED_PROCESS`) that writes `recipe_status.json` + the final PLY to the job dir. `/status` overlays that file. **So a server restart/crash never orphans a job** (fixes the orphaning hit during testing). Legacy COLMAP-SIFT flow stays as fallback.
**`pipeline/depth_seed.py`** gained `--gap-mult` (gap-fill: drop depth within `gap_mult × sparse_spacing` of an SfM point so already-good surfaces like the building keep clean SfM geometry; depth only fills textureless gaps) + `build_dense_seed_dir()` (depth-seed → COLMAP dense dir, reusable).
**KNOWN TODO:** (1) `compress_ply` SH1/SH2 partial-band selection is BUGGY — it keeps `f_rest_0..8` = all-RED (SH is channel-major R0-14/G15-29/B30-44) → green tint on SH1 files; fix = pick R[0:c],G[15:15+c],B[30:30+c]. SH3/SH0 are correct. (2) FBX matchmove backend. (3) Production recipe is wired but its full 4-hr run is unverified end-to-end (components individually proven). **Run real jobs via `python app.py` directly, NOT the preview/test server.**

**2026-06-23 — Frontend redesign + compression + auto-sigma + Mip-360 generalization + depth-seed.**
**UI (`templates/index.html`) restructured to input-driven** (verified live via preview): drop files → auto-route. **1 image → ML-Sharp automatically**; **image sequence/ZIP → Draft / Production** two-card toggle (Draft = Brush/medium, Production = MCMC/high — interim until the combine pipeline is wired into the app); **video → frame-sampling + "Export FBX matchmove camera"** toggle (forces interval=1, caps ~30s/300 frames). The Draft/Production cards **drive the old advanced fields**, which are now an **inert hidden block** (the **Settings tab was removed** from the nav — controls preserved as hidden defaults so the backend contract is unchanged). **Header actions (Open Splat/Logs/Stop/Clean) collapsed into a hamburger menu.** **Capture-tips row removed → preserved in [docs/capture-tips.md](docs/capture-tips.md)** for a future user guide. NOTE still TODO: FBX matchmove backend (read `export_fbx`, trigger camera-tracking per-frame, enforce cap) + Production=combine orchestration.
**`pipeline/compress_ply.py`** — pure-numpy PlayCanvas/SuperSplat compressed-PLY writer (reimplemented from our own MIT `supersplat-src/splat-serialize.ts`, NOT Sony's patented coder). 3.9x@SH3, 9.3x@SH1, 14.5x@SH0; round-trip verified (pos err ~0%). CLI: `python pipeline/compress_ply.py in.ply out.ply --max-sh 1`; `--verify` mode.
**`combine_fade` default flipped → `fade_mode='gaussian', sigma_mult=1.7`** (proven winner across parking-lot/garden/room by eye; `sigma_mult='auto'` Otsu-valley mode added but looked LOOSER than the eye prefers — kept as an option only).
**Generalization tested on Mip-NeRF 360** (garden + room): the combine is a general textured-vs-void splitter — it never no-ops (indoor walls/ceiling are as seedless as outdoor sky).
**`pipeline/depth_seed.py`** (Sony-inspired) — Depth-Anything-V2 → affine-align to sparse anchors → back-project → dense seed (parking lot 21k→1.07M). RAW POINTS are clean (flat asphalt, upright building); the earlier "bowl" was a preview-rendering artifact, not the geometry. Built `processing/scene-dense` (dense COLMAP points3D.bin) and an MCMC A/B is mid-run. **`datasets/` (Mip-360, 12.5 GB) is gitignored.**

**2026-06-20 — Sky model (Postshot-style) + the hybrid-combine journey.** Spent the session converging the 58-image parking-lot scene toward a commercial Postshot result. Built a **trainer-combine** pipeline: `pipeline/combine_splats.py` merges two splats trained on the SAME seed by spatial role (MCMC sharp near seeds = real surfaces; Brush clean far from seeds = sky), with a smooth **density crossfade** (`combine_fade`) and per-component **needle + white-bloom prune** for the Brush sky. Best result = `BEST_combine_2.74M.ply` (crossfade, preserved in Downloads). **A user screenshot of Postshot's training panel** revealed it uses the same algos (Splat3/MCMC/ADC, anti-aliasing, SH3, 30k, 3M cap) — the one missing piece was **"Create Sky Model"**. Implemented it: `pipeline/sky_model.py` (MLP: ray-dir→RGB) composited behind the Gaussians via alpha, wired into `train_mcmc(use_sky_model=True)` — keeps the foreground clean (no sky floaters, coherent thin structures like flagpoles) without sky-masking. **Hard-won lessons:** trust the EYE over metrics (chasing count/p99 made it visually worse); needles aren't all bad (poles/edges need them — the `aniso_reg` regularizer BACKFIRED, leave it 0); Brush stays (its sky is excellent). `sky_mask.py` is self-gating (no-ops indoors). **Next: verify the sky-model MCMC + Brush combine vs the 2.74M baseline.**

**2026-06-19 — Dense-seed workflow proven (learned matcher → anti-flicker MCMC).** On the 58-image parking-lot scene, COLMAP SIFT seeds only **3,651** points (textureless). `pipeline/learned_sfm.py` (SuperPoint+LightGlue, `--max-kpts 4096`, **default** pycolmap mapper) seeds **21,831 (6×)**, all 1653/1653 pairs verified. **Density levers that BACKFIRED** (recorded so we don't repeat): 8192 keypoints → 18,538 (weak extra points, BA deletes); loosening the mapper (keep 2-view tracks / lower min-angle / higher reproj) → 18,609; ALIKED extractor → 11,434. SuperPoint@4096 + default mapper is the optimum; RealityScan's 41,121 is diminishing returns for a splat seed (trainer densifies to millions anyway). Pipeline: learned seed → `colmap image_undistorter` (SIMPLE_RADIAL→PINHOLE, all points survive) → `train_mcmc(cap_max=4M, steps=30000, use_lpips=True, export_opacity_min=0.03)` → fills 4M, anti-flicker prune → **888,026 Gaussians, 200 MB, peak PSNR ~26.8** (vs ~24 on the thin seed), healthy soft opacity (median 0.247). `learned_sfm` gained `extractor`/mapper params + an ALIKED/DISK option. **NOTE:** launch GPU jobs as the DIRECT background process — an inner `python … &` inside a wrapper orphans + SIGHUP-kills it. **Next: run Brush on the same 21k seed for an A/B.**

**2026-06-17 — Learned-SfM front-end + viewer-flicker fix + FBX/`/ply` fixes.** Prompted by a 4-way A/B on a 58-image parking-lot scene (COLMAP SIFT seed only **3,325 pts** — textureless) vs Postshot (4.37M). Findings & work: (1) **MCMC viewer flicker** traced to sub-pixel Gaussians + a huge near-transparent "filler" pool (median opacity 0.013) — fixed with `rasterize_mode="antialiased"` + export **opacity-prune** (drop sigmoid(op)≤`export_opacity_min`=0.03) + **min-scale floor**; on the 4M run this pruned **4,000,000→956,344** (76% filler), 215 MB vs 900 MB. (2) **`pipeline/learned_sfm.py`** — SuperPoint+LightGlue→pycolmap DB (geometric verification via `estimate_two_view_geometry`, note `pycolmap.Database.open(path)` not the ctor) → `incremental_mapping` → `sparse/0`; deps `lightglue`(cvg git)+`kornia` in Py 3.14; validated 12img→3,484 pts (already beats COLMAP's 58-img 3,325). Run on CPU by default to avoid fighting a GPU training job. (3) **Binary FBX 7.4** (`pipeline/fbx_binary.py`) replaced ASCII (Blender rejects ASCII); Blender export now a **ZIP** (script+json). (4) **`/ply` 500** fixed: data folders anchored to app dir (cwd-independent). **Next: rerun the pipeline on the learned-SfM dense seed and retrain MCMC v2.**

**2026-06-15 — MCMC ADC mode + GPU-aware cap + Gaussian-budget UI.** Prompted by a same-dataset A/B: a commercial tool (`3698a389.ply`, **4,372,882 splats**, Postshot/adaptive-densification) vs our Brush 15k (`brush_15k.ply`, **110,932 splats**) — a ~40× Gaussian-count gap (under-densified + under-trained; MVS is irrelevant, the count comes from the trainer, not the init cloud). `gsplat_mcmc_trainer.train_mcmc` now takes `strategy_name='mcmc'|'adc'`: **ADC** uses gsplat `DefaultStrategy` (organic clone/split/prune, **no cap**, MCMC opacity/scale regularizers turned OFF, calls `step_pre_backward` before `loss.backward()` and `step_post_backward(..., packed=...)` after, `initialize_state(scene_scale=...)`). The MCMC auto-cap is now **GPU-bounded** (`torch.cuda.mem_get_info`, ~1.5 KB/Gaussian, ×0.7 safety, ≤8M) instead of a flat 2M ceiling. UI: third trainer card **MCMC ADC** + a **Gaussian Budget** select (auto/1M/2M/4M/8M) → `gaussian_budget` form field → `advanced_settings['mcmc_cap']` → `train_mcmc(cap_max=...)`. Verified: both files compile, server boots, the 3 trainer cards + budget select render with no console errors. **The training itself is UNTESTED — needs a real CUDA run** (the `DefaultStrategy` API points above are the unverified risk).

**2026-06-12 — MCMC trainer fixed and verified.** The trainer had been broken since inception: (1) c2w passed where gsplat expects w2c, (2) optimizers built over dead tensor copies (`grad=None`, silent no-op), (3) hand-rolled avg-pool SSIM with corrupt border gradients, (4) `export_splats` given activated values where the PLY format stores raw log/logit (splat rendered as giant blobs in viewers). All fixed; hyperparameters aligned to gsplat's reference trainer; LPIPS computed on a 512px crop. Verified: synthetic smoke test 13.4 → 39 dB PSNR, real 74-photo job 25.5 dB mean PSNR, 1M Gaussians, ~13 min training.

**2026-06-12 — GUI redesigned + rebrand.** App renamed **FonixFlow Splat** (directory name `simple_splat/` intentionally unchanged). UI is now tabbed (Create / Settings / Process / Results), no scrolling, auto-switches tabs as the job advances. All element IDs the JS depends on were preserved. Dev preview config in `.claude/launch.json` (name `fonixflow-splat`, port 5000, `autoPort: false` because app.py hardcodes the port).

**2026-06-13 — Deep audit, Brush v0.3.0 hardening, MCMC auto-cap, Full packaging.** Fixed 8 `app.py` bugs (image-count sampling that dropped 74→15 photos, ZIP nested-folder flatten, preset-downgrade `None` sentinels, Brush PLY size validation, filename collision dedup). Wired `GlobalMapper.*` preset params in `run_glomap.py` (every preset had been producing identical reconstructions). Brush: periodic **numbered** exports (`export_{iter}.ply`), **filesystem** progress + measured-rate ETA, **timeout salvage**; removed invalid `--with-viewer false`. Added a desktop **notification + beep** on job completion (`index.html`). MCMC `cap_max` now **auto-scales** to sparse-point count. `build_lite_package.py --with-mcmc` bundles torch+gsplat (**Full**, NVIDIA-only, ~4 GB); `START_SERVER_MCMC.bat` runs the app under the system CUDA Python. Verified live: MCMC on the real 74-photo scene grew 31k→437k Gaussians (7k steps) and filled a 4M cap by step ~10.5k.

**2026-06-14 — Brush ETA fix, MCMC verified + auto-cap, COLMAP HQ flags, RealityScan A/B, 3D alignment preview.** Brush v0.3.0 facts pinned: GUI-subsystem binary (no stdout, opens a viewer window), `--with-viewer false` is REJECTED, and it overwrites a single export unless `--export-name` carries `{iter}` → switched to numbered `export_{iter}.ply`, filesystem progress, ETA calibrated from export events. **MCMC ran for real for the first time** on the user's box (**Quadro RTX 8000, 48 GB**; torch **2.12.0+cu126** at **`C:\Program Files\Python314\python.exe`**): a 7k-step run reached ~25→30 dB and grew 31k→437k Gaussians; a 50k/4M-cap run filled exactly 4,000,000 Gaussians but looked **worse** (overfit/haze — 4M is ~4× too many for 74 photos, confirming auto-cap `sparse×30` clamped 0.5–2M is the right default). **COLMAP HQ flags (#1–4)** added for high+. **RealityScan** (Epic desktop, via the launcher) aligned the same 74 photos 74/74 with a **denser** sparse cloud (53,612 vs 31,373); its COLMAP export is FULL_OPENCV without images, its Radiance-Fields export (`transforms.json`, SIMPLE_RADIAL, references originals) is **Brush-usable** but MCMC needs a converter. **3D alignment preview** built: `sparse_preview.py` (pure-binary COLMAP parser, no pycolmap) → SuperSplat embedded in the Process tab. Tooling note: the Bash **safety classifier was down the whole session** (Auto Mode only) — exact-allowlisted commands + git still ran; fix is to switch off Auto Mode (manual approve) so commands prompt instead of being auto-vetted.

**2026-06-14 (cont.) — Repo reorg, Three.js alignment viewer, stage-timer fixes, repackage.** Moved processing modules into `simple_splat/App/pipeline/` (package; `app.py` imports `from pipeline.X`), tests into `tests/`, all docs into `docs/`. Removed 23 root scratch scripts; deleted 62 GB of gitignored data (`test_data/`, `mipnerf360.zip`, stray `Brush/`). **Replaced the SuperSplat-dots alignment preview with a dedicated Three.js viewer** (`static/align/viewer.html`, three.js bundled): grid + point cloud + clean wireframe camera frustums, **auto-leveled** to the cameras' average up-vector, frustums sized to camera spread. New `build_alignment_json` + `/align/<job_id>.json` route. **Stage timers fixed**: `transition_stage` no longer resets a re-triggered stage (Feature Matching was showing 0.0s), server computes live elapsed, freezes on completion; **Dense MVS stage removed** from the UI (always sparse-only). Process tab is now **full-width + responsive** (stacks vertically on portrait). Fixed a `/align` **HTTP 500** (abspath: `os.path.exists` vs Flask `send_file` cwd mismatch). **Repackaged the Lite dist** from the reorganized source — booted + verified (HTTP 200). Bash classifier outage worked around by switching **off Auto Mode** (manual approve). Best way to run on this box: `cd simple_splat/App && python app.py` from **Git Bash** (system Python 3.14 has torch+gsplat+pycolmap+flask+cv2; COLMAP at `C:\COLMAP`, Brush at `C:\Brush`).

**Known next steps** (see PROJECT.md): expose `mcmc_cap` in UI; **Phase 3 of the preview** (a slider to scrub training checkpoints and watch densification); run **HQ COLMAP** on the 74 photos and compare the sparse count vs 31,373 / RealityScan 53,612; write **`rs_to_colmap.py`** (transforms.json → pinhole COLMAP + undistorted images) so MCMC can train RealityScan poses, then the **4-way A/B** (brush/mcmc × colmap/realityscan); **learning-based matcher** (LightGlue/MASt3R/VGGT) as the real RealityScan gap-closer; **MCMC ADC mode** (gsplat `DefaultStrategy`, INRIA-style) which may beat MCMC visually; antialiased rasterization + bilateral grid options.
