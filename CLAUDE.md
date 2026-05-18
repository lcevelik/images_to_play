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
