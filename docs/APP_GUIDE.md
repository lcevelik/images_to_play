# FonixFlow Splat — Standalone Edition

A web application that converts images or video into 3D Gaussian Splats using COLMAP and Brush, with an integrated browser-based viewer powered by PlayCanvas SuperSplat.

This is a **fully standalone package** — Python, COLMAP, Brush, all Python dependencies, and the 3D viewer are all bundled. No internet connection or external installs required.

---

## Quick Start

1. Navigate to the root of the package folder
2. Double-click **`START_SERVER.bat`**
3. Open your browser at **http://localhost:5000**
4. Upload images and click **Start Processing**

---

## Features

- **Tabbed UI**: Create / Settings / Process / Results — auto-switches as the job advances
- **Image Upload**: Individual images (JPG/PNG), video files (MP4/MOV/AVI/MKV/WEBM), or ZIP archives
- **Video Frame Extraction**: Automatically extracts frames from video at configurable intervals
- **5 Quality Presets**: Fast (5K steps), Balanced (15K), High (30K), Quality (60K), Expert (100K)
- **Quality Scale**: Draft (0.3×) / Standard (1×) / Cinematic (2×) multiplier on training steps
- **Two Trainers**: Brush (standard 3DGS, default) or gsplat MCMC (auto Gaussian cap that scales to the scene, LPIPS loss, often higher quality — NVIDIA only)
- **Dense Reconstruction**: Optional COLMAP MVS for millions of points (off by default; toggle via upload form)
- **Camera Tracking Export**: FBX / GLTF / JSON / Blender script camera paths from the results panel
- **3D Alignment Preview**: the Process tab shows the sparse point cloud + camera frustums in 3D (RealityScan-style) as soon as COLMAP finishes, then training continues
- **Browser Viewer**: PlayCanvas SuperSplat (WebGPU-accelerated, no install needed)
- **Open Splat File**: Upload an existing `.ply` or `.splat` file for direct viewing
- **Download**: Export the trained `.ply` file or sparse reconstruction `.zip`
- **Real-time Logs**: Live processing log viewer at http://localhost:5000/logs
- **ML-Sharp** (optional): Single-image ultra-fast processing via Apple's ml-sharp (not bundled)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (64-bit) |
| RAM | 8GB minimum, 16GB+ recommended |
| GPU (Processing) | NVIDIA GPU with CUDA (strongly recommended) |
| GPU (Viewer) | Any GPU supporting WebGPU (NVIDIA RTX, AMD RX 5000+) |
| Browser | Chrome 113+ or Edge 113+ for the viewer |
| Disk | 10GB+ free space |

The 3D viewer requires WebGPU. Use Chrome or Edge 113+. You can verify at `chrome://gpu`.

---

## Processing Pipeline

```
1. Image Upload (or Video Frame Extraction)
        |
2. COLMAP Feature Extraction (SIFT, GPU-accelerated)
        |
3. Feature Matching (exhaustive or sequential)
        |
4. Sparse Reconstruction (COLMAP mapper)
        |
5. Dense MVS Reconstruction (optional, off by default)
   - Patch Match Stereo -> depth maps
   - Stereo Fusion -> dense point cloud (fused.ply)
        |
6. Gaussian Splat Training (user choice)
   - Brush: standard 3DGS clone/split densification
   - gsplat MCMC: stochastic relocation, auto Gaussian cap (scales to sparse points), LPIPS loss (NVIDIA only)
   - Exports gaussian_splat.ply
        |
7. View in Browser (PlayCanvas SuperSplat, WebGPU)
```

---

## Quality Presets

| Preset | Training Steps | Dense MVS | Est. Time |
|--------|---------------|-----------|-----------|
| Fast (`low`) | 5,000 | No | ~2-4 min |
| Balanced (`medium`) | 15,000 | No | ~8-15 min |
| High (`high`) | 30,000 | No | ~15-30 min |
| Quality (`quality`) | 60,000 | No | ~30-60 min |
| Expert (`expert`) | 100,000 | No | full lens calibration, longest |

Preset COLMAP parameters live in `presets/*.json` — edit the JSON to tune a preset, no code changes needed. Dense MVS is off by default for all presets (no benefit for splat training); enable it via the `enable_dense` upload field. The **Quality Scale** setting (Draft 0.3× / Standard 1× / Cinematic 2×) multiplies the preset's training steps.

---

## File Structure

```
GaussianSplatting_Standalone\
+-- START_SERVER.bat               <- Start the server (double-click this)
+-- README.txt                     <- User quick-start guide
+-- Python\                        <- Portable Python 3.11 (no install needed)
+-- COLMAP\
|   +-- bin\colmap.exe             <- COLMAP reconstruction engine
|   +-- lib\*.dll                  <- COLMAP runtime libraries
+-- Brush\
|   +-- brush_app.exe              <- Brush trainer (download separately - see below)
+-- App\
    +-- app.py                     <- Flask web server (main entry point)
    +-- run_glomap.py              <- COLMAP pipeline runner
    +-- dense_reconstruction.py    <- Dense MVS module
    +-- gaussian_splat_utils.py    <- PLY generation utilities
    +-- gsplat_mcmc_trainer.py     <- gsplat MCMC trainer (alternative to Brush)
    +-- test_mcmc_smoke.py         <- MCMC trainer smoke test (synthetic scene, needs CUDA)
    +-- camera_tracking.py         <- Camera path export (FBX/GLTF/JSON/Blender)
    +-- batch_processing.py        <- Batch job queue
    +-- presets\*.json             <- Quality preset definitions (editable)
    +-- requirements.txt           <- Python dependency list
    +-- README.md                  <- This file
    +-- SETUP.md                   <- Troubleshooting guide
    +-- wheels\                    <- Pre-downloaded pip wheels (offline install)
    |   +-- flask-3.0.0-*.whl
    |   +-- werkzeug-3.0.1-*.whl
    |   +-- flask_cors-6.0.2-*.whl
    |   +-- opencv_python-4.13.0.92-*.whl
    |   +-- psutil-7.2.2-*.whl
    |   +-- pycolmap-3.13.0-*.whl
    |   +-- (+ 7 transitive dependency wheels)
    +-- templates\
    |   +-- index.html             <- Main upload/processing UI
    |   +-- logs.html              <- Live log viewer
    |   +-- viewer.html            <- Legacy viewer (unused)
    +-- static\
        +-- supersplat\
            +-- index.html         <- SuperSplat viewer (WebGPU)
            +-- index.js           <- PlayCanvas engine + viewer (2.4MB, self-contained)
            +-- index.css          <- Viewer styles
            +-- settings.json      <- Default camera and scene settings
            +-- webxr-profiles\    <- VR controller profiles (offline, 123 files)
```

Processing creates temporary folders:
```
    +-- processing\<job-id>\
    |   +-- images\                <- Input images
    |   +-- sparse\0\              <- COLMAP sparse reconstruction
    |   +-- dense\                 <- Dense point cloud (if enabled)
    |   +-- gaussian_splat.ply     <- Final trained splat
    +-- uploads\                   <- Temporary upload storage (auto-cleaned)
```

---

## Python Dependencies

Installed offline from bundled wheels on first run. No internet required.

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework |
| Werkzeug | 3.0.1 | WSGI utilities |
| Flask-CORS | 6.0.2 | Cross-origin requests |
| opencv-python | 4.13.0.92 | Video frame extraction |
| psutil | 7.2.2 | Process management (kill jobs) |
| pycolmap | 3.13.0 | Read COLMAP reconstruction stats |
| numpy | 2.4.2 | Numerical operations (pycolmap dep) |
| + transitive deps | | blinker, click, itsdangerous, jinja2, markupsafe, colorama |

**Additional requirements for the gsplat MCMC trainer** (optional — Brush works without these):

| Package | Notes |
|---------|-------|
| torch (CUDA build) | Must be `+cu126`, not `+cpu` |
| gsplat | JIT-compiles CUDA kernels on first use — needs MSVC `cl.exe` on PATH (app.py auto-discovers it at startup); first compile ~10 min, cached afterwards |
| pytorch-msssim | Gaussian-windowed SSIM loss (do NOT substitute a hand-rolled avg-pool SSIM — it destroys training) |
| lpips | Optional perceptual loss (used for all presets except Fast) |
| scipy, plyfile | Init scales / PLY handling |

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main web UI |
| `/upload` | POST | Upload images/video/ZIP, start processing |
| `/status/<job_id>` | GET | Get job processing status and progress |
| `/ply/<job_id>` | GET | Serve PLY file for the viewer |
| `/download/<job_id>/ply` | GET | Download PLY file (as attachment) |
| `/download/<job_id>/sparse` | GET | Download sparse reconstruction ZIP |
| `/view/<job_id>` | GET | Redirect to SuperSplat viewer |
| `/upload-for-view` | POST | Upload existing .ply/.splat for direct viewing |
| `/logs` | GET | Live log viewer page |
| `/logs/stream` | GET | Server-Sent Events log stream |
| `/logs/json/<job_id>` | GET | Per-job log entries as JSON |
| `/api/camera-tracking` | GET | Export camera path (params: job_id, format, fps, pointcloud) |
| `/camera-tracking-info` | GET | Camera tracking availability for a job |
| `/download/<job_id>/tracking/<file>` | GET | Download exported tracking files |
| `/gpu-info` | GET | GPU detection for time estimates |
| `/mlsharp-info` | GET | ML-Sharp installation status |
| `/kill` | POST | Kill running COLMAP/Brush processes |
| `/cleanup` | POST | Delete all processing job folders |

---

## 3D Viewer (PlayCanvas SuperSplat)

The viewer is **PlayCanvas SuperSplat v1.15.0**, powered by PlayCanvas Engine v2.16.1.

- Accessed at: `http://localhost:5000/static/supersplat/index.html?load=/ply/<job_id>.ply`
- Loads `.ply` Gaussian splat files via the `?load=` URL parameter (also supports `?focal=x,y,z`, `?angles=az,elev`, `?distance=n`)
- Requires a WebGPU-capable browser (Chrome/Edge 113+)
- Fully offline — all viewer assets are bundled locally, including WebXR controller profiles
- Default camera: 2 metres from origin, facing center (configurable in `static/supersplat/settings.json`)

---

## Optional: ML-Sharp (Single-Image Processing)

ML-Sharp enables ultra-fast 3D Gaussian splat generation from a single image using Apple's transformer model. It is **not bundled** — install manually if needed.

```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt
sharp --version   # verify
```

- First run downloads ~500MB model checkpoint (one-time)
- Restart the server after installing — it auto-detects `sharp` in PATH
- GPU recommended but CPU fallback is available

---

## Downloading Brush

Brush is the Gaussian splat training engine. It is **not bundled** (the binary exceeds GitHub's file size limit) and must be downloaded once:

1. Visit **https://github.com/ArthurBrussee/brush/releases/latest**
2. Download the Windows build (`brush_app.exe`)
3. Create a folder named `Brush\` in the package root (next to `START_SERVER.bat`)
4. Place `brush_app.exe` inside `Brush\`

Without Brush the pipeline still completes — it falls back to exporting a basic COLMAP sparse point cloud instead of a trained Gaussian splat.

---

## Troubleshooting

**Viewer shows blank/white screen**
- Use Chrome 113+ or Edge 113+
- Your GPU must support WebGPU. Check `chrome://gpu`

**COLMAP not found / reconstruction failed**
- Bundled COLMAP is at `COLMAP\bin\colmap.exe`
- Ensure `START_SERVER.bat` is used (it sets up PATH automatically)
- If moved the package, ensure DLLs in `COLMAP\lib\` are alongside the binary

**Brush not found / no PLY generated**
- Download `brush_app.exe` from https://github.com/ArthurBrussee/brush/releases/latest
- Place it in `Brush\` folder next to `START_SERVER.bat` — see "Downloading Brush" above
- If Brush is missing or fails, the app falls back to a basic COLMAP sparse point cloud
- Check real-time logs at http://localhost:5000/logs for details

**COLMAP failed to reconstruct**
- Images need 60-80% overlap between consecutive shots
- Minimum ~10-20 sharp, well-lit images of the same subject
- Avoid blurry, reflective, or textureless surfaces

**Out of memory**
- Use Low preset
- Reduce image count (50-100 images is usually enough)
- Close other GPU-heavy applications

**Processing stuck / hung**
- Use the Kill button in the web UI, or visit `http://localhost:5000/kill` (POST)
- This terminates COLMAP and Brush processes and cancels the job

---

## Acknowledgements

- [COLMAP](https://github.com/colmap/colmap) - Structure-from-Motion and MVS pipeline
- [Brush](https://github.com/ArthurBrussee/brush) - 3D Gaussian Splatting trainer (Vulkan/WGPU)
- [gsplat](https://github.com/nerfstudio-project/gsplat) - CUDA rasterizer + MCMC strategy used by the built-in trainer
- [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc) - MCMC densification strategy (NeurIPS 2024)
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat) - Browser-based 3DGS viewer
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original research paper
