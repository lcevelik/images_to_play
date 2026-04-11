# Gaussian Splatting App - Standalone Edition

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

- **Image Upload**: Individual images (JPG/PNG), video files (MP4/MOV/AVI/MKV/WEBM), or ZIP archives
- **Video Frame Extraction**: Automatically extracts frames from video at configurable intervals
- **3 Quality Presets**: Low (fast preview), Medium (balanced), High (maximum quality)
- **Advanced Controls**: Fine-tune training steps, MVS quality mode, resolution, dense reconstruction
- **Dense Reconstruction**: Optional COLMAP MVS for millions of points (medium/high presets)
- **Gaussian Splat Training**: Brush MCMC-based trainer for high-quality `.ply` output
- **Browser Viewer**: PlayCanvas SuperSplat v1.15.0 (WebGPU-accelerated, no install needed)
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
5. Dense MVS Reconstruction (optional, medium/high presets)
   - Patch Match Stereo -> depth maps
   - Stereo Fusion -> dense point cloud (fused.ply)
        |
6. Gaussian Splat Training (Brush)
   - MCMC-based 3DGS training
   - Exports gaussian_splat.ply
        |
7. View in Browser (PlayCanvas SuperSplat, WebGPU)
```

---

## Quality Presets

| Preset | Training Steps | Dense MVS | Max Resolution | Est. Time |
|--------|---------------|-----------|----------------|-----------|
| Low | 3,000 | No | 1600px | 2-5 min |
| Medium | 10,000 | Yes (balanced) | 3200px | 10-30 min |
| High | 30,000 | Yes (quality mode) | 4800px | 30-90 min |

**Advanced settings** allow full control: up to 200,000 training steps, 6400px resolution, ultra sharpness MVS mode.

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
| `/gpu-info` | GET | GPU detection for time estimates |
| `/mlsharp-info` | GET | ML-Sharp installation status |
| `/kill` | POST | Kill running COLMAP/Brush processes |
| `/cleanup` | POST | Delete all processing job folders |

---

## 3D Viewer (PlayCanvas SuperSplat)

The viewer is **PlayCanvas SuperSplat v1.15.0**, powered by PlayCanvas Engine v2.16.1.

- Accessed at: `http://localhost:5000/static/supersplat/index.html?content=/ply/<job_id>&webgpu`
- Loads `.ply` Gaussian splat files via the `?content=` URL parameter
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
- [Brush](https://github.com/ArthurBrussee/brush) - MCMC-based 3D Gaussian Splatting trainer
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat) - Browser-based 3DGS viewer
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original research paper
