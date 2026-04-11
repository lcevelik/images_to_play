# Gaussian Splatting App - Setup & Troubleshooting

This is a **fully standalone** package. Everything is bundled — Python, COLMAP, Brush, all pip dependencies, and the 3D viewer. No external software installation or internet connection required.

---

## Starting the App

1. Double-click **`START_SERVER.bat`** in the root folder
2. A console window opens and Python packages install from bundled wheels (first run only, ~30 seconds)
3. Open your browser at **http://localhost:5000**

No pip install, no Python install, no COLMAP install needed.

---

## One-Time Download: Brush

Brush is the Gaussian splat trainer. It is **not bundled** (too large for distribution) and must be downloaded once:

1. Go to **https://github.com/ArthurBrussee/brush/releases/latest**
2. Download the Windows release (`brush_app.exe`)
3. Create a folder named **`Brush\`** in the root of this package (next to `START_SERVER.bat`)
4. Place `brush_app.exe` inside `Brush\`

Expected result:
```
GaussianSplatting_Standalone\
+-- Brush\
|   +-- brush_app.exe    <- place it here
+-- START_SERVER.bat
+-- QUICK_START.bat
+-- ...
```

Without Brush the pipeline still runs — it falls back to exporting a basic COLMAP sparse point cloud (PLY). You will still be able to view it in the browser, but it will not be a trained Gaussian splat.

---

## What Gets Installed on First Run

`START_SERVER.bat` installs pip packages **offline** from pre-downloaded wheels in `App\wheels\`:

| Package | Version |
|---------|---------|
| Flask | 3.0.0 |
| Werkzeug | 3.0.1 |
| Flask-CORS | 6.0.2 |
| opencv-python | 4.13.0.92 |
| psutil | 7.2.2 |
| pycolmap | 3.13.0 |
| numpy | 2.4.2 |
| blinker | 1.9.0 |
| click | 8.3.1 |
| itsdangerous | 2.2.0 |
| jinja2 | 3.1.6 |
| markupsafe | 3.0.3 |
| colorama | 0.4.6 |

After first-run install, a `.deps_installed` marker file is created in `App\` so subsequent starts skip installation and launch immediately.

To force reinstall (e.g. after updating wheels): delete `App\.deps_installed` and restart.

---

## Bundled Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Python 3.11 | `Python\python.exe` | Runtime (portable, no system install) |
| COLMAP | `COLMAP\bin\colmap.exe` | 3D reconstruction (Structure-from-Motion + MVS) |
| COLMAP libs | `COLMAP\lib\*.dll` | COLMAP runtime DLLs (CUDA, Qt, etc.) |
| Brush | `Brush\brush_app.exe` | Gaussian splat trainer (Vulkan/WGPU) |
| Flask app | `App\app.py` | Web server |
| SuperSplat viewer | `App\static\supersplat\` | Browser-based 3DGS viewer (WebGPU) |
| WebXR profiles | `App\static\supersplat\webxr-profiles\` | VR controller profiles (offline) |
| Python wheels | `App\wheels\` | Offline pip packages |

---

## Browser Requirements (for the Viewer)

The 3D viewer uses **WebGPU** for hardware-accelerated Gaussian splat rendering.

- **Chrome 113+** or **Edge 113+** — recommended, best support
- **Firefox** — enable at `about:config` → `dom.webgpu.enabled = true`

To verify WebGPU works in Chrome: visit `chrome://gpu` and check for "WebGPU: Enabled".

If the viewer shows a white screen or fails to load:
- Switch to Chrome or Edge
- Update your GPU drivers
- Check that your GPU supports WebGPU (NVIDIA RTX, AMD RX 5000+, Intel Arc)

---

## Sharing / Moving the Package

This package is fully self-contained and self-relocating:

- `START_SERVER.bat` uses `%~dp0` (always the absolute path of the .bat file itself)
- All internal paths are derived from that root — the package works from any location
- Copy or move the entire folder to USB, cloud storage, another PC, etc.
- Works on any Windows 10/11 64-bit machine with a WebGPU-capable browser

---

## Troubleshooting

### Server won't start

- Run `START_SERVER.bat` and read the console output for errors
- Check antivirus isn't blocking `python.exe` or `colmap.exe`
- Ensure `.deps_installed` was created (if missing, first-run install may have failed)
- Try deleting `App\.deps_installed` to trigger a fresh package install

### COLMAP not found / reconstruction fails

- The bundled COLMAP is at `COLMAP\bin\colmap.exe` — do not move it
- `START_SERVER.bat` adds `COLMAP\bin` and `COLMAP\lib` to PATH automatically
- If moved the package, restart via `START_SERVER.bat` (not `python app.py` directly)

**Reconstruction fails (no images registered):**
- Images need 60-80% overlap between consecutive shots
- Use at least 10-20 sharp, well-lit images of the same subject
- Avoid: blurry photos, reflective/transparent surfaces, plain untextured backgrounds
- Try more photos from multiple angles around the subject

### Brush not found / no PLY output

- Bundled Brush is at `Brush\brush_app.exe`
- `START_SERVER.bat` adds `Brush\` to PATH automatically
- If Brush fails, the app automatically falls back to a sparse COLMAP point cloud PLY
- Check real-time logs at http://localhost:5000/logs for error details

### Viewer shows blank/white screen

- Use Chrome 113+ or Edge 113+
- Verify WebGPU: visit `chrome://gpu` → look for "WebGPU: Enabled"
- Update GPU drivers (NVIDIA/AMD)
- WebGPU requires a dedicated GPU (not integrated graphics on older machines)

### Out of memory

- Use **Low** preset (sparse only, 3K training steps)
- Reduce image count (50-100 images is usually sufficient)
- Close other GPU-intensive applications
- For VRAM limits: Lower preset = lower VRAM usage

### Processing stuck or hung

- Click the **Kill** button in the web UI
- Or POST to http://localhost:5000/kill — terminates COLMAP and Brush processes
- Check logs at http://localhost:5000/logs for what stage it's stuck on

### Want to clear old job data

- Click **Cleanup** in the web UI, or POST to http://localhost:5000/cleanup
- Processing folders in `App\processing\` are also auto-deleted after 24 hours

---

## Optional: ML-Sharp (Single-Image, Not Bundled)

ML-Sharp can generate a Gaussian splat from a single image using Apple's transformer model. It is **not included** in this package and must be installed separately.

```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt
sharp --version
```

- Downloads ~500MB model on first run
- Restart the server after install — it auto-detects `sharp` in PATH
- Only works with the system Python that has ml-sharp installed (not the bundled Python)

---

## System Requirements Summary

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| RAM | 8 GB | 16-32 GB |
| GPU | Any NVIDIA GPU | NVIDIA RTX 3060+ (8GB+ VRAM) |
| VRAM | 4 GB | 8 GB+ |
| Disk | 10 GB free | 50 GB+ free (for large jobs) |
| Browser | Chrome 113 | Chrome latest |
