# FonixFlow Splat — Full Edition Build Plan

## What "Full" adds over Lite

| Feature | Lite | Full |
|---------|------|------|
| Brush trainer | yes (user downloads) | yes (bundled) |
| COLMAP 4.1 | yes (bundled) | yes (bundled) |
| gsplat MCMC trainer | no | yes (CUDA, optional) |
| PyTorch | no | yes (CUDA) |
| Trainer toggle in UI | Brush only | Brush + MCMC |
| MCMC quality cap UI | no | yes |
| ML-Sharp single-image | no | yes (optional) |
| `sharp` CLI in PATH | no | yes |
| Installer experience | manual bat | NSIS/Inno installer |
| Auto-update check | no | planned |
| Package size | ~1.3 GB | ~5-8 GB |

---

## Key challenges

### 1. PyTorch CUDA wheel is 2.5 GB

PyTorch for CUDA 12.6 (`torch==2.x+cu126`) weighs ~2.5 GB as a wheel.
This is too large to ship as a single download but can be handled with a
two-stage installer:

- **Stage 1**: ship the base package without torch (mirrors Lite)
- **Stage 2**: a `INSTALL_MCMC.bat` script downloads and installs torch
  on first use. Alternatively, offer a separate "Full + MCMC" download
  that includes the torch wheel pre-cached.

The staged approach keeps the core download manageable while giving users
the option to activate MCMC.

### 2. gsplat JIT compilation

`gsplat` JIT-compiles CUDA kernels on first training run (~10 min).
Requirements:
- MSVC `cl.exe` in PATH (app.py discovers it automatically on startup)
- CUDA toolkit `nvcc`
- First run must complete without interruption

For Full, document this as a one-time ~10 min setup step after installing
the MCMC add-on.

**Alternative**: use gsplat prebuilt wheels (Python 3.10 only as of June 2026).
If we switch the embed to Python 3.10, we can ship prebuilt gsplat wheels and
skip JIT entirely. Prebuilt wheels are at https://docs.gsplat.studio/whl.

### 3. ML-Sharp

ML-Sharp requires:
- `torch` (CUDA)
- `gsplat`
- `timm`, `imageio`, `imageio-ffmpeg`, `matplotlib`, `pillow-heif`, `scipy`
- `plyfile`
- ~500 MB model checkpoint (auto-downloads to `~/.cache/torch/hub/`)

The checkpoint auto-download is fine for Full (user has internet at install
time). The `sharp` binary goes into the embedded Python's `Scripts/` folder
and app.py detects it via `subprocess(['sharp', '--help'])`.

### 4. Installer UX

Lite ships as a zip you unpack. Full warrants a proper installer:
- **NSIS** or **Inno Setup** — free, well-documented, Windows-standard `.exe`
  installer experience
- Installer steps: pick install dir, optional MCMC add-on download, start menu
  shortcut, PATH entry

---

## Recommended build approach

```
packaging/build_full_package.py
```

Extend `build_lite_package.py`:

1. All Lite steps (Python embed, App, COLMAP, Brush, wheels, launchers)

2. **MCMC add-on step** (optional, `--with-mcmc` flag):
   - Download torch+cu126 wheel (~2.5 GB) into `App/wheels_mcmc/`
   - Download gsplat, pytorch-msssim, scipy, plyfile wheels
   - Write `INSTALL_MCMC.bat` — installs from `wheels_mcmc/` offline
   - Write `requirements_mcmc.txt`

3. **ML-Sharp step** (optional, `--with-mlsharp` flag):
   - `pip install -e ml-sharp/` into the embedded Python
   - Download timm, imageio, imageio-ffmpeg, pillow-heif, scipy, plyfile
   - The 500 MB model checkpoint downloads on first `sharp predict` run

4. **Installer packaging** (optional, `--make-installer` flag):
   - Requires Inno Setup (`iscc.exe`) on the build machine
   - Write `packaging/installer.iss` config
   - Run `iscc installer.iss` → produces `FonixFlowSplat_Full_Setup.exe`

---

## Python version recommendation for Full

Use **Python 3.10.x** (not 3.11 or 3.12) if including gsplat:
- gsplat prebuilt wheels are available for cp310 only (June 2026)
- Avoids the ~10 min JIT compile on first run
- All other dependencies (Flask, numpy, opencv, torch 2.x) support 3.10

If JIT compilation is acceptable, Python 3.11 works fine (same as Lite).

---

## Approximate Full package sizes

| Component | Size |
|-----------|------|
| Python embed 3.10 | ~15 MB |
| Python site-packages (Lite deps) | ~200 MB |
| PyTorch CUDA 12.6 | ~2,500 MB |
| gsplat + deps | ~100 MB |
| ML-Sharp model checkpoint | ~500 MB |
| COLMAP bin + lib | ~750 MB |
| Brush | ~160 MB |
| App source | ~80 MB |
| **Total** | **~4.3 GB** |

With a staged download (core + MCMC pack separately):
- Core installer: ~1.3 GB (same as Lite)
- MCMC add-on pack: ~2.7 GB (torch + gsplat)
- ML-Sharp pack: ~700 MB (model + deps)

---

## Milestone targets

| Milestone | Description |
|-----------|-------------|
| Full v0.2.0 | Core + optional MCMC installer, no ML-Sharp |
| Full v0.3.0 | Core + MCMC + ML-Sharp, Inno Setup installer |

Prerequisite: expose `mcmc_cap` in the Settings UI before Full v0.2.0
(listed in PROJECT.md TODO).
