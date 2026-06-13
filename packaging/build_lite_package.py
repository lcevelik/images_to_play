#!/usr/bin/env python3
"""
FonixFlow Splat — Lite package builder
=======================================
Produces a self-contained Windows distribution in:
    packaging/dist/FonixFlowSplat_Lite_v0.1.0/

Then zips it to:
    packaging/dist/FonixFlowSplat_Lite_v0.1.0.zip

Requirements to run this script:
  - Python 3.8+ on the build machine (uses system Python, NOT the embedded one)
  - pip accessible as 'pip' or 'python -m pip'
  - Internet connection (downloads python-embed and wheels)
  - C:/COLMAP/ — full COLMAP 4.x installation
  - C:/Brush/brush_app.exe — Brush Gaussian splat trainer

Usage:
    python packaging/build_lite_package.py [--no-zip] [--no-colmap] [--no-brush]
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import subprocess
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PACKAGE_VERSION = "0.1.0"
PACKAGE_NAME    = f"FonixFlowSplat_Lite_v{PACKAGE_VERSION}"

PYTHON_VERSION   = "3.11.9"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

REPO_ROOT    = Path(__file__).resolve().parent.parent
SIMPLE_SPLAT = REPO_ROOT / "simple_splat"
BUILD_DIR    = Path(__file__).resolve().parent / "build"
DIST_DIR     = Path(__file__).resolve().parent / "dist"
PACKAGE_DIR  = DIST_DIR / PACKAGE_NAME

# Source paths for external binaries
COLMAP_SRC = Path(r"C:\COLMAP")
BRUSH_SRC  = Path(r"C:\Brush\brush_app.exe")

# App source directories/files to EXCLUDE when copying App/
APP_EXCLUDES = {
    "__pycache__",
    "processing",
    "uploads",
    "job_status",
    "target",
    "server_run.log",
    ".deps_installed",
    "wheels",          # will be populated fresh
    "requirements.txt",  # will be replaced with lite version
}

# Lite requirements: no torch/gsplat/pytorch-msssim (MCMC trainer not bundled in Lite)
LITE_REQUIREMENTS = """\
# FonixFlow Splat Lite — offline pip requirements
# Installed automatically on first run from App/wheels/

# Web framework
Flask==3.0.0
Werkzeug==3.0.1
Flask-CORS==6.0.2

# Flask transitive dependencies
blinker==1.9.0
click==8.3.1
itsdangerous==2.2.0
Jinja2==3.1.6
MarkupSafe==3.0.3
colorama==0.4.6

# Image processing
opencv-python-headless
numpy
Pillow

# Process management
psutil
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"  {msg}")

def download(url: str, dest: Path):
    log(f"Downloading {url.split('/')[-1]} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    log(f"  -> {dest} ({dest.stat().st_size // 1024:,} KB)")

def run(cmd, **kwargs):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def copy_tree(src: Path, dst: Path, exclude: set = None):
    """Recursively copy src -> dst, skipping entries in exclude."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if exclude and item.name in exclude:
            continue
        if item.is_dir():
            copy_tree(item, dst / item.name)
        else:
            if item.suffix == ".pyc":
                continue
            shutil.copy2(item, dst / item.name)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_python_embed(python_dir: Path):
    """Download and extract Python 3.11 embed, then bootstrap pip."""
    print("\n[1/7] Python embed")
    if (python_dir / "python.exe").exists():
        log("Already present — skipping download")
        return

    embed_zip = BUILD_DIR / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    if not embed_zip.exists():
        download(PYTHON_EMBED_URL, embed_zip)

    log("Extracting python-embed ...")
    python_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(python_dir)

    # Enable 'import site' so pip works (it's commented out by default in embed)
    pth_files = list(python_dir.glob("python3*._pth"))
    if pth_files:
        pth = pth_files[0]
        content = pth.read_text()
        if "#import site" in content:
            pth.write_text(content.replace("#import site", "import site"))
            log(f"Patched {pth.name}: enabled 'import site'")

    # Bootstrap pip
    get_pip = BUILD_DIR / "get-pip.py"
    if not get_pip.exists():
        download(GET_PIP_URL, get_pip)

    python_exe = python_dir / "python.exe"
    log("Installing pip into embedded Python ...")
    run([python_exe, str(get_pip), "--no-warn-script-location"])
    log("pip bootstrapped OK")


def step_download_wheels(wheels_dir: Path, python_dir: Path):
    """Download all Lite wheels for Python 3.11 / win_amd64."""
    print("\n[2/7] Downloading wheels")
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Use the embedded pip to download wheels for the correct platform
    pip_exe = python_dir / "Scripts" / "pip.exe"
    if not pip_exe.exists():
        pip_exe = python_dir / "Scripts" / "pip3.exe"

    # Parse packages from LITE_REQUIREMENTS (skip comment lines and blank lines)
    packages = [
        line.strip()
        for line in LITE_REQUIREMENTS.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    log(f"Downloading {len(packages)} packages for cp311 / win_amd64 ...")
    run([
        pip_exe, "download",
        "--only-binary", ":all:",
        "--python-version", "311",
        "--platform", "win_amd64",
        "-d", str(wheels_dir),
        *packages,
    ])
    log(f"Wheels saved to {wheels_dir}")


def step_copy_app(app_src: Path, app_dst: Path):
    """Copy App/ source, excluding generated files."""
    print("\n[3/7] Copying App")
    if app_dst.exists():
        shutil.rmtree(app_dst)
    copy_tree(app_src, app_dst, exclude=APP_EXCLUDES)
    log(f"Copied App/ ({sum(1 for _ in app_dst.rglob('*'))} files)")

    # Write the Lite requirements.txt
    (app_dst / "requirements.txt").write_text(LITE_REQUIREMENTS)
    log("Wrote requirements.txt (Lite)")

    # Create empty dirs that the app expects at runtime
    for d in ("processing", "uploads", "job_status"):
        (app_dst / d).mkdir(exist_ok=True)
        (app_dst / d / ".gitkeep").write_text("")


def step_copy_colmap(colmap_src: Path, colmap_dst: Path):
    """Copy COLMAP bin/ and lib/ from system install."""
    print("\n[4/7] Copying COLMAP")
    if not colmap_src.exists():
        print(f"  WARNING: COLMAP not found at {colmap_src} — skipping.")
        print("  The app will look for COLMAP in PATH at runtime.")
        return

    for sub in ("bin", "lib"):
        src = colmap_src / sub
        dst = colmap_dst / sub
        if src.exists():
            log(f"Copying COLMAP/{sub}/ ...")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) // (1024 * 1024)
            log(f"  -> {dst} ({size_mb} MB)")
        else:
            log(f"  COLMAP/{sub}/ not found — skipping")

    # Copy plugins if present
    plugins_src = colmap_src / "plugins"
    if plugins_src.exists():
        shutil.copytree(plugins_src, colmap_dst / "plugins", dirs_exist_ok=True)


def step_copy_brush(brush_exe: Path, brush_dst_dir: Path):
    """Copy Brush executable."""
    print("\n[5/7] Copying Brush")
    if not brush_exe.exists():
        print(f"  WARNING: Brush not found at {brush_exe} — skipping.")
        print("  Users will need to place brush_app.exe in the Brush/ folder manually.")
        return

    brush_dst_dir.mkdir(parents=True, exist_ok=True)
    dst = brush_dst_dir / brush_exe.name
    shutil.copy2(brush_exe, dst)
    size_mb = dst.stat().st_size // (1024 * 1024)
    log(f"Copied {brush_exe.name} ({size_mb} MB)")


def step_write_launchers(package_dir: Path):
    """Write START_SERVER.bat and QUICK_START.bat with FonixFlow branding."""
    print("\n[6/7] Writing launchers")

    start_server = r"""@echo off
title FonixFlow Splat Server
color 0A

echo ========================================================================
echo   FonixFlow Splat  ^|  Standalone Edition  v""" + PACKAGE_VERSION + r"""
echo   Everything Bundled -- Zero Installation Required
echo ========================================================================
echo.

REM All paths relative to this .bat file (self-relocating)
set "ROOT=%~dp0"
set "PYTHON=%ROOT%Python"
set "APP=%ROOT%App"
set "COLMAP=%ROOT%COLMAP\bin"
set "COLMAP_LIB=%ROOT%COLMAP\lib"
set "BRUSH=%ROOT%Brush"

REM Add bundled tools to PATH
set "PATH=%PYTHON%;%PYTHON%\Scripts;%COLMAP%;%COLMAP_LIB%;%BRUSH%;%PATH%"

REM -------------------------------------------------------------------------
echo [1/3] Checking Python...
"%PYTHON%\python.exe" --version
if errorlevel 1 (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)
echo       Python OK

REM -------------------------------------------------------------------------
echo [2/3] Installing dependencies ^(first run only^)...
cd /d "%APP%"

if exist ".deps_installed" goto :deps_done

echo       Installing Python packages from bundled wheels...

if exist "%APP%\wheels" goto :install_offline

echo       WARNING: wheels folder not found -- trying online install...
"%PYTHON%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
if errorlevel 1 goto :deps_error
goto :deps_mark_done

:install_offline
"%PYTHON%\python.exe" -m pip install --no-warn-script-location --no-index --find-links="%APP%\wheels" -r requirements.txt
if errorlevel 1 goto :deps_error

:deps_mark_done
echo.  > .deps_installed
echo       Dependencies installed!
goto :start_server

:deps_error
echo.
echo ERROR: Failed to install dependencies.
echo Check that App\wheels\ folder exists and contains the .whl files.
pause
exit /b 1

:deps_done
echo       Dependencies already installed ^(skipping^)

REM -------------------------------------------------------------------------
:start_server
echo [3/3] Starting FonixFlow Splat server...
echo.
echo ========================================================================
echo   Open your browser at:  http://localhost:5000
echo   Press Ctrl+C to stop
echo ========================================================================
echo.

"%PYTHON%\python.exe" app.py

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo   ERROR: Server stopped unexpectedly. Check above for details.
    echo ========================================================================
)
pause
"""

    quick_start = r"""@echo off
title FonixFlow Splat

echo ========================================================================
echo   FonixFlow Splat -- Quick Start
echo ========================================================================
echo.

REM Start the server in a separate minimized window
start "FonixFlow Splat Server" /min "%~dp0START_SERVER.bat"

echo   Server starting in background...
echo   ^(First run installs packages -- may take ~30 seconds^)
echo.

REM Wait for the server to respond
set MAX_WAIT=120
set WAITED=0

:wait_loop
timeout /t 1 /nobreak >nul
set /a WAITED+=1

curl -s --max-time 1 http://localhost:5000 >nul 2>&1
if %errorlevel% == 0 goto server_ready

if %WAITED% GEQ %MAX_WAIT% goto timeout_error

echo   Waiting for server... (%WAITED%s / %MAX_WAIT%s max)
goto wait_loop

:server_ready
echo.
echo   Server ready! Opening browser...
echo.
start http://localhost:5000
goto end

:timeout_error
echo.
echo   Server did not respond within %MAX_WAIT% seconds.
echo   Run START_SERVER.bat directly to see any error messages.
echo.
pause

:end
"""

    (package_dir / "START_SERVER.bat").write_text(start_server)
    (package_dir / "QUICK_START.bat").write_text(quick_start)
    log("Wrote START_SERVER.bat")
    log("Wrote QUICK_START.bat")

    # Copy README.txt from source
    readme_src = SIMPLE_SPLAT / "README.txt"
    if readme_src.exists():
        shutil.copy2(readme_src, package_dir / "README.txt")
        log("Copied README.txt")


def step_zip(package_dir: Path, zip_path: Path):
    """Zip the package directory."""
    print("\n[7/7] Creating zip")
    log(f"Zipping {package_dir.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(package_dir.rglob("*")):
            if f.is_file():
                arcname = PACKAGE_NAME + "/" + str(f.relative_to(package_dir))
                zf.write(f, arcname)
    size_mb = zip_path.stat().st_size // (1024 * 1024)
    log(f"Created {zip_path.name} ({size_mb} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FonixFlow Splat Lite package")
    parser.add_argument("--no-zip",    action="store_true", help="Skip final zip step")
    parser.add_argument("--no-colmap", action="store_true", help="Skip COLMAP copy (for faster iteration)")
    parser.add_argument("--no-brush",  action="store_true", help="Skip Brush copy")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FonixFlow Splat Lite — Package Builder v{PACKAGE_VERSION}")
    print("=" * 60)
    print(f"  Output: {PACKAGE_DIR}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    python_dir = PACKAGE_DIR / "Python"
    wheels_dir = PACKAGE_DIR / "App" / "wheels"

    step_python_embed(python_dir)
    # Copy App/ first (rmtree + copy), THEN download wheels into App/wheels/
    step_copy_app(SIMPLE_SPLAT / "App", PACKAGE_DIR / "App")
    step_download_wheels(wheels_dir, python_dir)

    if not args.no_colmap:
        step_copy_colmap(COLMAP_SRC, PACKAGE_DIR / "COLMAP")
    else:
        log("Skipping COLMAP copy (--no-colmap)")

    if not args.no_brush:
        step_copy_brush(BRUSH_SRC, PACKAGE_DIR / "Brush")
    else:
        log("Skipping Brush copy (--no-brush)")

    step_write_launchers(PACKAGE_DIR)

    if not args.no_zip:
        zip_path = DIST_DIR / f"{PACKAGE_NAME}.zip"
        step_zip(PACKAGE_DIR, zip_path)

    # Summary
    total_mb = sum(
        f.stat().st_size for f in PACKAGE_DIR.rglob("*") if f.is_file()
    ) // (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"  Build complete!")
    print(f"  Package size:  {total_mb:,} MB")
    print(f"  Location:      {PACKAGE_DIR}")
    if not args.no_zip:
        zip_path = DIST_DIR / f"{PACKAGE_NAME}.zip"
        if zip_path.exists():
            print(f"  Zip:           {zip_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
