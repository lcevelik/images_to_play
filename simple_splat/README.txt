
================================================================================
  FONIXFLOW SPLAT  --  Standalone Edition
  Create 3D Gaussian Splats from photos. Everything bundled.
================================================================================

QUICK START:
   1. Extract this folder anywhere on your computer
   2. Double-click: QUICK_START.bat
      (opens the browser automatically when the server is ready)
   3. Upload images and start processing!

ALTERNATIVE START (shows console output):
   - Double-click: START_SERVER.bat
   - Then open: http://localhost:5000

WHAT'S INCLUDED:
   [x] Python 3.11 (portable, no install needed)
   [x] FonixFlow Splat Web App
   [x] COLMAP 4.1 (3D reconstruction engine)
   [x] All Python dependencies (offline wheels, no internet required)
   [x] SuperSplat Viewer (browser-based, WebGPU)

ONE-TIME DOWNLOAD -- Brush trainer:
   Brush is the Gaussian splat training engine. It is too large to bundle
   and must be downloaded once from GitHub:

   1. Go to: https://github.com/ArthurBrussee/brush/releases/latest
   2. Download the Windows build (brush_app.exe)
   3. Create a folder named "Brush\" next to this README.txt
   4. Place brush_app.exe inside it

   Without Brush the app still works but exports a basic point cloud
   instead of a trained Gaussian splat.

NO OTHER INSTALLATION REQUIRED:
   - No Python to install
   - No COLMAP to install
   - No pip install
   - No internet connection needed at runtime
   - Just extract, add Brush, and run

SYSTEM REQUIREMENTS:
   - Windows 10/11 (64-bit)
   - 16 GB+ RAM recommended (8 GB minimum)
   - NVIDIA GPU (strongly recommended for COLMAP and Brush)
   - Any modern GPU with WebGPU support for the browser viewer
   - 10 GB+ free disk space (more for large jobs)

BROWSER REQUIREMENTS (3D viewer):
   - Chrome 113+ or Edge 113+ -- recommended
   - Firefox: enable at about:config -> dom.webgpu.enabled

FIRST RUN:
   First start installs Python packages from bundled wheels (~30 seconds).
   Subsequent starts are instant.

TROUBLESHOOTING:
   See App\SETUP.md for full troubleshooting guide.

   Server won't start?     -> Run START_SERVER.bat, read the console output
   Viewer blank/white?     -> Use Chrome 113+ with WebGPU-capable GPU
   COLMAP fails?           -> Use 10-20+ sharp overlapping images
   Out of memory?          -> Switch to Low preset, close other apps

FOLDER STRUCTURE:
   FonixFlowSplat_Lite\
   +-- START_SERVER.bat       <- Double-click to start server
   +-- QUICK_START.bat        <- Starts server + opens browser
   +-- README.txt             <- This file
   +-- Python\                <- Portable Python 3.11
   +-- COLMAP\                <- Bundled COLMAP 4.1 (bin\ + lib\)
   +-- Brush\                 <- Brush trainer (add brush_app.exe here)
   +-- App\                   <- Web application source
       +-- app.py             <- Flask server entry point
       +-- requirements.txt   <- Python deps list
       +-- wheels\            <- Pre-downloaded pip wheels (offline install)
       +-- templates\         <- HTML pages
       +-- static\
           +-- supersplat\    <- 3D Gaussian splat viewer (WebGPU)

SHARING:
   The package is fully self-contained and portable.
   Copy to USB, cloud storage, or another PC -- it just works on any
   Windows 10/11 machine with a WebGPU-capable browser.

================================================================================
