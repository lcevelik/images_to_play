
================================================================================
  GAUSSIAN SPLATTING - STANDALONE EDITION
  Zero Dependencies - Everything Included!
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
   [x] Gaussian Splatting Web App (Flask)
   [x] COLMAP (3D reconstruction engine)
   [x] All Python dependencies (offline wheels, no internet required)
   [x] PlayCanvas SuperSplat Viewer v1.15.0 (browser-based, WebGPU)
   [x] WebXR controller profiles (for VR viewing, fully offline)

DOWNLOAD REQUIRED (one-time):
   [ ] Brush - Gaussian splat trainer (too large to bundle, download separately)
       1. Go to: https://github.com/ArthurBrussee/brush/releases/latest
       2. Download the Windows build (brush_app.exe)
       3. Create a folder called "Brush\" next to this README.txt
       4. Place brush_app.exe inside it
   Without Brush the app still works but outputs a basic point cloud
   instead of a full trained Gaussian splat.

NO OTHER INSTALLATION REQUIRED!
   - No Python install
   - No COLMAP install
   - No pip install
   - No internet connection needed at runtime
   - Just extract, add Brush, and run!

SYSTEM REQUIREMENTS:
   - Windows 10/11 (64-bit)
   - 16GB+ RAM recommended (8GB minimum)
   - NVIDIA GPU with CUDA (strongly recommended for processing)
   - GPU with WebGPU support (required for browser viewer - any modern NVIDIA/AMD GPU)
   - 10GB+ free disk space (more for large jobs)

BROWSER REQUIREMENTS (for the 3D viewer):
   - Chrome 113+ or Edge 113+ (recommended - best WebGPU support)
   - Firefox with WebGPU enabled (about:config -> dom.webgpu.enabled)
   - The viewer uses WebGPU for hardware-accelerated Gaussian splat rendering

FIRST RUN:
   - First start installs Python packages from bundled wheels (~30 seconds, no internet)
   - Subsequent starts are instant

TROUBLESHOOTING:

   Server won't start?
   - Run START_SERVER.bat to see error messages in the console
   - Check that antivirus isn't blocking Python or COLMAP

   Viewer shows blank/white screen?
   - Ensure you are using Chrome 113+ or Edge 113+
   - Your GPU must support WebGPU (most NVIDIA RTX and AMD RX 5000+ GPUs)
   - Try visiting: chrome://gpu to verify WebGPU is enabled

   Out of memory during processing?
   - Use "Low" preset in the app
   - Close other applications
   - Reduce image count or resolution

   COLMAP fails to reconstruct?
   - Ensure images have 60-80% overlap between consecutive shots
   - Use at least 10-20 sharp, well-lit images of the same subject
   - Avoid plain/textureless or reflective surfaces

   Need help?
   - Check App\README.md for detailed documentation
   - Check App\SETUP.md for troubleshooting

FOLDER STRUCTURE:
   GaussianSplatting_Standalone\
   +-- START_SERVER.bat       <- Double-click to start
   +-- README.txt             <- This file
   +-- Python\                <- Portable Python 3.11
   +-- COLMAP\                <- Bundled COLMAP (bin\ + lib\)
   +-- Brush\                 <- Bundled Brush trainer (brush_app.exe)
   +-- App\                   <- Web application
       +-- app.py             <- Flask server
       +-- requirements.txt   <- Python deps list
       +-- wheels\            <- Pre-downloaded pip wheels (offline install)
       +-- templates\         <- HTML pages
       +-- static\
           +-- supersplat\    <- PlayCanvas SuperSplat viewer (WebGPU)

SHARING:
   - This entire folder is self-contained and portable
   - Copy to USB drive, cloud storage, another PC - it just works
   - Works on any Windows 10/11 PC with a WebGPU-capable browser

APPROXIMATE PACKAGE SIZE: ~1.5GB
   - Python:              ~50MB
   - COLMAP + libs:       ~200MB
   - Brush:               ~200MB
   - App + wheels:        ~150MB
   - SuperSplat viewer:   ~3MB
   - WebXR profiles:      ~100MB

================================================================================
Ready to create amazing 3D Gaussian Splats!
No installation, no internet, no hassle - just works!
================================================================================
