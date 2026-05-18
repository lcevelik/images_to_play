# check-setup

Verify the full tool chain is installed and working.

## Checks to run

```bash
# COLMAP version (should be 4.1.0)
C:\COLMAP\bin\colmap.exe --version

# COLMAP global_mapper available (replaces GLOMAP in 4.x)
C:\COLMAP\bin\colmap.exe global_mapper --help

# Brush binary
simple_splat\Brush\brush_app.exe --version

# Python + CUDA torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ML-Sharp CLI
sharp --help

# gsplat (needed for ml-sharp)
python -c "import gsplat; print('gsplat ok')"

# pycolmap (optional, for stats)
python -c "import pycolmap; print('pycolmap ok')"
```

## Expected results
- COLMAP: `4.1.0`
- Torch: `2.x+cu126`, CUDA: `True`
- Brush: prints version or usage
- sharp: prints usage (may take 6s — PyTorch import)

## Common failures
| Symptom | Fix |
|---------|-----|
| COLMAP DLL error | Add `C:\COLMAP\bin` to PATH |
| torch.cuda.is_available() = False | Reinstall with `--index-url https://download.pytorch.org/whl/cu126 --force-reinstall` |
| sharp not found | Run `pip install -e ml-sharp/ --no-deps` |
| ML-Sharp shows False in app | Restart server; check PATH |
