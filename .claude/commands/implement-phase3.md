# implement-phase3

Implement Phase 3: gsplat MCMC trainer as an alternative to Brush.

**Complete Phases 1 and 2 first. Read CLAUDE.md "Phase 3" section for full code.**

## Overview

Add a second training backend using `gsplat` Python library's `MCMCStrategy`.
The RTX 8000 (48GB VRAM) can handle up to 1,500,000 Gaussians with MCMC.
Expose as `trainer` form field: `brush` (default) or `gsplat_mcmc`.

## New file to create

`simple_splat/App/gsplat_mcmc_trainer.py`

See CLAUDE.md for the full implementation including:
- `train_mcmc(image_folder, sparse_path, output_ply, num_iterations, cap_max)` function
- Camera loading from COLMAP cameras.bin/images.bin
- Coordinate system conversion (COLMAP → OpenGL)
- PLY export compatible with SuperSplat viewer
- Progress callback for streaming to job status

## Integration points

File: `simple_splat/App/app.py`

In `process_images_async()`:
- Check `job_params["trainer"]` (from form)
- If `gsplat_mcmc`: call `gsplat_mcmc_trainer.train_mcmc()`
- If `brush` (default): existing Brush subprocess path unchanged

## Key parameters for MCMC on RTX 8000
- `cap_max = 1_500_000` Gaussians
- `num_iterations = 30_000` (medium quality)
- `strategy = MCMCStrategy(cap_max=cap_max)`

## Verify
Submit a job with `trainer=gsplat_mcmc`. Confirm `.ply` is produced and loads in SuperSplat viewer.
Check that `brush` jobs still work unchanged.
