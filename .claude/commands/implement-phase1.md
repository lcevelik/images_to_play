# implement-phase1

Implement Phase 1 quick-win improvements from the improvement roadmap. See CLAUDE.md for full code snippets and rationale.

**Do not start this skill without reading CLAUDE.md "Improvement Roadmap" section first.**

## Items (implement in order)

### 1. Auto-resize images (saves 16x COLMAP time for 4K input)

File: `simple_splat/App/app.py`

Add function `resize_images_for_colmap(image_folder, scene_type)` before `process_images_async()`.
- Outdoor scenes: cap 1920px long edge
- Indoor/object: cap 2560px long edge
- All scenes: cap 3840px absolute max
- Use `PIL.Image` — already installed
- Call it at start of `process_images_async()` before COLMAP

See CLAUDE.md for exact implementation.

### 2. Skip MVS toggle

File: `simple_splat/App/run_glomap.py`

Add `enable_dense_override=None` param to `run_colmap()`. When `False`, skip MVS even if preset has `dense=True`.
Expose as `skip_mvs` checkbox in the advanced settings form in `templates/index.html`.

### 3. Phase labels in job status

File: `simple_splat/App/app.py`

Add `stage` key to `processing_status[job_id]` dict.
Values: `uploading`, `colmap`, `mvs`, `training`, `done`, `failed`
The frontend `static/js/main.js` already has stage display elements — just needs the backend field.

### 4. Wire preset JSON files

File: `simple_splat/App/run_glomap.py`

Add `_load_presets()` function that reads from `presets/*.json`.
Replace the inline `PRESETS` dict with this loader.
Preset files already exist at `simple_splat/App/presets/`.

### 5. Blur filter

File: `simple_splat/App/app.py`

Add `filter_blurry_images(image_folder, threshold=100)` using OpenCV Laplacian variance.
Warn-only by default (log to status, don't delete).
Call after resize in `process_images_async()`.

## Verify each step
After each item: restart server, submit a test job, confirm it reaches the next stage.
