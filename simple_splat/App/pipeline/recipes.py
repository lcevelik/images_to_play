"""Draft / Production pipeline orchestrators.

Both drive the validated recipes via SUBPROCESSES (so torch/transformers/lightglue
load in child processes, not the Flask server):

  DRAFT      learned-SfM -> undistort -> gap-fill dense seed -> ONE fast Brush pass
             (~10 min). No combine.
  PRODUCTION learned-SfM -> undistort -> MCMC + Brush -> combine(sigma=1.7) -> compress
             (~4 hr). The champion recipe.

Each writes the final splat to <job_dir>/gaussian_splat.ply and reports progress via
status_cb(step:str, progress:int, stage:str|None).
"""
import os
import sys
import glob
import time
import shutil
import struct
import subprocess

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)  # so direct `import pipeline.X` works when run as a script
PY = sys.executable
COLMAP = next((p for p in (r"C:\COLMAP\bin\colmap.exe", "colmap") if os.path.exists(p) or p == "colmap"), "colmap")
BRUSH = next((p for p in (r"C:\Brush\brush_app.exe",
                          os.path.join(APP_DIR, "Brush", "brush_app.exe")) if os.path.exists(p)), None)

# Suppress console/GUI pop-up windows for every spawned subprocess (Windows).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def _hidden_si():
    """STARTUPINFO that hides a child's window (used for the Brush GUI binary)."""
    if os.name != 'nt':
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


def _run(cmd, log, cwd=APP_DIR, timeout=None):
    """Run a subprocess, streaming stdout to log(). Raises on non-zero exit."""
    log(f"$ {' '.join(str(c) for c in cmd)}", "DEBUG")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=os.environ.copy(),
                         creationflags=_NO_WINDOW, startupinfo=_hidden_si())
    start = time.time()
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log(line, "INFO")
        if timeout and time.time() - start > timeout:
            p.kill()
            raise TimeoutError(f"step exceeded {timeout}s")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"subprocess failed (exit {p.returncode}): {cmd[0]}")


def _py(code, log, timeout=None):
    _run([PY, "-c", f"import sys; sys.path.insert(0, r'{APP_DIR}')\n{code}"], log, timeout=timeout)


def _learned_sfm(images_dir, job_dir, log, status_cb=None):
    seed = os.path.join(job_dir, "seed")
    _seen = {'s': None}

    def _slog(line, lvl="INFO"):
        log(line, lvl)
        if not status_cb:
            return
        # drive the UI stage timers from learned-SfM's own phase output (debounced)
        if "extracted" in line and _seen['s'] != 'fe':
            _seen['s'] = 'fe'; status_cb("Feature extraction...", 6, "feature_extraction")
        elif "matched through" in line and _seen['s'] != 'fm':
            _seen['s'] = 'fm'; status_cb("Feature matching...", 9, "feature_matching")
        elif ("Registering image" in line or "Retriangulation" in line) and _seen['s'] != 'mp':
            _seen['s'] = 'mp'; status_cb("3D mapping...", 12, "mapping")

    _run([PY, os.path.join(APP_DIR, "pipeline", "learned_sfm.py"),
          "-i", images_dir, "-o", seed, "--device", "cuda", "--max-kpts", "4096"], _slog, timeout=1800)
    return os.path.join(seed, "sparse", "0")


def _build_preview(scene, job_dir, log):
    """Write the 3D alignment preview (sparse points + camera frustums) for the Process tab."""
    try:
        from pipeline.sparse_preview import build_alignment_json
        build_alignment_json(os.path.join(scene, "sparse", "0"), os.path.join(job_dir, "alignment.json"))
        log("alignment preview written", "INFO")
    except Exception as e:
        log(f"alignment preview failed: {e}", "WARNING")


def _export_fbx_camera(scene, job_dir, log, fps=30):
    """Export a per-frame FBX matchmove camera from the undistorted reconstruction
    -> job_dir/camera_tracking/cameras.fbx (downloadable). Returns True on success."""
    try:
        from pipeline.camera_tracking import extract_camera_poses, export_camera_fbx
        tdir = os.path.join(job_dir, "camera_tracking")
        os.makedirs(tdir, exist_ok=True)
        poses = extract_camera_poses(os.path.join(scene, "sparse", "0"))
        export_camera_fbx(poses, os.path.join(tdir, "cameras.fbx"), fps)
        log(f"FBX matchmove camera exported ({len(poses)} frames) -> camera_tracking/cameras.fbx", "INFO")
        return True
    except Exception as e:
        log(f"FBX camera export failed: {e}", "WARNING")
        return False


def _undistort(images_dir, sparse_dir, job_dir, log):
    scene = os.path.join(job_dir, "scene")
    _run([COLMAP, "image_undistorter", "--image_path", images_dir,
          "--input_path", sparse_dir, "--output_path", scene, "--output_type", "COLMAP"], log, timeout=900)
    s0 = os.path.join(scene, "sparse", "0")
    os.makedirs(s0, exist_ok=True)
    for f in glob.glob(os.path.join(scene, "sparse", "*.bin")):
        shutil.move(f, os.path.join(s0, os.path.basename(f)))
    return scene


def _brush(seed_dir, out_dir, steps, res, growth_stop, log, timeout=14400):
    """Launch Brush, wait for export_{steps}.ply (GUI binary: no stdout, poll files)."""
    if not BRUSH:
        raise RuntimeError("Brush not found")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"export_{steps}.ply")
    cmd = [BRUSH, seed_dir, "--total-steps", str(steps), "--max-resolution", str(res),
           "--sh-degree", "3", "--growth-grad-threshold", "0.00002", "--growth-select-fraction", "0.15",
           "--refine-every", "100", "--growth-stop-iter", str(growth_stop),
           "--export-path", out_dir, "--export-name", "export_{iter}.ply",
           "--export-every", str(max(2000, steps // 4))]
    log(f"$ {' '.join(cmd)}", "DEBUG")
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=_NO_WINDOW, startupinfo=_hidden_si())
    start = time.time()
    stall_secs = 900           # if no NEW export for 15 min, salvage instead of hanging
    last_export_count, last_progress_t = 0, time.time()
    while time.time() - start < timeout:
        if os.path.exists(final):
            time.sleep(3)  # let the write finish
            break
        if p.poll() is not None and not os.path.exists(final):
            log("Brush exited before final export — salvaging most-progressed", "WARNING")
            break
        # stall detection: count exports; reset the clock when a new one appears
        nexports = len(glob.glob(os.path.join(out_dir, "export_*.ply")))
        if nexports > last_export_count:
            last_export_count, last_progress_t = nexports, time.time()
        elif nexports > 0 and time.time() - last_progress_t > stall_secs:
            log(f"Brush stalled (no new export in {stall_secs}s) — salvaging", "WARNING")
            try:
                p.terminate()
            except Exception:
                pass
            break
        time.sleep(5)
    try:
        p.terminate()
    except Exception:
        pass
    if not os.path.exists(final):
        # salvage most-progressed export
        exports = sorted(glob.glob(os.path.join(out_dir, "export_*.ply")),
                         key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))) or 0))
        if not exports:
            raise RuntimeError("Brush produced no export")
        final = exports[-1]
    return final


def run_draft(images_dir, job_dir, status_cb, log, export_fbx=False):
    """~10 min: gap-fill dense seed -> one fast Brush pass. Writes job_dir/gaussian_splat.ply."""
    out_ply = os.path.join(job_dir, "gaussian_splat.ply")

    status_cb("Feature extraction...", 5, "feature_extraction")
    sparse = _learned_sfm(images_dir, job_dir, log, status_cb)

    status_cb("Undistorting...", 18, "mapping")
    scene = _undistort(images_dir, sparse, job_dir, log)
    _build_preview(scene, job_dir, log)   # sparse point cloud preview for the Process tab
    if export_fbx:
        _export_fbx_camera(scene, job_dir, log)

    status_cb("Depth-seeding (gap-fill)...", 25, "mapping")
    dense = os.path.join(job_dir, "scene-dense")
    _py(f"from pipeline.depth_seed import build_dense_seed_dir\n"
        f"build_dense_seed_dir(r'{scene}', r'{dense}', gap_mult=12)", log, timeout=900)

    status_cb("Fast Brush pass...", 35, "training")
    export = _brush(dense, os.path.join(job_dir, "draft-brush"), steps=8000, res=1280,
                    growth_stop=4000, log=log, timeout=3600)
    shutil.copy2(export, out_ply)
    status_cb("Draft complete!", 100, None)
    return out_ply


def run_production(images_dir, job_dir, status_cb, log, export_fbx=False):
    """~2.5 hr: MCMC + Brush (parallel) + combine(sigma=1.7) + compress. Writes job_dir/gaussian_splat.ply."""
    out_ply = os.path.join(job_dir, "gaussian_splat.ply")

    status_cb("Feature extraction...", 4, "feature_extraction")
    sparse = _learned_sfm(images_dir, job_dir, log, status_cb)

    status_cb("Undistorting...", 10, "mapping")
    scene = _undistort(images_dir, sparse, job_dir, log)
    _build_preview(scene, job_dir, log)   # sparse point cloud preview for the Process tab
    if export_fbx:
        _export_fbx_camera(scene, job_dir, log)

    # Train MCMC and Brush IN PARALLEL — they're independent (same seed) and fit in 48 GB
    # together (~14 GB each on the RTX 8000), so wall-clock ~halves vs sequential (~4 hr -> ~2.5 hr).
    # Each runs as its own GPU subprocess; threads here just wait on them concurrently.
    import threading
    status_cb("Training MCMC + Brush in parallel (~2.5 hr)...", 15, "training")
    mcmc = os.path.join(job_dir, "mcmc", "gaussian_splat.ply")
    os.makedirs(os.path.dirname(mcmc), exist_ok=True)
    errors, brush_holder = {}, {}

    def _train_mcmc():
        try:
            _py(f"from pipeline.gsplat_mcmc_trainer import train_mcmc\n"
                f"train_mcmc(r'{scene}', total_steps=30000, cap_max=4000000, sh_degree=3, use_lpips=True,\n"
                f"           strategy_name='mcmc', export_opacity_min=0.03, output_ply_path=r'{mcmc}')",
                log, timeout=18000)
        except Exception as e:
            errors['mcmc'] = e

    def _train_brush():
        try:
            brush_holder['ply'] = _brush(scene, os.path.join(job_dir, "brush"), steps=30000,
                                         res=3968, growth_stop=15000, log=log, timeout=18000)
        except Exception as e:
            errors['brush'] = e

    t_mcmc, t_brush = threading.Thread(target=_train_mcmc), threading.Thread(target=_train_brush)
    t_mcmc.start(); t_brush.start()
    t_mcmc.join(); t_brush.join()
    if errors:
        raise RuntimeError("parallel training failed — " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    brush_export = brush_holder['ply']

    status_cb("Combining (sigma=1.7)...", 95, "training")
    combined = os.path.join(job_dir, "combined.ply")
    _py(f"from pipeline.combine_splats import combine_fade\n"
        f"combine_fade(r'{mcmc}', r'{brush_export}', r'{os.path.join(scene, 'sparse', '0')}', r'{combined}',\n"
        f"             fade_mode='gaussian', sigma_mult=1.7)", log, timeout=1800)

    status_cb("Compressing...", 98, "training")
    _py(f"from pipeline.compress_ply import compress\n"
        f"compress(r'{combined}', r'{out_ply}', max_sh=3)", log, timeout=900)
    status_cb("Production complete!", 100, None)
    return out_ply


def _write_status(job_dir, step, progress, stage, status='processing', ply_path=None, fbx=False):
    import json
    s = {'step': step, 'progress': progress, 'stage': stage, 'status': status}
    if ply_path:
        s['ply_path'] = ply_path
    if fbx:
        s['fbx'] = True
    tmp = os.path.join(job_dir, 'recipe_status.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(s, f)
    os.replace(tmp, os.path.join(job_dir, 'recipe_status.json'))


if __name__ == '__main__':
    # Detached entry point: python pipeline/recipes.py <draft|production> <images_dir> <job_dir> [--fbx]
    # Runs independently of the Flask server and writes status/output to job_dir, so a
    # server restart/crash never orphans the job.
    quality, images_dir, job_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    export_fbx = '--fbx' in sys.argv
    logf = open(os.path.join(job_dir, 'recipe.log'), 'a', buffering=1, encoding='utf-8')

    def log(m, lvl='INFO'):
        logf.write(f"[{lvl}] {m}\n")

    def status(step, progress, stage):
        _write_status(job_dir, step, progress, stage)
        log(f"[{progress}%] {step}")

    try:
        runner = run_draft if quality == 'draft' else run_production
        out = runner(images_dir, job_dir, status, log, export_fbx=export_fbx)
        _have_fbx = os.path.exists(os.path.join(job_dir, 'camera_tracking', 'cameras.fbx'))
        _write_status(job_dir, 'Processing complete!', 100, None, status='completed', ply_path=out, fbx=_have_fbx)
    except Exception as e:
        import traceback
        log(traceback.format_exc(), 'ERROR')
        _write_status(job_dir, f'Failed: {e}', 0, None, status='error')
        sys.exit(1)
