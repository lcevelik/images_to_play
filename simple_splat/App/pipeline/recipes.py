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
import threading
import subprocess

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)  # so direct `import pipeline.X` works when run as a script
PY = sys.executable
COLMAP = next((p for p in (r"C:\COLMAP\bin\colmap.exe", "colmap") if os.path.exists(p) or p == "colmap"), "colmap")
from pipeline.run_glomap import find_brush_binary, add_msvc_to_path
BRUSH = find_brush_binary()

# Suppress console/GUI pop-up windows for every spawned subprocess (Windows).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def _resolve_brush_path():
    """Return a usable Brush executable path, refreshing it on every call."""
    global BRUSH
    if BRUSH and os.path.exists(BRUSH):
        return BRUSH

    resolved = find_brush_binary()
    if resolved and os.path.exists(resolved):
        BRUSH = resolved
        return BRUSH

    # Last-resort fallback: accept a PATH-resolved executable even if the
    # cached value was stale or missing at import time.
    if BRUSH and os.path.isabs(BRUSH):
        return BRUSH
    return None


def _hidden_si():
    """STARTUPINFO that hides a child's window (used for the Brush GUI binary)."""
    if os.name != 'nt':
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


_CUDA_ENV = {}


def _cuda_env():
    """Env for a CUDA-compiling child (gsplat JIT-builds its kernels with nvcc).

    CUDA_PATH can point at an OLD toolkit — on this box v11.8, whose nvcc rejects the
    -std=c++20 that torch 2.13 passes ("nvcc fatal: Value 'c++20' is not defined for
    option 'std'"), so every gsplat kernel fails to build and MCMC dies at startup.
    Pick the toolkit matching torch's own CUDA build instead (newest as a fallback)."""
    if 'env' in _CUDA_ENV:
        return _CUDA_ENV['env']

    def _ver(d):
        try:
            return tuple(int(x) for x in os.path.basename(d).lstrip('v').split('.'))
        except ValueError:
            return (0,)

    roots = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
    cands = sorted((d for d in roots if os.path.exists(os.path.join(d, "bin", "nvcc.exe"))),
                   key=_ver, reverse=True)
    env = os.environ.copy()
    if cands:
        chosen = cands[0]
        try:  # prefer the toolkit torch itself was built against
            want = subprocess.run([PY, "-c", "import torch; print(torch.version.cuda)"],
                                  capture_output=True, text=True, timeout=120,
                                  creationflags=_NO_WINDOW).stdout.strip()
            chosen = next((d for d in cands if os.path.basename(d) == f"v{want}"), chosen)
        except Exception:
            pass
        env["CUDA_HOME"] = env["CUDA_PATH"] = chosen
        env["PATH"] = os.path.join(chosen, "bin") + os.pathsep + env.get("PATH", "")

    # nvcc drives MSVC as its host compiler, and torch's ninja writer bails outright
    # on `where cl` — a missing cl.exe is as fatal as the wrong nvcc. app.py puts it on
    # PATH at startup, but don't depend on the spawn chain having come through app.py.
    cl_dir = add_msvc_to_path()   # mutates os.environ; mirror it into the child env
    if cl_dir and cl_dir not in env.get("PATH", ""):
        env["PATH"] = cl_dir + os.pathsep + env.get("PATH", "")

    _CUDA_ENV['env'] = env
    return env


def _run(cmd, log, cwd=APP_DIR, timeout=None, env=None):
    """Run a subprocess, streaming stdout to log(). Raises on non-zero exit.

    stdout is drained on a reader thread so the timeout fires even when the child
    hangs SILENTLY — checking the clock only per output line never times out a
    child that stops writing (a detached recipe job would then be stuck forever)."""
    log(f"$ {' '.join(str(c) for c in cmd)}", "DEBUG")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env or os.environ.copy(),
                         creationflags=_NO_WINDOW, startupinfo=_hidden_si())

    def _drain():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                log(line, "INFO")

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise TimeoutError(f"step exceeded {timeout}s: {cmd[0]}")
    reader.join(timeout=10)
    if p.returncode != 0:
        raise RuntimeError(f"subprocess failed (exit {p.returncode}): {cmd[0]}")


def _py(code, log, timeout=None, env=None):
    _run([PY, "-c", f"import sys; sys.path.insert(0, r'{APP_DIR}')\n{code}"], log,
         timeout=timeout, env=env)


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
          "--images", images_dir, "--output", seed, "--device", "cpu", "--max-kpts", "1024", "--pair-window", "6", "--max-images", "200"], _slog, timeout=7200)
    sparse_dir = os.path.join(seed, "sparse", "0")
    required = ("cameras.bin", "images.bin", "points3D.bin")
    if not all(os.path.exists(os.path.join(sparse_dir, f)) for f in required):
        # mapper likely couldn't register images past the initial pair (e.g. too
        # little parallax between sampled frames) — fail now, not inside COLMAP's undistorter
        raise RuntimeError(
            "learned-SfM produced no usable reconstruction — the mapper could not "
            "register images beyond an initial pair. Try a clip with more camera "
            "movement/parallax, or a different frame sample.")
    return sparse_dir


def _gpu_free_mib(log=None):
    """Free VRAM per GPU index, from nvidia-smi. [] when there is no NVIDIA GPU.

    Queried at runtime rather than assumed: this recipe has run on a single 48 GB card
    and on a 2x48 GB box, and the right split differs."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception as e:
        if log:
            log(f"GPU probe failed ({e}) — leaving device selection to the trainers", "WARNING")
        return []


def _mcmc_gpu(log):
    """GPU index to pin MCMC to, or None to leave it on the default device.

    Brush is a wgpu app: CUDA_VISIBLE_DEVICES does not move it, and two identical cards
    make WGPU_ADAPTER_NAME useless — it always takes the default adapter (GPU 0). So the
    only lever is MCMC, and moving it to another card is enough to stop the two trainers
    sharing one. With a single GPU they share it, as they always have.
    Override with SPLAT_MCMC_GPU."""
    override = os.environ.get("SPLAT_MCMC_GPU")
    if override and override.strip().lstrip('-').isdigit():
        idx = int(override)
        if idx < 0:   # explicit opt-out: let torch pick, same as a single-GPU box
            log("GPU plan: MCMC left on the default device (SPLAT_MCMC_GPU=-1)")
            return None
        log(f"GPU plan: MCMC pinned to GPU {idx} (SPLAT_MCMC_GPU); Brush on the default adapter")
        return idx

    free = _gpu_free_mib(log)
    if len(free) >= 2:
        # most free VRAM among the non-default cards — Brush keeps GPU 0
        idx = max(range(1, len(free)), key=lambda i: free[i])
        log(f"GPU plan: {len(free)} GPUs — MCMC -> GPU {idx} ({free[idx]:,} MiB free), "
            f"Brush -> GPU 0 ({free[0]:,} MiB free, default adapter)")
        return idx
    if free:
        log(f"GPU plan: single GPU ({free[0]:,} MiB free) — MCMC and Brush share it")
    return None


def _system_profile(log):
    """Probe the GPUs and pick production settings that suit THIS machine.

    The recipe was tuned on one 48 GB card and hardcoded that machine's answers, which
    on smaller hardware means an OOM rather than a slower run. Tiers are deliberately
    coarse — they reproduce the validated settings on a >=24 GB card and degrade below
    that, instead of chasing a formula nobody has measured.

    Note the cap is a CEILING, not a target: MCMC fills toward it and the export prunes
    what stayed transparent, and an over-large cap costs quality (haze/overfit), so these
    scale down with VRAM but never up past the validated 4M.

    Returns {mcmc_gpu, mcmc_cap, brush_res, parallel}.
    """
    free = _gpu_free_mib(log)
    mcmc_gpu = _mcmc_gpu(log)

    # Headroom of the card each trainer will actually get. With 2+ cards they don't
    # share, so each is sized by its own card; on one card they split it.
    if len(free) >= 2:
        mcmc_mib = free[mcmc_gpu] if mcmc_gpu is not None and mcmc_gpu < len(free) else free[0]
        brush_mib, parallel = free[0], True
    elif free:
        # One card. Decide the mode FIRST, because it determines the headroom: run both
        # at once only if the card is big enough to halve, otherwise run them one after
        # the other — where each gets the WHOLE card and keeps full-quality settings.
        parallel = free[0] >= 24_000
        mcmc_mib = brush_mib = free[0] // 2 if parallel else free[0]
    else:
        mcmc_mib = brush_mib = 0
        parallel = False

    def _tier(mib, tiers, fallback):
        for floor, value in tiers:
            if mib >= floor:
                return value
        return fallback

    profile = {
        'mcmc_gpu': mcmc_gpu,
        'mcmc_cap': _tier(mcmc_mib, [(20_000, 4_000_000), (10_000, 2_000_000)], 1_000_000),
        'brush_res': _tier(brush_mib, [(20_000, 3968), (10_000, 2560)], 1600),
        'parallel': parallel,
    }
    log(f"System profile: MCMC cap {profile['mcmc_cap']:,} ({mcmc_mib:,} MiB), "
        f"Brush res {profile['brush_res']} ({brush_mib:,} MiB), "
        f"training {'in parallel' if parallel else 'SEQUENTIALLY (limited VRAM)'}")
    return profile


def _existing_scene(job_dir):
    """Return an already-undistorted scene dir worth resuming from, else None.

    Training is the expensive half (~2.5 hr); learned-SfM + undistort is ~12 min that a
    restart would otherwise repeat verbatim — so a crash (or a trainer that failed for
    environmental reasons) can pick up where it left off instead of starting over."""
    scene = os.path.join(job_dir, "scene")
    s0 = os.path.join(scene, "sparse", "0")
    if all(os.path.exists(os.path.join(s0, f))
           for f in ("cameras.bin", "images.bin", "points3D.bin")):
        return scene
    return None


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
        from pipeline.camera_tracking import extract_camera_poses, export_camera_fbx, video_frame_timing
        tdir = os.path.join(job_dir, "camera_tracking")
        os.makedirs(tdir, exist_ok=True)
        poses = extract_camera_poses(os.path.join(scene, "sparse", "0"))
        v_fps, frame_numbers = video_frame_timing(job_dir, len(poses),
                                                  names=[p['name'] for p in poses])
        if v_fps:
            fps = v_fps
        export_camera_fbx(poses, os.path.join(tdir, "cameras.fbx"), fps, frame_numbers)
        log(f"FBX matchmove camera exported ({len(poses)} frames @ {fps:g} fps) -> camera_tracking/cameras.fbx", "INFO")
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
    brush_path = _resolve_brush_path()
    if not brush_path:
        raise RuntimeError("Brush not found")
    os.makedirs(out_dir, exist_ok=True)
    seed_dir_str = os.fspath(seed_dir)
    out_dir_str = os.fspath(out_dir)
    final = os.path.join(out_dir_str, f"export_{steps}.ply")
    cmd = [str(brush_path), seed_dir_str, "--total-steps", str(steps), "--max-resolution", str(res),
           "--sh-degree", "3", "--growth-grad-threshold", "0.00002", "--growth-select-fraction", "0.15",
           "--refine-every", "100", "--growth-stop-iter", str(growth_stop),
           "--export-path", out_dir_str, "--export-name", "export_{iter}.ply",
           "--export-every", str(max(2000, steps // 4))]
    log(f"$ {' '.join(str(c) for c in cmd)}", "DEBUG")
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=_NO_WINDOW, startupinfo=_hidden_si())
    start = time.time()
    # Stall detection must SCALE with the export interval: exports get slower as the
    # splat densifies, so a fixed threshold kills a perfectly healthy run (it once
    # terminated Brush at step 7.5k/30k because export #1 took 780s vs a flat 900s
    # budget). Allow 3x the slowest interval seen so far, floor 30 min.
    stall_floor, last_interval = 1800, 0
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
        stall_secs = max(stall_floor, int(3 * last_interval))
        if nexports > last_export_count:
            now = time.time()
            last_interval = max(last_interval, now - last_progress_t)
            last_export_count, last_progress_t = nexports, now
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

    scene = _existing_scene(job_dir)
    if scene:
        log("Reusing the existing undistorted scene — skipping learned-SfM + undistort", "INFO")
        status_cb("Reusing existing reconstruction...", 10, "mapping")
    else:
        status_cb("Feature extraction...", 4, "feature_extraction")
        sparse = _learned_sfm(images_dir, job_dir, log, status_cb)

        status_cb("Undistorting...", 10, "mapping")
        scene = _undistort(images_dir, sparse, job_dir, log)

    _build_preview(scene, job_dir, log)   # sparse point cloud preview for the Process tab
    if export_fbx:
        _export_fbx_camera(scene, job_dir, log)

    # Train MCMC and Brush IN PARALLEL — they're independent (same seed), so wall-clock
    # ~halves vs sequential (~4 hr -> ~2.5 hr). Each runs as its own GPU subprocess;
    # threads here just wait on them concurrently. Where those subprocesses land is
    # decided at runtime by _mcmc_gpu() — one card each when the box has two.
    profile = _system_profile(log)
    status_cb("Training MCMC + Brush {} (~2.5 hr)...".format(
        "in parallel" if profile['parallel'] else "sequentially"), 15, "training")
    mcmc = os.path.join(job_dir, "mcmc", "gaussian_splat.ply")
    os.makedirs(os.path.dirname(mcmc), exist_ok=True)
    errors, brush_holder = {}, {}

    mcmc_env = dict(_cuda_env())
    if profile['mcmc_gpu'] is not None:
        mcmc_env["CUDA_VISIBLE_DEVICES"] = str(profile['mcmc_gpu'])

    def _train_mcmc():
        try:
            _py(f"from pipeline.gsplat_mcmc_trainer import train_mcmc\n"
                f"train_mcmc(r'{scene}', total_steps=30000, cap_max={profile['mcmc_cap']}, sh_degree=3, use_lpips=True,\n"
                f"           strategy_name='mcmc', export_opacity_min=0.03, output_ply_path=r'{mcmc}')",
                log, timeout=18000, env=mcmc_env)
        except Exception as e:
            errors['mcmc'] = e

    def _train_brush():
        try:
            brush_holder['ply'] = _brush(scene, os.path.join(job_dir, "brush"), steps=30000,
                                         res=profile['brush_res'], growth_stop=15000,
                                         log=log, timeout=18000)
        except Exception as e:
            errors['brush'] = e

    if profile['parallel']:
        t_mcmc, t_brush = threading.Thread(target=_train_mcmc), threading.Thread(target=_train_brush)
        t_mcmc.start(); t_brush.start()
        t_mcmc.join(); t_brush.join()
    else:
        # Not enough VRAM to hold both at once — same work, ~2x the wall-clock, but it
        # finishes instead of one trainer OOMing the other out.
        _train_mcmc()
        _train_brush()
    if 'brush' in errors:
        raise RuntimeError("parallel training failed — " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    brush_export = brush_holder['ply']

    if 'mcmc' in errors:
        # MCMC failed (e.g. gsplat CUDA JIT build error) but Brush still produced a usable
        # splat — use it alone rather than discarding a good result over the failed half.
        log(f"MCMC training failed ({errors['mcmc']}) — falling back to Brush-only output", "WARNING")
        status_cb("Compressing (Brush-only fallback)...", 98, "training")
        _py(f"from pipeline.compress_ply import compress\n"
            f"compress(r'{brush_export}', r'{out_ply}', max_sh=3)", log, timeout=900)
        status_cb("Production complete (Brush-only fallback)!", 100, None)
        return out_ply

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


def parse_cli_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] in {'draft', 'production'}:
        quality, images_dir, job_dir = argv[0], argv[1], argv[2]
        export_fbx = '--fbx' in argv
        return quality, images_dir, job_dir, export_fbx

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--images', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--fbx', action='store_true')
    args = parser.parse_args(argv)
    quality = 'draft' if args.draft else 'production' if args.production else 'draft'
    return quality, args.images, args.job_dir, args.fbx


if __name__ == '__main__':
    # Detached entry point: either python pipeline/recipes.py <draft|production> <images_dir> <job_dir> [--fbx]
    # or python pipeline/recipes.py --draft/--production --images ... --job-dir ... [--fbx].
    # Runs independently of the Flask server and writes status/output to job_dir, so a
    # server restart/crash never orphans the job.
    quality, images_dir, job_dir, export_fbx = parse_cli_args()
    os.makedirs(job_dir, exist_ok=True)
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
