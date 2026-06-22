"""Gaussian-falloff combine: MCMC foreground + Brush sky/bg.

sigma_mult=1.7 — tightened a notch from 2.0 (user feedback: cars good, blend
nice, asphalt needs a bit more tightening). Net effect:
  car body (d~0.5*base) -> w ~0.92  (was 0.94, ~unchanged)
  car detail (d~base)   -> w ~0.71  (was 0.78, -10%)
  mid asphalt (d~2*base)-> w ~0.25  (was 0.37, -33%)
  open asphalt (d~3*base)-> w ~0.045 (was 0.10, -55%)
"""
import sys, time, shutil
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.combine_splats import combine_fade

MCMC     = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-mcmc/gaussian_splat.ply"
BRUSH    = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-brush/export_30000.ply"
SEED     = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene/sparse/0"
OUT      = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-combine-gauss17.ply"
DOWNLOAD = r"C:/Users/f/Downloads/COMBINE_gauss_sigma17.ply"

t0 = time.time()
def log(m): print(f"[{int(time.time()-t0)}s] {m}", flush=True)

log("=== combine_fade: Gaussian falloff sigma_mult=1.7 ===")
combine_fade(MCMC, BRUSH, SEED, OUT,
             fade_mode='gaussian', sigma_mult=1.7,
             progress=log)
shutil.copy2(OUT, DOWNLOAD)
log(f"DONE -> {DOWNLOAD}")
