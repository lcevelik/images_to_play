"""Aniso-prune MCMC asphalt spikes, then re-combine with Brush sky/bg.

First attempt at fixing the foreground tradeoff (MCMC cars sharp + asphalt
spikes vs Brush asphalt clean + cars noisy): drop MCMC's needle Gaussians
(needle = long_axis/middle_axis > aniso_max) and re-run the density crossfade.
"""
import sys, time, shutil
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.clean_floaters import clean_floaters
from pipeline.combine_splats import combine_fade

MCMC_RAW    = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-mcmc/gaussian_splat.ply"
MCMC_PRUNED = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-mcmc/gaussian_splat_aniso8.ply"
BRUSH       = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-brush/export_30000.ply"
SEED        = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene/sparse/0"
OUT         = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-combine-aniso8.ply"
DOWNLOAD    = r"C:/Users/f/Downloads/COMBINE_anisoprune_v1.ply"

t0 = time.time()
def log(m): print(f"[{int(time.time()-t0)}s] {m}", flush=True)

log("=== STEP 1: aniso-prune MCMC (aniso_max=8, SOR off — only kill needles) ===")
# sor_std=1e9 effectively disables SOR (don't want to lose legit isolated detail)
clean_floaters(MCMC_RAW, MCMC_PRUNED, aniso_max=8.0, sor_k=16, sor_std=1e9,
               bbox_k=0.0, progress=log)

log("=== STEP 2: combine_fade (pruned MCMC + Brush) ===")
combine_fade(MCMC_PRUNED, BRUSH, SEED, OUT, progress=log)

log(f"=== STEP 3: copy to Downloads ===")
shutil.copy2(OUT, DOWNLOAD)
log(f"DONE -> {DOWNLOAD}")
