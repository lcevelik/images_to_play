"""Generalization test: gaussian-fade combine (sigma_mult=1.7) on a Mip-360 scene.
Usage: python run_testscene_combine.py <scene>
"""
import sys, time, shutil
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.combine_splats import combine_fade

scene = sys.argv[1] if len(sys.argv) > 1 else "garden"
MCMC  = rf"F:/Codebase/images_to_play/simple_splat/App/processing/{scene}-mcmc/gaussian_splat.ply"
BRUSH = rf"F:/Codebase/images_to_play/simple_splat/App/processing/{scene}-brush/export_30000.ply"
SEED  = rf"F:/Codebase/images_to_play/datasets/{scene}/sparse/0"
OUT   = rf"F:/Codebase/images_to_play/simple_splat/App/processing/{scene}-combine-gauss17.ply"
DL    = rf"C:/Users/f/Downloads/{scene.upper()}_combine_sigma17.ply"
t0 = time.time()
def log(m): print(f"[{int(time.time()-t0)}s] {m}", flush=True)

log(f"=== combine_fade gaussian sigma_mult=1.7 on '{scene}' ===")
combine_fade(MCMC, BRUSH, SEED, OUT, fade_mode='gaussian', sigma_mult=1.7, progress=log)
shutil.copy2(OUT, DL)
log(f"DONE -> {DL}")
