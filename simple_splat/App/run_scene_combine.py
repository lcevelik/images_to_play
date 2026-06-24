import sys, time
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.combine_splats import combine_fade

MCMC  = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-mcmc/gaussian_splat.ply"   # sharp foreground
BRUSH = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-brush/export_30000.ply"      # clean sky/bg
SEED  = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene/sparse/0"
OUT   = r"F:/Codebase/images_to_play/simple_splat/App/processing/scene-combine.ply"
t0 = time.time()
def log(m): print(f"[{int(time.time()-t0)}s] {m}", flush=True)

print("=== combine_fade: MCMC foreground + Brush sky, density crossfade ===", flush=True)
combine_fade(MCMC, BRUSH, SEED, OUT, progress=log)
print("=== DONE COMBINE ===", flush=True)
