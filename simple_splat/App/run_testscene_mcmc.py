"""Generalization test: run the winning MCMC recipe on a Mip-NeRF 360 scene.
Usage: python run_testscene_mcmc.py <scene>   (scene = garden | room | ...)
Recipe identical to the parking-lot winner EXCEPT max_size=1600 (probe speed).
"""
import os, sys, time
msvc = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
if os.path.isdir(msvc): os.environ["PATH"] = msvc + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.gsplat_mcmc_trainer import train_mcmc

scene = sys.argv[1] if len(sys.argv) > 1 else "garden"
D   = rf"F:/Codebase/images_to_play/datasets/{scene}"
OUT = rf"F:/Codebase/images_to_play/simple_splat/App/processing/{scene}-mcmc/gaussian_splat.ply"
t0 = time.time()
def log(m, l="INFO"): print(f"[{int(time.time()-t0)}s][{l}] {m}", flush=True)

print(f"=== MCMC foreground on Mip-360 '{scene}' (4M cap, 30k, LPIPS, max_size=1600) ===", flush=True)
train_mcmc(D, total_steps=30000, cap_max=4_000_000, sh_degree=3, use_lpips=True,
           strategy_name='mcmc', export_opacity_min=0.03, max_size=1600,
           output_ply_path=OUT, progress_callback=log)
print("=== DONE MCMC ===", flush=True)
