import os, sys, time
msvc=r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
if os.path.isdir(msvc): os.environ["PATH"]=msvc+os.pathsep+os.environ.get("PATH","")
sys.path.insert(0, r"F:/Codebase/images_to_play/simple_splat/App")
from pipeline.gsplat_mcmc_trainer import train_mcmc
D=r"F:/Codebase/images_to_play/simple_splat/App/processing/learned-21k"            # has sparse/0 + images + masks
OUT=r"F:/Codebase/images_to_play/simple_splat/App/processing/learned-21k-skymask/gaussian_splat.ply"
t0=time.time()
def log(m,l="INFO"): print(f"[{int(time.time()-t0)}s][{l}] {m}",flush=True)
print("=== MCMC 4M + SKY MASK + anti-flicker + LPIPS ===",flush=True)
train_mcmc(D, total_steps=30000, cap_max=4_000_000, sh_degree=3, use_lpips=True,
           strategy_name='mcmc', export_opacity_min=0.03, output_ply_path=OUT, progress_callback=log)
print("=== DONE ===",flush=True)
