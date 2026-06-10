"""Quick test v2: train 1K steps with ALL fixes (intrinsics + far_plane + viewmat)."""
import os, sys, time
sys.path.insert(0, r"F:\Codebase\images_to_play\simple_splat\App")
os.environ["PATH"] = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" + ";" + os.environ.get("PATH", "")

from gsplat_mcmc_trainer import train_mcmc

WORK_DIR = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc"
OUTPUT = r"F:\Codebase\images_to_play\test_data\benchmark\mcmc\gaussian_splat_v2.ply"

print("Starting quick 1K test v2 (all fixes)...", flush=True)
start = time.time()
result = train_mcmc(
    WORK_DIR,
    total_steps=1000,
    cap_max=1_500_000,
    progress_callback=lambda msg, lvl="INFO": print(f"[{lvl}] {msg}", flush=True),
    output_ply_path=OUTPUT,
)
elapsed = time.time() - start
if result:
    size_mb = os.path.getsize(result) / (1024*1024)
    print(f"\nSUCCESS: {result} ({size_mb:.1f} MB, {elapsed:.1f}s)")
else:
    print(f"\nFAILED after {elapsed:.1f}s")
