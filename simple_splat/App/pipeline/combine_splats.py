"""Combine two 3DGS splats trained on the SAME seed, by spatial role.

Insight: SfM seed points only exist on real *textured* surfaces (cars, lot,
building, trees) — never in the sky. So:
  - near a seed point  -> real surface -> take the SHARP splat (MCMC)
  - far from every seed -> sky / void  -> take the CLEAN splat (Brush)

Both inputs must be in the same coordinate frame (same seed). Properties may be
in a different ORDER between the two files; we realign by name before merging.
"""
import numpy as np
from pipeline.clean_floaters import _read_ply, _write_ply


def _seed_xyz(sparse_dir):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse_dir)
    return np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)


def combine(sharp_ply, clean_ply, sparse_dir, out_ply,
            mcmc_mult=8.0, brush_mult=None, progress=print):
    """MCMC for dist<=mcmc_mult*spacing (foreground); Brush for dist>brush_mult*spacing
    (sky). brush_mult>mcmc_mult drops the ambiguous boundary SHELL that causes the
    tree halo. brush_mult=None -> single hard cut (old behaviour)."""
    A, pa = _read_ply(sharp_ply)    # MCMC — sharp foreground source
    B, pb = _read_ply(clean_ply)    # Brush — clean sky/background source
    assert set(pa) == set(pb), "property sets differ; cannot merge"

    seed = _seed_xyz(sparse_dir)
    from scipy.spatial import cKDTree
    tree = cKDTree(seed)
    da, _ = tree.query(np.stack([A['x'], A['y'], A['z']], 1))
    db, _ = tree.query(np.stack([B['x'], B['y'], B['z']], 1))

    dnn, _ = tree.query(seed, k=2)
    spacing = float(np.median(dnn[:, 1]))          # typical seed point spacing
    t_mcmc = spacing * mcmc_mult
    t_brush = spacing * (brush_mult if brush_mult else mcmc_mult)
    progress(f"[combine] seed spacing={spacing:.4f} | MCMC<= {t_mcmc:.3f} ({mcmc_mult}x), "
             f"Brush> {t_brush:.3f} ({brush_mult or mcmc_mult}x), gap drops the boundary shell")
    for label, d in (('MCMC', da), ('Brush', db)):
        ps = np.percentile(d, [50, 90, 99])
        progress(f"[combine] {label} dist-to-seed: p50={ps[0]:.3f} p90={ps[1]:.3f} p99={ps[2]:.3f}")

    a_keep = da <= t_mcmc           # MCMC near seed  -> foreground (sharp)
    b_keep = db > t_brush           # Brush far away  -> sky/background (clean)
    progress(f"[combine] MCMC foreground: {int(a_keep.sum()):,}/{len(A):,} | "
             f"Brush sky/bg: {int(b_keep.sum()):,}/{len(B):,}")

    A2 = A[a_keep]
    Bsel = B[b_keep]
    B2 = np.empty(len(Bsel), dtype=A.dtype)   # realign Brush fields to MCMC order
    for name in pa:
        B2[name] = Bsel[name]

    out = np.concatenate([A2, B2])
    _write_ply(out_ply, out, pa)
    progress(f"[combine] -> {len(out):,} Gaussians "
             f"({len(A2):,} MCMC foreground + {len(B2):,} Brush sky) -> {out_ply}")
    return out_ply


if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--sharp", required=True, help="MCMC PLY (foreground source)")
    p.add_argument("--clean", required=True, help="Brush PLY (sky/background source)")
    p.add_argument("--sparse", required=True, help="sparse/0 dir (seed points)")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--mcmc-mult", type=float, default=8.0, help="MCMC kept within this x seed-spacing")
    p.add_argument("--brush-mult", type=float, default=None, help="Brush kept beyond this x seed-spacing (>mcmc-mult opens a gap)")
    a = p.parse_args()
    t0 = time.time()
    combine(a.sharp, a.clean, a.sparse, a.out, a.mcmc_mult, a.brush_mult,
            progress=lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True))
