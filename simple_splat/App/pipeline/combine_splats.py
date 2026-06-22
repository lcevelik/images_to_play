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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def combine_fade(sharp_ply, clean_ply, sparse_dir, out_ply, k=6,
                 lo_mult=2.0, hi_mult=12.0, min_opacity=0.02, fade_sharp=True,
                 brush_needle_max=0.0, brush_white_max=1.0,
                 fade_mode='smoothstep', sigma_mult=2.0, progress=print):
    """Smooth crossfade by LOCAL SEED DENSITY (no hard cut -> no seam/halo).

    Density proxy = distance to the k-th nearest seed point (small = dense surface,
    large = sparse/sky). A weight w(=1 dense .. 0 sparse) scales the SHARP splat's
    opacity by w and the CLEAN splat's by (1-w). Faded-out Gaussians
    (final opacity < min_opacity) are dropped.

    fade_mode:
      'smoothstep' (default) — plateau at 1 for d<lo, smoothstep down, plateau at 0 for d>hi.
                              Tune via lo_mult/hi_mult.
      'gaussian'             — w = exp(-(d/sigma)^2). No plateaus, continuous everywhere.
                              Tune via sigma_mult (sigma = sigma_mult * base).
    """
    A, pa = _read_ply(sharp_ply)    # MCMC — sharp
    B, pb = _read_ply(clean_ply)    # Brush — clean sky
    assert set(pa) == set(pb), "property sets differ"

    seed = _seed_xyz(sparse_dir)
    from scipy.spatial import cKDTree
    tree = cKDTree(seed)
    # density scale from the seeds' own k-th NN distance
    dks, _ = tree.query(seed, k=k + 1)
    base = float(np.median(dks[:, -1]))

    if fade_mode == 'gaussian':
        sigma = base * sigma_mult
        progress(f"[fade] mode=gaussian k={k} base={base:.4f} sigma={sigma:.3f} (continuous, no plateaus)")
        def fg_weight(xyz):
            dk, _ = tree.query(xyz, k=k)
            d = dk[:, -1]
            return np.exp(-(d / sigma) ** 2)
    else:
        lo, hi = base * lo_mult, base * hi_mult
        progress(f"[fade] mode=smoothstep k={k} base={base:.4f} -> fade {lo:.3f}..{hi:.3f}")
        def fg_weight(xyz):
            dk, _ = tree.query(xyz, k=k)
            d = dk[:, -1]
            t = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            return 1.0 - (t * t * (3 - 2 * t))      # smoothstep, 1 near dense seeds

    # If the sharp (MCMC) side is sky-masked it has no sky junk — keep ALL of it
    # (including its dense soft layer) and fade in ONLY Brush for the sky.
    if fade_sharp:
        wA = fg_weight(np.stack([A['x'], A['y'], A['z']], 1))
    else:
        wA = np.ones(len(A), dtype=np.float64)
        progress("[fade] keep_all_sharp: MCMC kept in full (sky-masked); only Brush is faded")
    wB = 1.0 - fg_weight(np.stack([B['x'], B['y'], B['z']], 1))   # Brush keeps sky/sparse

    def apply(arr, w):
        op = _sigmoid(arr['opacity']) * w
        arr = arr.copy()
        arr['opacity'] = _logit(op).astype(np.float32)
        return arr, op
    A2, opA = apply(A, wA)
    B2, opB = apply(B, wB)
    A2 = A2[opA >= min_opacity]
    B2sel = B2[opB >= min_opacity]
    progress(f"[fade] MCMC kept {len(A2):,}/{len(A):,} | Brush kept {len(B2sel):,}/{len(B):,}")

    B3 = np.empty(len(B2sel), dtype=A.dtype)     # realign Brush fields to MCMC order
    for name in pa:
        B3[name] = B2sel[name]

    # Per-component cleaning: the Brush side is now ONLY sky/background, which must
    # be smooth -> ANY needle Gaussian there is an artifact. Prune them hard. (This
    # would damage a textured foreground, but is exactly right on sky.) MCMC is
    # left untouched — it owns the clean foreground.
    if brush_needle_max and brush_needle_max > 0:
        bsc = np.exp(np.stack([B3['scale_0'], B3['scale_1'], B3['scale_2']], 1))
        bss = np.sort(bsc, axis=1)
        bnd = bss[:, 2] / np.maximum(bss[:, 1], 1e-9)
        bkeep = bnd <= brush_needle_max
        progress(f"[fade] Brush(sky) needle prune (max/mid>{brush_needle_max}): drop {int((~bkeep).sum()):,}")
        B3 = B3[bkeep]

    # WHITE-bloom prune: real sky is BLUE, so a sky Gaussian whose colour is
    # whitish (high RED channel — blue sky has low red) is a halo/bloom artifact.
    # SH DC -> colour: c = 0.5 + 0.28209*f_dc. Drop sky Gaussians with red > thresh.
    if brush_white_max and brush_white_max < 1.0:
        red = 0.5 + 0.28209 * B3['f_dc_0']
        wkeep = red <= brush_white_max
        progress(f"[fade] Brush(sky) white-bloom prune (red>{brush_white_max}): drop {int((~wkeep).sum()):,}")
        B3 = B3[wkeep]

    out = np.concatenate([A2, B3])
    _write_ply(out_ply, out, pa)
    progress(f"[fade] -> {len(out):,} Gaussians -> {out_ply}")
    return out_ply


if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--sharp", required=True, help="MCMC PLY (foreground source)")
    p.add_argument("--clean", required=True, help="Brush PLY (sky/background source)")
    p.add_argument("--sparse", required=True, help="sparse/0 dir (seed points)")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--fade", action="store_true", help="smooth density crossfade (recommended)")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--lo-mult", type=float, default=2.0)
    p.add_argument("--hi-mult", type=float, default=12.0)
    p.add_argument("--min-opacity", type=float, default=0.02)
    p.add_argument("--keep-all-sharp", action="store_true", help="don't fade MCMC (use when it's sky-masked/clean)")
    p.add_argument("--mcmc-mult", type=float, default=8.0)
    p.add_argument("--brush-mult", type=float, default=None)
    p.add_argument("--brush-needle-max", type=float, default=0.0, help="hard-prune Brush(sky) needles above this max/mid (0=off)")
    p.add_argument("--brush-white-max", type=float, default=1.0, help="drop Brush(sky) Gaussians with red colour above this (1=off; ~0.6 kills white blooms)")
    a = p.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True)
    if a.fade:
        combine_fade(a.sharp, a.clean, a.sparse, a.out, a.k, a.lo_mult, a.hi_mult,
                     a.min_opacity, fade_sharp=not a.keep_all_sharp,
                     brush_needle_max=a.brush_needle_max, brush_white_max=a.brush_white_max, progress=log)
    else:
        combine(a.sharp, a.clean, a.sparse, a.out, a.mcmc_mult, a.brush_mult, progress=log)
