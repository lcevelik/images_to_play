"""Post-process floater cleanup for a 3DGS PLY — no retraining.

Targets the two classic textureless-region artifacts (sky blobs + streaks):
  1. ANISOTROPY prune — drop "needle" Gaussians where one scale axis is hugely
     longer than the others (those are the spiky streaks).
  2. STATISTICAL OUTLIER removal (SOR) — drop Gaussians whose local neighbourhood
     is sparse (isolated floaters sitting away from the real scene).

Preserves all PLY properties (SH, opacity, scale, rot) so the result loads in any
splat viewer. CPU-only; safe to run while a GPU job trains.
"""
import numpy as np


def _read_ply(path):
    f = open(path, 'rb')
    props, n = [], 0
    while True:
        l = f.readline().decode('ascii', 'replace').rstrip()
        if l.startswith('element vertex'):
            n = int(l.split()[-1])
        elif l.startswith('property'):
            props.append(l.split()[-1])
        elif l == 'end_header':
            break
    off = f.tell(); f.close()
    a = np.fromfile(path, dtype=np.dtype([(p, '<f4') for p in props]), offset=off, count=n)
    return a, props


def _write_ply(path, a, props):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(a)}\n"
              + ''.join(f"property float {p}\n" for p in props)
              + "end_header\n")
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(a.tobytes())


def clean_floaters(in_ply, out_ply, aniso_max=25.0, sor_k=16, sor_std=2.0,
                   bbox_k=0.0, progress=print):
    a, props = _read_ply(in_ply)
    n0 = len(a)
    keep = np.ones(n0, dtype=bool)

    # 1. needle prune: a SPIKE has its longest axis >> the MIDDLE axis (a rod).
    # A legit flat surface Gaussian is a pancake — its two largest axes are
    # similar (only the smallest is thin), so max/mid ~1 and it's spared. Using
    # max/min here would wrongly delete every flat surface splat.
    sc = np.exp(np.stack([a['scale_0'], a['scale_1'], a['scale_2']], 1))
    ss = np.sort(sc, axis=1)                       # ascending [s0,s1,s2]
    needle = ss[:, 2] / np.maximum(ss[:, 1], 1e-9)  # longest / middle
    m = needle <= aniso_max
    progress(f"[clean] needle(max/mid)>{aniso_max}: drop {int((~m).sum()):,}")
    keep &= m

    xyz = np.stack([a['x'], a['y'], a['z']], 1)

    # 2. optional robust bounding box (drop far outliers like a high sky clump)
    if bbox_k > 0:
        med = np.median(xyz[keep], 0)
        q1, q3 = np.percentile(xyz[keep], [25, 75], 0)
        iqr = np.maximum(q3 - q1, 1e-6)
        within = np.all(np.abs(xyz - med) <= bbox_k * iqr, axis=1)
        progress(f"[clean] bbox (±{bbox_k}*IQR): drop {int((keep & ~within).sum()):,}")
        keep &= within

    # 3. statistical outlier removal on survivors (isolated floaters)
    from scipy.spatial import cKDTree
    idx = np.where(keep)[0]
    tree = cKDTree(xyz[idx])
    d, _ = tree.query(xyz[idx], k=sor_k + 1)   # +1 includes self (dist 0)
    md = d[:, 1:].mean(1)
    thr = md.mean() + sor_std * md.std()
    sor_keep = md <= thr
    progress(f"[clean] SOR (k={sor_k}, std={sor_std}, thr={thr:.4f}): drop {int((~sor_keep).sum()):,}")
    keep[idx[~sor_keep]] = False

    a2 = a[keep]
    _write_ply(out_ply, a2, props)
    progress(f"[clean] {n0:,} -> {len(a2):,} ({n0 - len(a2):,} removed, "
             f"{100 * (n0 - len(a2)) / n0:.1f}%) -> {out_ply}")
    return out_ply


if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--aniso-max", type=float, default=25.0)
    p.add_argument("--sor-k", type=int, default=16)
    p.add_argument("--sor-std", type=float, default=2.0)
    p.add_argument("--bbox-k", type=float, default=0.0, help="0=off; e.g. 8 drops far outliers")
    a = p.parse_args()
    t0 = time.time()
    clean_floaters(a.input, a.output, a.aniso_max, a.sor_k, a.sor_std, a.bbox_k,
                   progress=lambda m: print(f"[{int(time.time()-t0)}s] {m}", flush=True))
