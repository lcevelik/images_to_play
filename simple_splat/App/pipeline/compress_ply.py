"""Compress a standard 3DGS PLY into the PlayCanvas/SuperSplat compressed-PLY
format (chunk-of-256 quantization + morton sort). Loads natively in the
SuperSplat viewer we already bundle.

Format (binary_little_endian), reimplemented from supersplat-src/src/splat-serialize.ts
(MIT). Per 256-splat chunk: store per-chunk float min/max, then pack each splat to
16 bytes (position 11/10/11, rotation smallest-three 2+10+10+10, scale 11/10/11,
color rgba 8888) + 1 byte per SH coefficient.

Typical result: ~8-20x smaller than raw float PLY, near-identical in the viewer.

CLI: python pipeline/compress_ply.py <in.ply> <out.ply> [--max-sh 3]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.combine_splats import _read_ply

SH_C0 = 0.28209479177387814
SH_BAND_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}   # f_rest coeffs per channel by band


def _pack_unorm(v, bits):
    t = (1 << bits) - 1
    return np.clip(np.floor(v * t + 0.5), 0, t).astype(np.uint32)


def _normalize_chunked(vals, C):
    """Per-chunk min/max. vals shape (C*256,). Returns (mn, mx) with shape (C,)."""
    m = vals.reshape(C, 256)
    mn = m.min(axis=1)
    mx = m.max(axis=1)
    return mn, mx


def _apply_norm(vals, mn, mx, C):
    m = vals.reshape(C, 256)
    lo = mn[:, None]
    hi = mx[:, None]
    span = hi - lo
    out = np.where(span < 0.00001, 0.0, (m - lo) / np.where(span == 0, 1, span))
    out = np.clip(out, 0.0, 1.0)
    return out.reshape(-1)


def _morton_sort(x, y, z):
    def norm10(a):
        lo, hi = a.min(), a.max()
        t = np.zeros_like(a, dtype=np.float64) if hi - lo < 1e-12 else (a - lo) / (hi - lo)
        return np.clip((t * 1024).astype(np.int64), 0, 1023).astype(np.uint64)

    def part1by2(v):
        v = v & np.uint64(0x000003ff)
        v = (v | (v << np.uint64(16))) & np.uint64(0xff0000ff)
        v = (v | (v << np.uint64(8))) & np.uint64(0x0300f00f)
        v = (v | (v << np.uint64(4))) & np.uint64(0x030c30c3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x09249249)
        return v

    xi, yi, zi = norm10(x), norm10(y), norm10(z)
    code = (part1by2(zi) << np.uint64(2)) | (part1by2(yi) << np.uint64(1)) | part1by2(xi)
    return np.argsort(code, kind='stable')


def _pack_rotation(r0, r1, r2, r3):
    n = len(r0)
    q = np.stack([r0, r1, r2, r3], axis=1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-20
    largest = np.argmax(np.abs(q), axis=1)
    sign = np.where(q[np.arange(n), largest] < 0, -1.0, 1.0)
    q *= sign[:, None]
    norm = np.sqrt(2.0) * 0.5
    result = largest.astype(np.uint32)
    for i in range(4):
        mask = (i != largest)
        packed = _pack_unorm(q[:, i] * norm + 0.5, 10)
        result = np.where(mask, (result << np.uint32(10)) | packed, result)
    return result.astype(np.uint32)


def compress(in_ply, out_ply, max_sh=3, progress=print):
    A, props = _read_ply(in_ply)
    N = len(A)
    progress(f"[compress] read {N:,} gaussians from {in_ply}")

    # determine SH bands present
    n_frest = sum(1 for p in props if p.startswith('f_rest_'))
    bands_present = next((b for b in (3, 2, 1, 0) if SH_BAND_COEFFS[b] * 3 <= n_frest and SH_BAND_COEFFS[b] * 3 == n_frest), 0)
    if SH_BAND_COEFFS.get(bands_present, -1) * 3 != n_frest:
        # fall back: pick the largest band whose coeff count fits
        bands_present = max((b for b in (0, 1, 2, 3) if SH_BAND_COEFFS[b] * 3 <= n_frest), default=0)
    out_bands = min(max_sh, bands_present)
    out_coeffs = SH_BAND_COEFFS[out_bands]          # per channel
    progress(f"[compress] SH bands in={bands_present} out={out_bands} ({out_coeffs*3} f_rest coeffs)")

    # morton sort for chunk locality
    order = _morton_sort(A['x'], A['y'], A['z'])
    A = A[order]

    C = (N + 255) // 256
    M = C * 256
    pad = M - N

    def col(name):
        v = np.asarray(A[name], dtype=np.float64)
        if pad:
            v = np.concatenate([v, np.repeat(v[-1], pad)])
        return v

    x, y, z = col('x'), col('y'), col('z')
    s0, s1, s2 = col('scale_0'), col('scale_1'), col('scale_2')
    op = col('opacity')
    r0, r1, r2, r3 = col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')
    # f_dc -> color space
    c0 = col('f_dc_0') * SH_C0 + 0.5
    c1 = col('f_dc_1') * SH_C0 + 0.5
    c2 = col('f_dc_2') * SH_C0 + 0.5

    # per-chunk min/max
    pxmn, pxmx = _normalize_chunked(x, C); pymn, pymx = _normalize_chunked(y, C); pzmn, pzmx = _normalize_chunked(z, C)
    sxmn, sxmx = _normalize_chunked(s0, C); symn, symx = _normalize_chunked(s1, C); szmn, szmx = _normalize_chunked(s2, C)
    crmn, crmx = _normalize_chunked(c0, C); cgmn, cgmx = _normalize_chunked(c1, C); cbmn, cbmx = _normalize_chunked(c2, C)
    # clamp scale ranges (values sometimes at +/-inf)
    for arr in (sxmn, sxmx, symn, symx, szmn, szmx):
        np.clip(arr, -20, 20, out=arr)

    # pack
    pos = (_pack_unorm(_apply_norm(x, pxmn, pxmx, C), 11) << np.uint32(21)) | \
          (_pack_unorm(_apply_norm(y, pymn, pymx, C), 10) << np.uint32(11)) | \
          (_pack_unorm(_apply_norm(z, pzmn, pzmx, C), 11))
    scl = (_pack_unorm(_apply_norm(s0, sxmn, sxmx, C), 11) << np.uint32(21)) | \
          (_pack_unorm(_apply_norm(s1, symn, symx, C), 10) << np.uint32(11)) | \
          (_pack_unorm(_apply_norm(s2, szmn, szmx, C), 11))
    sig = 1.0 / (1.0 + np.exp(-op))
    color = (_pack_unorm(_apply_norm(c0, crmn, crmx, C), 8) << np.uint32(24)) | \
            (_pack_unorm(_apply_norm(c1, cgmn, cgmx, C), 8) << np.uint32(16)) | \
            (_pack_unorm(_apply_norm(c2, cbmn, cbmx, C), 8) << np.uint32(8)) | \
            (_pack_unorm(sig, 8))
    rot = _pack_rotation(r0, r1, r2, r3)

    # chunk float records (18 per chunk)
    chunk = np.stack([
        pxmn, pymn, pzmn, pxmx, pymx, pzmx,
        sxmn, symn, szmn, sxmx, symx, szmx,
        crmn, cgmn, cbmn, crmx, cgmx, cbmx,
    ], axis=1).astype('<f4')

    vert = np.stack([pos[:N], rot[:N], scl[:N], color[:N]], axis=1).astype('<u4')

    # SH bytes. f_rest storage is CHANNEL-MAJOR: R coeffs [0:in_coeffs],
    # G [in_coeffs:2*in_coeffs], B [2*in_coeffs:]. When truncating bands
    # (out_coeffs < in_coeffs) we must take the first out_coeffs of EACH channel —
    # taking f_rest_0..(out*3) contiguously grabs all-red coeffs (green-tint bug).
    sh_bytes = None
    if out_coeffs > 0:
        in_coeffs = SH_BAND_COEFFS[bands_present]   # per channel, in the source file
        sh_cols = []
        for ch in range(3):
            for j in range(out_coeffs):
                v = col(f'f_rest_{ch * in_coeffs + j}')[:N]
                q = np.clip(np.trunc((v / 8.0 + 0.5) * 256), 0, 255).astype(np.uint8)
                sh_cols.append(q)
        sh_bytes = np.stack(sh_cols, axis=1)        # (N, coeffs*3) uint8

    # header
    chunk_props = ['min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z',
                   'min_scale_x', 'min_scale_y', 'min_scale_z', 'max_scale_x', 'max_scale_y', 'max_scale_z',
                   'min_r', 'min_g', 'min_b', 'max_r', 'max_g', 'max_b']
    lines = ['ply', 'format binary_little_endian 1.0', 'comment compressed by compress_ply.py',
             f'element chunk {C}']
    lines += [f'property float {p}' for p in chunk_props]
    lines += [f'element vertex {N}']
    lines += [f'property uint {p}' for p in ('packed_position', 'packed_rotation', 'packed_scale', 'packed_color')]
    if out_coeffs > 0:
        lines += [f'element sh {N}']
        lines += [f'property uchar f_rest_{i}' for i in range(out_coeffs * 3)]
    lines += ['end_header', '']
    header = '\n'.join(lines).encode('ascii')

    with open(out_ply, 'wb') as f:
        f.write(header)
        f.write(chunk.tobytes())
        f.write(vert.tobytes())
        if sh_bytes is not None:
            f.write(sh_bytes.tobytes())

    sz_in = os.path.getsize(in_ply)
    sz_out = os.path.getsize(out_ply)
    progress(f"[compress] {sz_in/1e6:.1f} MB -> {sz_out/1e6:.1f} MB ({sz_in/sz_out:.1f}x) -> {out_ply}")
    return out_ply


def verify_roundtrip(src_ply, comp_ply, progress=print):
    """Decode the compressed positions and compare to the source (morton-aligned).
    Proves the bit-packing/field-order is correct without a GPU/viewer."""
    A, _ = _read_ply(src_ply)
    order = _morton_sort(A['x'], A['y'], A['z'])
    sx, sy, sz = A['x'][order], A['y'][order], A['z'][order]
    N = len(A)

    with open(comp_ply, 'rb') as f:
        # parse header
        hdr = b''
        while b'end_header\n' not in hdr:
            hdr += f.read(1)
        text = hdr.decode('ascii')
        C = int([l for l in text.splitlines() if l.startswith('element chunk')][0].split()[-1])
        chunk = np.frombuffer(f.read(C * 18 * 4), dtype='<f4').reshape(C, 18)
        vert = np.frombuffer(f.read(N * 16), dtype='<u4').reshape(N, 4)

    pos = vert[:, 0]
    x11 = ((pos >> 21) & 2047).astype(np.float64) / 2047
    y10 = ((pos >> 11) & 1023).astype(np.float64) / 1023
    z11 = (pos & 2047).astype(np.float64) / 2047
    ci = np.arange(N) // 256
    dx = x11 * (chunk[ci, 3] - chunk[ci, 0]) + chunk[ci, 0]
    dy = y10 * (chunk[ci, 4] - chunk[ci, 1]) + chunk[ci, 1]
    dz = z11 * (chunk[ci, 5] - chunk[ci, 2]) + chunk[ci, 2]

    ext = float(max(sx.max() - sx.min(), sy.max() - sy.min(), sz.max() - sz.min()))
    err = np.sqrt((dx - sx[:N]) ** 2 + (dy - sy[:N]) ** 2 + (dz - sz[:N]) ** 2)
    progress(f"[verify] position error vs source: mean={err.mean():.6f} max={err.max():.6f} "
             f"(scene extent={ext:.3f}, mean err={err.mean()/ext*100:.4f}% of extent)")
    return err.mean() / ext


if __name__ == '__main__':
    if sys.argv[1] == '--verify':
        verify_roundtrip(sys.argv[2], sys.argv[3])
    else:
        inp, outp = sys.argv[1], sys.argv[2]
        max_sh = int(sys.argv[sys.argv.index('--max-sh') + 1]) if '--max-sh' in sys.argv else 3
        compress(inp, outp, max_sh=max_sh)
