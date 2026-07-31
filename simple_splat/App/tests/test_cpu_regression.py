"""CPU-only regression tests — no GPU, no COLMAP, no server, no network.

Covers the pure-Python/numpy parts of the pipeline that have historically
regressed silently: preset loading, the COLMAP binary readers, the PLY
fallback writers, compress_ply's channel-major SH selection, and the binary
FBX conventions.

    cd simple_splat/App && python -m pytest tests/test_cpu_regression.py -q
"""

import math
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import fbx_binary
from pipeline.camera_tracking import extract_camera_poses, export_point_cloud_ply
from pipeline.compress_ply import SH_BAND_COEFFS, compress, verify_roundtrip
from pipeline.gaussian_splat_utils import generate_ply_from_colmap, prune_gaussian_ply
from pipeline.run_glomap import get_app_settings, load_presets
from pipeline.sparse_preview import (_read_cameras_bin, _read_images_bin,
                                     _read_points3D_bin, build_alignment_json)


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------

def _write_colmap_model(sparse_dir, names=('frame_10.jpg', 'frame_2.jpg', 'frame_1.jpg')):
    """Write a minimal but format-exact COLMAP binary model. Returns the poses used."""
    os.makedirs(sparse_dir, exist_ok=True)

    with open(os.path.join(sparse_dir, 'cameras.bin'), 'wb') as f:
        f.write(struct.pack('<Q', 1))
        # cam_id, model_id 1 (PINHOLE, 4 params), width, height, fx fy cx cy
        f.write(struct.pack('<iiQQ', 1, 1, 1920, 1080))
        f.write(struct.pack('<dddd', 1500.0, 1500.0, 960.0, 540.0))

    poses = []
    with open(os.path.join(sparse_dir, 'images.bin'), 'wb') as f:
        f.write(struct.pack('<Q', len(names)))
        for i, name in enumerate(names):
            ang = 0.3 * (i + 1)
            qvec = (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)   # rot about +Y
            tvec = (0.1 * i, 0.2 * i, 1.0 + i)
            poses.append({'name': name, 'qvec': qvec, 'tvec': tvec})
            f.write(struct.pack('<idddddddi', i + 1, *qvec, *tvec, 1))
            f.write(name.encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 2))                             # 2 x 2D points
            for j in range(2):
                f.write(struct.pack('<ddQ', float(j), float(j), j + 1))

    pts = [((1.0, 2.0, 3.0), (255, 0, 0)),
           ((-1.0, 0.5, 2.0), (0, 128, 64)),
           ((0.0, 0.0, 0.0), (10, 20, 30))]
    with open(os.path.join(sparse_dir, 'points3D.bin'), 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))
        for i, (xyz, rgb) in enumerate(pts):
            f.write(struct.pack('<Q', i + 1))
            f.write(struct.pack('<ddd', *xyz))
            f.write(struct.pack('<BBB', *rgb))
            f.write(struct.pack('<d', 0.5))                           # reprojection error
            f.write(struct.pack('<Q', 1))                             # track length
            f.write(struct.pack('<ii', 1, 0))
    return poses, pts


def _write_3dgs_ply(path, n=300, sh_bands=3, opacity=None):
    """Write a standard binary-LE 3DGS PLY. f_rest is channel-major: R|G|B."""
    coeffs = SH_BAND_COEFFS[sh_bands]
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    names += [f'f_rest_{i}' for i in range(coeffs * 3)]
    names += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    rng = np.random.default_rng(0)
    data = np.zeros(n, dtype=np.dtype([(nm, '<f4') for nm in names]))
    data['x'] = rng.uniform(-5, 5, n)
    data['y'] = rng.uniform(-5, 5, n)
    data['z'] = rng.uniform(-5, 5, n)
    data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = 0.2, -0.1, 0.4
    for ch, val in enumerate((1.0, 2.0, 3.0)):          # R=1, G=2, B=3 marker
        for j in range(coeffs):
            data[f'f_rest_{ch * coeffs + j}'] = val
    data['opacity'] = rng.uniform(-2, 4, n) if opacity is None else opacity
    for k in range(3):
        data[f'scale_{k}'] = rng.uniform(-6, -3, n)
    data['rot_0'] = 1.0

    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}']
    header += [f'property float {nm}' for nm in names]
    header += ['end_header', '']
    with open(path, 'wb') as f:
        f.write('\n'.join(header).encode('ascii'))
        f.write(data.tobytes())
    return data


def _read_ply_generic(path):
    """Minimal binary-LE PLY reader for assertions (single vertex element)."""
    with open(path, 'rb') as f:
        header, props, count = b'', [], 0
        while not header.endswith(b'end_header\n'):
            header += f.read(1)
        for line in header.decode('ascii').splitlines():
            if line.startswith('element vertex'):
                count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[2], parts[1]))
        tmap = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint': '<u4', 'int': '<i4'}
        dt = np.dtype([(n, tmap[t]) for n, t in props])
        return np.frombuffer(f.read(count * dt.itemsize), dtype=dt, count=count)


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def test_all_presets_load():
    """Regression: _load_presets pointed at a nonexistent dir for weeks, so every
    job silently used the stale inline dict."""
    presets = load_presets()
    assert set(presets) == {'low', 'medium', 'high', 'quality', 'expert'}
    for name, p in presets.items():
        for key in ('features', 'peak', 'octaves', 'match_ratio', 'tri_angle',
                    'reproj_error', 'dense', 'app'):
            assert key in p, f"{name} preset missing '{key}'"
        assert p['dense'] is False, f"{name} ships dense=True — MVS adds nothing for splats"


def test_get_app_settings():
    for name in ('low', 'medium', 'high', 'quality', 'expert'):
        cfg = get_app_settings(name)
        assert cfg['detail_level'] == name
        assert cfg['training_steps'] > 0
        assert cfg['max_image_size'] > 0
        assert 'description' in cfg


def test_get_app_settings_returns_a_copy():
    """Callers mutate the returned dict (enable_dense/quality_scale) — it must not
    write through to the cached preset."""
    a = get_app_settings('medium')
    a['training_steps'] = -1
    assert get_app_settings('medium')['training_steps'] > 0


# --------------------------------------------------------------------------
# COLMAP binary readers (the pycolmap-free path)
# --------------------------------------------------------------------------

def test_sparse_readers(tmp_path):
    sparse = str(tmp_path / 'sparse' / '0')
    poses, pts = _write_colmap_model(sparse)

    cams = _read_cameras_bin(os.path.join(sparse, 'cameras.bin'))
    assert cams == {1: {'w': 1920, 'h': 1080, 'f': 1500.0}}

    imgs = _read_images_bin(os.path.join(sparse, 'images.bin'))
    assert [i['name'] for i in imgs] == [p['name'] for p in poses]
    assert all(i['cam_id'] == 1 for i in imgs)
    np.testing.assert_allclose(imgs[0]['qvec'], poses[0]['qvec'])
    np.testing.assert_allclose(imgs[0]['tvec'], poses[0]['tvec'])

    xyz, rgb01 = _read_points3D_bin(os.path.join(sparse, 'points3D.bin'))
    np.testing.assert_allclose(xyz, [p[0] for p in pts], rtol=1e-6)
    np.testing.assert_allclose(rgb01 * 255.0, [p[1] for p in pts], atol=1e-3)


def test_generate_ply_from_colmap_needs_no_pycolmap(tmp_path):
    sparse = str(tmp_path / 'sparse' / '0')
    _, pts = _write_colmap_model(sparse)
    out = str(tmp_path / 'point_cloud.ply')

    assert generate_ply_from_colmap(sparse, out) is True
    v = _read_ply_generic(out)
    assert len(v) == len(pts)
    np.testing.assert_allclose(np.c_[v['x'], v['y'], v['z']], [p[0] for p in pts], rtol=1e-6)
    assert [tuple(int(c) for c in row) for row in np.c_[v['red'], v['green'], v['blue']]] \
        == [p[1] for p in pts]


def test_generate_ply_does_not_recenter(tmp_path):
    """Recentering only this cloud puts it out of register with the poses/splat."""
    sparse = str(tmp_path / 'sparse' / '0')
    _write_colmap_model(sparse)
    out = str(tmp_path / 'pc.ply')
    generate_ply_from_colmap(sparse, out)
    v = _read_ply_generic(out)
    assert abs(float(np.mean(np.c_[v['x'], v['y'], v['z']]))) > 1e-6


def test_export_point_cloud_ply_needs_no_pycolmap(tmp_path):
    sparse = str(tmp_path / 'sparse' / '0')
    _, pts = _write_colmap_model(sparse)
    out = str(tmp_path / 'ref.ply')
    assert export_point_cloud_ply(sparse, out) == out
    assert len(_read_ply_generic(out)) == len(pts)


def test_extract_camera_poses_needs_no_pycolmap(tmp_path):
    sparse = str(tmp_path / 'sparse' / '0')
    raw, _ = _write_colmap_model(sparse)
    poses = extract_camera_poses(sparse)

    # natural sort: frame_1 < frame_2 < frame_10 (plain string sort scrambles this)
    assert [p['name'] for p in poses] == ['frame_1.jpg', 'frame_2.jpg', 'frame_10.jpg']
    assert [p['frame'] for p in poses] == [0, 1, 2]

    for p in poses:
        assert (p['width'], p['height']) == (1920, 1080)
        assert p['focal_length'] == pytest.approx(1500.0)
        R = np.array(p['rotation_matrix'])
        assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9
        assert np.linalg.det(R) == pytest.approx(1.0)

    # position must be the camera CENTRE (-R^T t), not the raw COLMAP translation
    src = {r['name']: r for r in raw}['frame_1.jpg']
    w, x, y, z = src['qvec']
    R_w2c = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    expect = -R_w2c.T @ np.array(src['tvec'])
    np.testing.assert_allclose(poses[0]['position'], expect, atol=1e-9)


def test_build_alignment_json(tmp_path):
    sparse = str(tmp_path / 'sparse' / '0')
    _, pts = _write_colmap_model(sparse)
    out = str(tmp_path / 'alignment.json')
    build_alignment_json(sparse, out)

    import json
    data = json.loads(open(out).read())
    assert len(data['points']) == len(pts) * 3
    assert len(data['colors']) == len(pts) * 3
    assert len(data['frustums']) == 3
    assert len(data['frustums'][0]) == 15          # apex + 4 corners, xyz each


# --------------------------------------------------------------------------
# PLY utilities
# --------------------------------------------------------------------------

def test_prune_gaussian_ply(tmp_path):
    src = str(tmp_path / 'in.ply')
    dst = str(tmp_path / 'out.ply')
    # sigmoid: -10 -> 4.5e-5 (drop), -5 -> 0.0067 (keep), 0 -> 0.5, 5 -> 0.993
    op = np.tile(np.array([-10.0, -5.0, 0.0, 5.0], np.float32), 75)
    _write_3dgs_ply(src, n=300, opacity=op)

    total, kept = prune_gaussian_ply(src, dst, opacity_threshold=0.005)
    assert (total, kept) == (300, 225)
    v = _read_ply_generic(dst)
    assert len(v) == 225
    assert float(v['opacity'].min()) == pytest.approx(-5.0)


def test_prune_ignores_non_3dgs_ply(tmp_path):
    """A plain XYZ+RGB point cloud has no 'opacity' — must be a no-op, not a crash."""
    src = str(tmp_path / 'pc.ply')
    sparse = str(tmp_path / 'sparse' / '0')
    _write_colmap_model(sparse)
    generate_ply_from_colmap(sparse, src)
    assert prune_gaussian_ply(src, str(tmp_path / 'out.ply')) == (0, 0)


def test_compress_roundtrip_positions(tmp_path):
    src = str(tmp_path / 'in.ply')
    dst = str(tmp_path / 'out.ply')
    _write_3dgs_ply(src, n=300)
    compress(src, dst, max_sh=3, progress=lambda *a: None)
    assert os.path.getsize(dst) < os.path.getsize(src)
    assert verify_roundtrip(src, dst, progress=lambda *a: None)


def test_compress_sh_truncation_is_channel_major(tmp_path):
    """Regression (green tint): f_rest is channel-major R|G|B, so truncating to
    SH1 must take the first 3 coeffs of EACH channel, not f_rest_0..8 (all red)."""
    src = str(tmp_path / 'in.ply')
    dst = str(tmp_path / 'out.ply')
    _write_3dgs_ply(src, n=300, sh_bands=3)          # R=1.0, G=2.0, B=3.0
    compress(src, dst, max_sh=1, progress=lambda *a: None)

    with open(dst, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.read(1)
        lines = header.decode('ascii').splitlines()
        n_sh = int([l for l in lines if l.startswith('element sh')][0].split()[-1])
        n_chunk = int([l for l in lines if l.startswith('element chunk')][0].split()[-1])
        n_vert = int([l for l in lines if l.startswith('element vertex')][0].split()[-1])
        n_coeff = sum(1 for l in lines if l.startswith('property uchar f_rest_'))
        f.seek(len(header) + n_chunk * 18 * 4 + n_vert * 4 * 4)
        sh = np.frombuffer(f.read(n_sh * n_coeff), np.uint8).reshape(n_sh, n_coeff)

    assert (n_sh, n_coeff) == (300, 9)               # SH1 = 3 coeffs x 3 channels
    q = lambda v: int(np.clip(np.trunc((v / 8.0 + 0.5) * 256), 0, 255))
    assert set(np.unique(sh[:, 0:3])) == {q(1.0)}    # R band
    assert set(np.unique(sh[:, 3:6])) == {q(2.0)}    # G band
    assert set(np.unique(sh[:, 6:9])) == {q(3.0)}    # B band


def test_compress_sh0_drops_all_f_rest(tmp_path):
    src = str(tmp_path / 'in.ply')
    dst = str(tmp_path / 'out.ply')
    _write_3dgs_ply(src, n=300)
    compress(src, dst, max_sh=0, progress=lambda *a: None)
    assert b'element sh' not in open(dst, 'rb').read(4096)
    assert os.path.getsize(src) / os.path.getsize(dst) > 10


# --------------------------------------------------------------------------
# binary FBX camera export
# --------------------------------------------------------------------------

def _fbx_keytimes(data):
    """Decode the first KeyTime array out of a binary FBX (arrays are deflated)."""
    i = data.index(b'KeyTime') + len(b'KeyTime')
    assert data[i:i + 1] == b'l'                     # int64 array
    count, encoding, comp_len = struct.unpack('<III', data[i + 1:i + 13])
    raw = data[i + 13:i + 13 + comp_len]
    if encoding:
        import zlib
        raw = zlib.decompress(raw)
    return list(struct.unpack(f'<{count}q', raw))


def _fbx_poses(n=5):
    return [{'name': f'frame_{i}.jpg', 'frame': i,
             'position': [float(i), 1.0, 2.0],
             'rotation_matrix': np.eye(3).tolist(),
             'rotation_quat': [1.0, 0.0, 0.0, 0.0],
             'focal_length': 1500.0, 'width': 1920, 'height': 1080}
            for i in range(n)]


def test_fbx_is_wellformed_binary_74(tmp_path):
    out = str(tmp_path / 'cam.fbx')
    fbx_binary.write_camera_fbx(_fbx_poses(), out, fps=30)
    data = open(out, 'rb').read()
    assert data[:20] == b'Kaydara FBX Binary  '
    assert struct.unpack('<I', data[23:27])[0] == 7400
    assert fbx_binary.roundtrip_ok(data)


def test_fbx_object_names_use_binary_separator(tmp_path):
    """Regression: ASCII-style 'Class::Name' crashed Blender's importer
    (ValueError: not enough values to unpack). Binary FBX uses Name\\x00\\x01Class."""
    out = str(tmp_path / 'cam.fbx')
    fbx_binary.write_camera_fbx(_fbx_poses(), out, fps=30)
    data = open(out, 'rb').read()
    assert b'TrackingCamera\x00\x01Model' in data
    assert b'\x00\x01NodeAttribute' in data
    assert b'\x00\x01AnimStack' in data
    assert b'::' not in data


def test_fbx_declares_metre_scale(tmp_path):
    """Regression: without UnitScaleFactor=100 the camera imported 100x too small."""
    out = str(tmp_path / 'cam.fbx')
    fbx_binary.write_camera_fbx(_fbx_poses(), out, fps=30)
    data = open(out, 'rb').read()
    i = data.index(b'UnitScaleFactor')
    assert struct.pack('<d', 100.0) in data[i:i + 96]


def test_fbx_declares_source_fps(tmp_path):
    """Regression: keys authored at 30fps in a file with no declared rate got
    resampled onto Blender's 25fps grid (fractional frames, mid-clip drift)."""
    for fps, timemode in ((30, 6), (24, 11), (23.976, 14)):
        out = str(tmp_path / f'cam_{fps}.fbx')
        fbx_binary.write_camera_fbx(_fbx_poses(), out, fps=fps)
        data = open(out, 'rb').read()
        i = data.index(b'CustomFrameRate')
        assert struct.pack('<d', float(fps)) in data[i:i + 96]
        assert fbx_binary._fps_to_timemode(fps)[0] == timemode


def test_fbx_keys_land_on_source_frame_numbers(tmp_path):
    """Video decimated by N must key at frames 0, N, 2N... so the camera lines up
    with the original plate."""
    out = str(tmp_path / 'cam.fbx')
    frames = [0, 3, 6, 9, 12]
    fbx_binary.write_camera_fbx(_fbx_poses(5), out, fps=30, frame_numbers=frames)
    expect = [int(round(fn / 30 * fbx_binary.FBX_KTIME)) for fn in frames]
    assert _fbx_keytimes(open(out, 'rb').read()) == expect


def test_fbx_ntsc_keys_use_the_nominal_integer_grid(tmp_path):
    """Regression (Blender 4.3 verified): importers convert ktime->frame with the
    INTEGER rate and carry the NTSC fraction in fps_base. Authoring an NTSC clip's
    times at 23.976 spaced the keys 1.001 frames apart; the integer grid lands them
    on 1,2,3... while the file still declares 23.976 playback."""
    out = str(tmp_path / 'cam.fbx')
    fbx_binary.write_camera_fbx(_fbx_poses(5), out, fps=24000 / 1001)
    data = open(out, 'rb').read()
    assert _fbx_keytimes(data) == [int(round(n / 24 * fbx_binary.FBX_KTIME)) for n in range(5)]
    i = data.index(b'CustomFrameRate')
    assert struct.pack('<d', 24000 / 1001) in data[i:i + 96]


def test_fbx_rejects_empty_poses(tmp_path):
    with pytest.raises(ValueError):
        fbx_binary.write_camera_fbx([], str(tmp_path / 'cam.fbx'))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
