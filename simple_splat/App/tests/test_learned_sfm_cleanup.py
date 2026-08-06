import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline.learned_sfm as learned_sfm


def test_reset_output_dir_removes_stale_sfm_artifacts(tmp_path):
    output_dir = tmp_path / 'seed'
    output_dir.mkdir()
    (output_dir / 'database.db').write_bytes(b'dummy')
    (output_dir / 'sparse').mkdir()
    (output_dir / 'sparse' / '0').mkdir()
    (output_dir / 'sparse' / '0' / 'cameras.bin').write_text('old')

    learned_sfm.reset_output_dir(str(output_dir))

    assert not (output_dir / 'database.db').exists()
    assert not (output_dir / 'sparse' / '0' / 'cameras.bin').exists()
    assert (output_dir / 'sparse').exists()


def test_iter_candidate_pairs_uses_a_window_for_large_sets():
    pairs = list(learned_sfm.iter_candidate_pairs(10, pair_window=2))

    assert (0, 1) in pairs
    assert any(b - a > 2 for a, b in pairs)
    assert all(b - a <= 8 for a, b in pairs)
    assert len(pairs) == len(set(pairs))


def test_sanitize_matches_removes_invalid_and_duplicate_rows():
    matches = np.array([[0, 3], [0, 3], [1, 4], [-1, 2]], dtype=np.int64)

    cleaned = learned_sfm._sanitize_matches(matches)

    assert cleaned.dtype == np.uint32
    assert cleaned.tolist() == [[0, 3], [1, 4]]


def test_select_image_subset_keeps_evenly_spaced_samples(tmp_path):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    for idx in range(10):
        (image_dir / f'frame_{idx:02d}.jpg').write_bytes(b'img')

    selected = learned_sfm.select_image_subset(str(image_dir), max_images=4)

    assert len(selected) == 4
    assert [Path(p).name for p in selected] == [
        'frame_00.jpg',
        'frame_03.jpg',
        'frame_06.jpg',
        'frame_09.jpg',
    ]


def test_evenly_spaced_handles_degenerate_caps():
    items = list(range(10))

    assert learned_sfm._evenly_spaced(items, 0) == items     # 0 == no cap
    assert learned_sfm._evenly_spaced(items, -5) == items
    assert learned_sfm._evenly_spaced(items, 1) == [0]       # must not divide by zero
    assert learned_sfm._evenly_spaced(items, 20) == items
    assert learned_sfm._evenly_spaced([], 4) == []


def test_cli_forwards_max_images(monkeypatch, tmp_path):
    """--max-images must reach run_learned_sfm; it was hardcoded to 200."""
    captured = {}

    def _fake(image_dir, output_dir, **kw):
        captured.update(kw)
        return str(tmp_path / 'sparse' / '0')

    monkeypatch.setattr(learned_sfm, 'run_learned_sfm', _fake)
    rc = learned_sfm.main(['--images', str(tmp_path), '--output', str(tmp_path),
                           '--max-images', '37'])

    assert rc == 0
    assert captured['max_images'] == 37


def test_cli_exits_nonzero_when_mapping_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(learned_sfm, 'run_learned_sfm', lambda *a, **k: None)

    assert learned_sfm.main(['--images', str(tmp_path), '--output', str(tmp_path)]) == 1
