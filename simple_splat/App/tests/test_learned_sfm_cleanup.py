import os
import sys
import tempfile
from pathlib import Path

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
    assert (8, 9) in pairs
    assert (0, 3) not in pairs
    assert max(b - a for a, b in pairs) <= 2
