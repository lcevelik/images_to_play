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
