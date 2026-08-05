import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def test_prepare_input_images_extracts_from_video_when_folder_empty(tmp_path):
    job_folder = tmp_path / "job"
    job_folder.mkdir()
    images_folder = job_folder / "images"
    images_folder.mkdir()
    video_path = job_folder / "clip.mp4"
    video_path.write_bytes(b"dummy")

    with patch.object(app, "extract_frames_from_video", return_value=3) as mock_extract:
        prepared = app.ensure_input_images(
            str(job_folder),
            str(images_folder),
            frame_interval=10,
            max_frames=1000,
            log_fn=lambda *args, **kwargs: None,
        )

    assert prepared is True
    mock_extract.assert_called_once_with(
        str(video_path),
        str(images_folder),
        frame_interval=10,
        max_frames=1000,
        job_dir=str(job_folder),
    )
