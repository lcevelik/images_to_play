import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import run_glomap, recipes


def test_find_brush_binary_falls_back_to_downloads_path(monkeypatch, tmp_path):
    brush_path = tmp_path / "brush-app" / "brush_app.exe"
    brush_path.parent.mkdir(parents=True)
    brush_path.write_bytes(b"dummy")

    def fake_expanduser(path):
        if path.startswith("~\\Downloads"):
            return str(tmp_path)
        return path

    monkeypatch.setattr(run_glomap.os.path, "exists", lambda path: str(path) == str(brush_path))
    monkeypatch.setattr(run_glomap.os.path, "isdir", lambda path: str(path) == str(tmp_path))
    monkeypatch.setattr(run_glomap.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(run_glomap, "which", lambda path: None)

    assert run_glomap.find_brush_binary() == str(brush_path)


def test_recipes_resolve_brush_path_at_runtime(monkeypatch, tmp_path):
    brush_path = tmp_path / "brush-app" / "brush_app.exe"
    brush_path.parent.mkdir(parents=True)
    brush_path.write_bytes(b"dummy")

    monkeypatch.setattr(recipes, "BRUSH", None)
    monkeypatch.setattr(recipes, "find_brush_binary", lambda: str(brush_path))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    final_export = out_dir / "export_100.ply"
    final_export.write_bytes(b"dummy")

    class DummyProcess:
        def __init__(self):
            self.returncode = None
        def poll(self):
            return 0
        def terminate(self):
            return None

    monkeypatch.setattr(recipes.subprocess, "Popen", lambda *args, **kwargs: DummyProcess())
    monkeypatch.setattr(recipes.os.path, "exists", lambda path: str(path) == str(brush_path) or str(path) == str(final_export))
    monkeypatch.setattr(recipes.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(recipes.glob, "glob", lambda pattern: [str(final_export)] if str(pattern).endswith("export_*.ply") else [])

    out = recipes._brush(tmp_path, out_dir, steps=100, res=128, growth_stop=10, log=lambda *args, **kwargs: None)

    assert out == str(final_export)
