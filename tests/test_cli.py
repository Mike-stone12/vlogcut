from __future__ import annotations

import json

import pytest

from vlogcut import cli
from vlogcut.pipeline import find_footage


def test_find_footage_filters_and_sorts(tmp_path):
    for name in ["b.MOV", "a.mp4", "c.MKV", ".hidden.mp4", "photo.JPG", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.mp4").write_bytes(b"x")
    assert [p.name for p in find_footage(tmp_path)] == ["a.mp4", "b.MOV", "c.MKV"]


def test_parser_defaults():
    args = cli.build_parser().parse_args(["cut", "day1"])
    assert (args.target, args.strategy, args.background) == (180.0, "coverage", "black")
    assert not args.no_subtitles and not args.no_render
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["cut", "day1", "--strategy", "random"])


def test_cli_errors_are_one_line_and_exit_1(tmp_path, capsys):
    assert cli.main(["cut", str(tmp_path / "missing")]) == 1
    assert "is not a folder" in capsys.readouterr().err
    assert cli.main(["cut", str(tmp_path)]) == 1
    assert "no video files" in capsys.readouterr().err
    (tmp_path / "x.mp4").write_bytes(b"x")
    assert cli.main(["cut", str(tmp_path), "--music", str(tmp_path / "nope.mp3")]) == 1
    assert "music file not found" in capsys.readouterr().err


@pytest.mark.ffmpeg
def test_only_broken_files_is_a_clean_error(tmp_path, capsys):
    (tmp_path / "broken.mp4").write_bytes(b"\x00" * 64)
    assert cli.main(["cut", str(tmp_path), "-w", str(tmp_path / "ws")]) == 1
    assert "usable footage" in capsys.readouterr().err


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_cut_end_to_end(make_clip, tmp_path, capsys, monkeypatch):
    day = tmp_path / "day"
    day.mkdir()
    for i, hour in enumerate([9, 12, 18]):
        clip = make_clip(
            f"IMG_{i}.mp4", duration=3, size="360x640",
            metadata={"creation_time": f"2026-10-01T{hour:02d}:00:00Z"},
        )  # fmt: skip
        clip.rename(day / clip.name)
    (day / "broken.mov").write_bytes(b"\x00" * 64)
    ws = tmp_path / "ws"

    assert cli.main(["cut", str(day), "-t", "9", "-w", str(ws)]) == 0
    out = capsys.readouterr().out
    assert "footage   3 clips" in out and "skipped   broken.mov" in out
    preview = ws / "preview_coverage_9s.mp4"
    assert preview.is_file() and f"preview   {preview}" in out
    edl = json.loads((ws / "edl" / "coverage_9s.json").read_text(encoding="utf-8"))
    assert len(edl["clips"]) == 3  # 9 s at 3 s pacing: one clip per moment of the day
    assert (ws / "shots.json").is_file()

    # second run: analysis comes from the caches, only planning runs again
    from vlogcut.analyze import quality, shots

    monkeypatch.setattr(quality, "measure_shot", lambda *a: pytest.fail("re-measured"))
    monkeypatch.setattr(shots, "detect_cuts", lambda *a, **k: pytest.fail("re-detected"))
    assert cli.main(["cut", str(day), "-t", "4", "-w", str(ws), "--no-render"]) == 0
    assert (ws / "edl" / "coverage_4s.json").is_file()
    assert not (ws / "preview_coverage_4s.mp4").exists()
