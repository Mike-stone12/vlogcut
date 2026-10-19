from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlogcut import cli
from vlogcut.pipeline import CutResult, find_footage
from vlogcut.schemas import EDL, Clip, DurationStats, StyleProfile
from vlogcut.validate import Violation


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
    assert args.style is None and args.music is None
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
    assert cli.main(["cut", str(tmp_path), "--style", str(tmp_path / "nope.json")]) == 1
    assert "style profile not found" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text('{"cuts_per_min": -1}', encoding="utf-8")
    assert cli.main(["cut", str(tmp_path), "--style", str(bad), "-w", str(tmp_path / "ws")]) == 1
    err = capsys.readouterr().err
    assert "invalid style profile" in err and err.count("\n") == 1


@pytest.mark.parametrize("clip_index", [0, None])
def test_cut_reports_violations_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    clip_index: int | None,
) -> None:
    violation = Violation(code="example", clip_index=clip_index, message="Needs attention")
    result = _cut_result(tmp_path, [violation])
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *args, **kwargs: result)
    assert cli.main(["cut", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "validation: 1 issue" in out
    prefix = "clip 0: " if clip_index is not None else ""
    assert f"  {prefix}example: Needs attention\n" in out
    assert "clip None" not in out


def test_cut_reports_validation_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    result = _cut_result(tmp_path, [])
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *args, **kwargs: result)
    assert cli.main(["cut", str(tmp_path), "--no-render"]) == 0
    assert "validation: OK\n" in capsys.readouterr().out


def _cut_result(tmp_path: Path, violations: list[Violation]) -> CutResult:
    return CutResult(
        assets=[], skipped=[], shots=[],
        edl=EDL(
            variant_id="test", target_duration_s=3,
            clips=[Clip(shot_id="s", in_s=0, out_s=3)],
        ),
        edl_path=tmp_path / "coverage_3s.json", preview_path=None, violations=violations,
    )  # fmt: skip


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
    assert (ws / "edl" / "coverage_9s.violations.json").is_file()

    # second run: analysis comes from the caches, only planning runs again
    from vlogcut.analyze import quality, shots

    monkeypatch.setattr(quality, "measure_shot", lambda *a: pytest.fail("re-measured"))
    monkeypatch.setattr(shots, "detect_cuts", lambda *a, **k: pytest.fail("re-detected"))
    assert cli.main(["cut", str(day), "-t", "4", "-w", str(ws), "--no-render"]) == 0
    assert (ws / "edl" / "coverage_4s.json").is_file()
    assert (ws / "edl" / "coverage_4s.violations.json").is_file()
    assert not (ws / "preview_coverage_4s.mp4").exists()

    # a style profile changes the pacing and gets its own output name; still no re-analysis
    fast = tmp_path / "fast.json"
    pacing = DurationStats(p10=0.5, median=1.0, p90=1.5)
    fast.write_text(
        StyleProfile(shot_duration_s=pacing, cuts_per_min=60).model_dump_json(), encoding="utf-8"
    )
    argv = ["cut", str(day), "-t", "4", "-w", str(ws), "--no-render", "--style", str(fast)]
    assert cli.main(argv) == 0
    assert "style     fast.json" in capsys.readouterr().out
    plain = json.loads((ws / "edl" / "coverage_4s.json").read_text(encoding="utf-8"))
    styled = json.loads((ws / "edl" / "coverage_4s_fast.json").read_text(encoding="utf-8"))
    assert len(styled["clips"]) > len(plain["clips"])
    assert all(c["out_s"] - c["in_s"] <= 1.5 + 1e-6 for c in styled["clips"])
    assert "style fast.json" in styled["notes"]
    assert (ws / "edl" / "coverage_4s_fast.violations.json").is_file()


# ---------------------------------------------------------------------------
# vlogcut style
# ---------------------------------------------------------------------------


def test_style_parser_defaults():
    args = cli.build_parser().parse_args(["style", "build", "a.mp4", "b.mp4"])
    assert args.refs == [Path("a.mp4"), Path("b.mp4")]
    assert args.out == Path("profile.json")
    diff = cli.build_parser().parse_args(["style", "diff", "a.json", "b.json"])
    assert (diff.style_command, diff.a, diff.b) == ("diff", Path("a.json"), Path("b.json"))
    for argv in (["style"], ["style", "build"], ["style", "show"], ["style", "diff", "a.json"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(argv)


def test_style_missing_reference_is_a_one_line_error(tmp_path, capsys):
    missing = tmp_path / "nope.mp4"
    assert cli.main(["style", "build", str(missing), "-w", str(tmp_path / "ws")]) == 1
    err = capsys.readouterr().err
    assert "reference not found" in err and err.count("\n") == 1


def _write_profile(path: Path, p10: float, median: float, p90: float, cpm: float) -> Path:
    stats = DurationStats(p10=p10, median=median, p90=p90)
    profile = StyleProfile(shot_duration_s=stats, cuts_per_min=cpm, notes="handmade")
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    return path


def test_style_show_and_diff(tmp_path, capsys):
    fast = _write_profile(tmp_path / "fast.json", 0.6, 1.8, 5.2, 28.8)
    slow = _write_profile(tmp_path / "slow.json", 2.1, 3.4, 9.3, 9.0)
    assert cli.main(["style", "show", str(fast)]) == 0
    out = capsys.readouterr().out
    assert "handmade" in out and "cuts/min" in out and "28.80" in out
    assert "transitions   not measured" in out
    assert cli.main(["style", "diff", str(fast), str(slow)]) == 0
    out = capsys.readouterr().out
    assert "octaves" in out and "fast.json" in out and "slow.json" in out
    for dim in ("p10", "median", "p90", "cuts/min"):
        assert dim in out


def test_style_show_measured_transitions(tmp_path, capsys):
    path = _write_profile(tmp_path / "mix.json", 1, 2, 4, 20)
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["transitions"] = {"hard_cut": 0.75, "dissolve": 0.25}
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert cli.main(["style", "show", str(path)]) == 0
    assert "transitions   hard cut 75.0%, dissolve 25.0%" in capsys.readouterr().out


def test_style_show_and_diff_errors_are_one_line(tmp_path, capsys):
    good = _write_profile(tmp_path / "good.json", 1.0, 2.0, 4.0, 20.0)
    assert cli.main(["style", "show", str(tmp_path / "nope.json")]) == 1
    assert "profile not found" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert cli.main(["style", "diff", str(good), str(bad)]) == 1
    err = capsys.readouterr().err
    assert "invalid style profile" in err and err.count("\n") == 1


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_style_end_to_end(make_edited_clip, tmp_path, capsys):
    ref = make_edited_clip("ref.mp4", shot_lengths=[1.0, 2.0, 1.0])
    out = tmp_path / "p.json"
    argv = ["style", "build", str(ref), "-o", str(out), "-w", str(tmp_path / "ws")]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "1 reference," in text and str(out) in text
    profile = StyleProfile.model_validate_json(out.read_text(encoding="utf-8"))
    assert profile.cuts_per_min == pytest.approx(30.0, abs=15)  # 2 cuts in 4 s, +-1 cut
    assert profile.source_refs == [str(ref)]
