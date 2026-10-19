from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlogcut import cli
from vlogcut.learn.ranker import RankerModel
from vlogcut.llm.router import ModelRouter
from vlogcut.pipeline import CutResult, find_footage
from vlogcut.schemas import EDL, Clip, DurationStats, QualityFlags, Shot, StyleProfile
from vlogcut.validate import Violation


@pytest.mark.parametrize("payload", [None, b"{", b"{}", b"null", b"\xff"])
@pytest.mark.parametrize("planner", ["rules", "llm"])
def test_bad_ranker_fails_before_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes | None,
    planner: str,
) -> None:
    path = tmp_path / "ranker.json"
    if payload is not None:
        path.write_bytes(payload)
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **kw: pytest.fail("started cut"))
    monkeypatch.setattr(ModelRouter, "from_yaml", lambda *a: pytest.fail("loaded config"))
    assert cli.main(["cut", "day", "--planner", planner, "--ranker", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert captured.err.startswith("vlogcut: error: invalid ranker model")
    assert str(path) in captured.err


@pytest.mark.parametrize("enabled", [False, True])
def test_ranker_loaded_and_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ranker_model: RankerModel, enabled: bool
) -> None:
    path = tmp_path / "ranker.json"
    path.write_text(ranker_model.model_dump_json(), encoding="utf-8")
    captured: dict[str, object] = {}

    def cut(*args: object, **kwargs: object) -> CutResult:
        captured.update(kwargs)
        return _cut_result(tmp_path, [])

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    argv = ["cut", str(tmp_path), "--no-render"]
    if enabled:
        argv += ["--ranker", str(path)]
    assert cli.main(argv) == 0
    assert captured["ranker"] == (ranker_model if enabled else None)
    if enabled:
        assert isinstance(captured["ranker"], RankerModel)


def test_ranker_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["cut", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--ranker MODEL_JSON" in help_text
    assert "scores only shape selection and length" in help_text
    assert "hard filters and chronology are unchanged" in help_text


@pytest.mark.parametrize(
    "payload", [None, b"{", b"{}", b'"text"', b"null", b"[1]", b"[null]", b'["valid", {}]', b"\xff"]
)
@pytest.mark.parametrize("planner", ["rules", "llm"])
def test_bad_constraints_fail_before_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes | None,
    planner: str,
) -> None:
    path = tmp_path / "constraints.json"
    if payload is not None:
        path.write_bytes(payload)
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **kw: pytest.fail("started cut"))
    monkeypatch.setattr(ModelRouter, "from_yaml", lambda *a: pytest.fail("loaded config"))
    assert cli.main(["cut", "day", "--planner", planner, "--constraints", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert captured.err.startswith("vlogcut: error:")
    assert str(path) in captured.err


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
    assert args.jobs == 1
    assert not args.caption
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["cut", "day1", "--strategy", "random"])


@pytest.mark.parametrize("jobs", [0, -1])
def test_invalid_jobs_fail_before_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, jobs: int
) -> None:
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **k: pytest.fail("started cut"))
    assert cli.main(["cut", str(tmp_path), "--jobs", str(jobs)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "vlogcut: error: --jobs must be at least 1\n"


@pytest.mark.parametrize("jobs", [1, 2])
def test_jobs_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int) -> None:
    def cut(*args: object, **kwargs: object) -> CutResult:
        assert kwargs["workers"] == jobs
        return _cut_result(tmp_path, [])

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    argv = ["cut", str(tmp_path), "--no-render"]
    if jobs != 1:
        argv += ["--jobs", str(jobs)]
    assert cli.main(argv) == 0


@pytest.mark.parametrize("cold_open", [False, True])
def test_cold_open_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cold_open: bool
) -> None:
    def cut(*args: object, **kwargs: object) -> CutResult:
        assert kwargs["cold_open"] is cold_open
        return _cut_result(tmp_path, [])

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    argv = ["cut", str(tmp_path), "--no-render"]
    if cold_open:
        argv.append("--cold-open")
    assert cli.main(argv) == 0


def test_cut_help_lists_music(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["cut", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--music PATH" in captured.out and "beat snapping" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("music", [False, True])
@pytest.mark.parametrize("combined", [False, True])
def test_music_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, music: bool, combined: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    song = Path("song.wav")
    song.write_bytes(b"unused audio placeholder")
    captured: dict[str, object] = {}

    def cut(*args: object, **kwargs: object) -> CutResult:
        captured.update(kwargs)
        return _cut_result(tmp_path, [])

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    argv = ["cut", str(tmp_path), "--no-render"]
    if music:
        argv += ["--music", str(song)]
    if combined:
        profile = _write_profile(tmp_path / "style.json", 1, 2, 4, 20)
        config = tmp_path / "models.yaml"
        config.write_text(
            "providers: {fake: {base_url: 'http://fake/v1'}}\n"
            "roles: {planner: {provider: fake, model: test}, "
            "captioner: {provider: fake, model: test}}\n",
            encoding="utf-8",
        )
        argv += [
            "--style", str(profile), "--planner", "llm", "--cold-open", "--caption",
            "--config", str(config),
        ]  # fmt: skip
    else:
        monkeypatch.setattr(ModelRouter, "from_yaml", lambda *a: pytest.fail("read config"))
    assert cli.main(argv) == 0
    assert captured["music"] == (song if music else None)
    assert captured["cold_open"] is combined and captured["caption"] is combined
    assert captured["planner"] == ("llm" if combined else "rules")
    if combined:
        assert captured["style"] == profile
        assert isinstance(captured["router"], ModelRouter)


def test_missing_music_fails_before_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    missing = tmp_path / "missing.wav"
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **k: pytest.fail("started cut"))
    assert cli.main(["cut", str(tmp_path), "--music", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"vlogcut: error: music file not found: {missing}\n"


@pytest.mark.parametrize("detail", ["missing analysis dependency", "missing analysis\ndependency"])
def test_music_import_error_is_one_line_with_analysis_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, detail: str
) -> None:
    song = tmp_path / "song.wav"
    song.write_bytes(b"unused audio placeholder")

    def cut(*args: object, **kwargs: object) -> CutResult:
        assert kwargs["music"] == song
        raise ImportError(detail)

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    assert cli.main(["cut", str(tmp_path), "--music", str(song)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "vlogcut: error: missing analysis dependency; "
        'install the analysis extra: pip install -e ".[analyze]"\n'
    )


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


@pytest.mark.parametrize("repaired", [False, True])
def test_cut_reports_validation_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, repaired: bool
) -> None:
    result = _cut_result(tmp_path, [])
    result.repaired = repaired
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *args, **kwargs: result)
    assert cli.main(["cut", str(tmp_path), "--no-render"]) == 0
    out = capsys.readouterr().out
    assert "validation: OK\n" in out
    assert out.count(f"manifest  {result.manifest}\n") == 1
    assert out.count("repaired EDL\n") == int(repaired)
    if repaired:
        assert out.index("repaired EDL\n") < out.index("validation: OK\n")


def _cut_result(tmp_path: Path, violations: list[Violation]) -> CutResult:
    return CutResult(
        assets=[], skipped=[], shots=[],
        edl=EDL(
            variant_id="test", target_duration_s=3,
            clips=[Clip(shot_id="s", in_s=0, out_s=3)],
        ),
        edl_path=tmp_path / "coverage_3s.json", preview_path=None, violations=violations,
        manifest=tmp_path / "runs" / "0123456789abcdef.json",
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
    # Exact output captured before kind_alternation became a distance dimension.
    assert out == (
        "fast.json: handmade\n"
        "p10 shot s         0.60\n"
        "median shot s      1.80\n"
        "p90 shot s         5.20\n"
        "cuts/min          28.80\n"
        "transitions   not measured\n"
    )
    assert cli.main(["style", "diff", str(fast), str(slow)]) == 0
    out = capsys.readouterr().out
    # Only the existing unmeasured summary gains the new dimension's name.
    assert out.replace(", kind_alternation", "") == (
        "              fast.json  slow.json     delta\n"
        "p10 shot s         0.60       2.10      1.81\n"
        "median shot s      1.80       3.40      0.92\n"
        "p90 shot s         5.20       9.30      0.84\n"
        "cuts/min          28.80       9.00      1.68\n"
        "distance      1.31 octaves; mean over 4 dims\n"
        "not measured on both sides: intro_pace, beat_aligned_ratio, speech_ratio, "
        "dissolve, montage_share, brightness, saturation, warmth\n"
    )


@pytest.mark.parametrize(("a_value", "b_value"), [(0.2, 0.7), (0.0, 0.0)])
def test_style_show_and_diff_measured_kind_alternation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], a_value: float, b_value: float
) -> None:
    paths = [_write_profile(tmp_path / name, 1, 2, 4, 20) for name in ("a.json", "b.json")]
    for path, value in zip(paths, [a_value, b_value], strict=True):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["source_refs"] = ["edl"]
        profile["structure"] = {"montage_share": 0, "kind_alternation": value}
        path.write_text(json.dumps(profile), encoding="utf-8")
        assert cli.main(["style", "show", str(path)]) == 0
        rows = [line.split() for line in capsys.readouterr().out.splitlines()]
        assert ["kind_alternation", f"{value:.2f}"] in rows
    assert cli.main(["style", "diff", *map(str, paths)]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [
        "kind_alternation",
        f"{a_value:.2f}",
        f"{b_value:.2f}",
        f"{abs(a_value - b_value):.2f}",
    ] in rows


@pytest.mark.parametrize(("a_value", "b_value"), [(0.2, 0.7), (0.0, 0.0)])
def test_style_show_and_diff_measured_speech_ratio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], a_value: float, b_value: float
) -> None:
    paths = [_write_profile(tmp_path / name, 1, 2, 4, 20) for name in ("a.json", "b.json")]
    for path, value in zip(paths, [a_value, b_value], strict=True):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["speech_ratio"] = value
        path.write_text(json.dumps(profile), encoding="utf-8")
        assert cli.main(["style", "show", str(path)]) == 0
        rows = [line.split() for line in capsys.readouterr().out.splitlines()]
        assert ["speech", "ratio", f"{value:.2f}"] in rows
    assert cli.main(["style", "diff", *map(str, paths)]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [
        "speech", "ratio", f"{a_value:.2f}", f"{b_value:.2f}", f"{abs(a_value - b_value):.2f}",
    ] in rows  # fmt: skip


def test_style_diff_measured_look(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    paths = [_write_profile(tmp_path / name, 1, 2, 4, 20) for name in ("a.json", "b.json")]
    for path, warmth in zip(paths, [-0.2, 0.2], strict=True):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["look"] = dict(brightness=0.3, saturation=0.4, warmth=warmth, frames=3)
        path.write_text(json.dumps(profile), encoding="utf-8")
    assert cli.main(["style", "diff", *map(str, paths)]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["brightness", "0.30", "0.30", "0.00"] in rows
    assert ["saturation", "0.40", "0.40", "0.00"] in rows
    assert ["warmth", "-0.20", "0.20", "0.20"] in rows


@pytest.mark.parametrize("value", [None, 120.0, 123.456])
def test_style_show_music_bpm_and_diff_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: float | None
) -> None:
    path = _write_profile(tmp_path / "tempo.json", 1, 2, 4, 20)
    other = _write_profile(tmp_path / "other.json", 1, 2, 4, 20)
    assert cli.main(["style", "diff", str(path), str(other)]) == 0
    baseline_diff = capsys.readouterr().out
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["music_bpm"] = value
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert cli.main(["style", "show", str(path)]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    if value is None:
        assert not any(row[:2] == ["music", "BPM"] for row in rows)
    else:
        assert ["music", "BPM", f"{value:.1f}"] in rows
    assert cli.main(["style", "diff", str(path), str(other)]) == 0
    assert capsys.readouterr().out == baseline_diff


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


@pytest.mark.parametrize(
    "content",
    [
        None,
        "[",
        "",
        "[]",
        "providers: {}",
        "providers: {}\nroles: {}",
        "providers: {}\nroles: {planner: {provider: missing, model: test}}",
    ],
)
@pytest.mark.parametrize(
    "options", [["--planner", "llm"], ["--caption"], ["--caption", "--planner", "llm"]]
)
def test_model_config_errors_fail_before_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    content: str | None,
    options: list[str],
) -> None:
    config = tmp_path / "models.yaml"
    if content is not None:
        config.write_text(content, encoding="utf-8")
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **k: pytest.fail("started cut"))
    assert cli.main(["cut", str(tmp_path), *options, "--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model config" in captured.err and captured.err.count("\n") == 1


@pytest.mark.parametrize("planner", ["rules", "llm"])
def test_caption_loads_one_router_and_reports_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, planner: str
) -> None:
    config = tmp_path / "models.yaml"
    roles = {"captioner": {"provider": "fake", "model": "caption-model"}}
    if planner == "llm":
        roles["planner"] = {"provider": "fake", "model": "plan-model"}
    config.write_text(
        json.dumps(
            {
                "providers": {"fake": {"base_url": "http://fake/v1"}},
                "roles": roles,
            }
        ),
        encoding="utf-8",
    )
    routers: list[ModelRouter] = []
    from_yaml = ModelRouter.from_yaml

    def load(path: Path) -> ModelRouter:
        assert path == config
        router = from_yaml(path, client_factory=lambda _: pytest.fail("called provider"))
        routers.append(router)
        return router

    def cut(*args: object, **kwargs: object) -> CutResult:
        assert kwargs["caption"] is True and kwargs["planner"] == planner
        assert len(routers) == 1 and kwargs["router"] is routers[0]
        assert routers[0].roles["captioner"].model == "caption-model"
        if planner == "llm":
            assert routers[0].roles["planner"].model == "plan-model"
        result = _cut_result(tmp_path, [])
        result.shots = [
            Shot(
                shot_id=f"s{i}", asset_id="a", start_s=i * 3, end_s=(i + 1) * 3,
                quality=QualityFlags(sharpness=200, exposure_clip=0, shake=0),
                caption="Coffee" if i == 0 else None,
            )
            for i in range(2)
        ]  # fmt: skip
        return result

    monkeypatch.setattr(ModelRouter, "from_yaml", load)
    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    assert (
        cli.main(
            [
                "cut",
                str(tmp_path),
                "--caption",
                "--planner",
                planner,
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines().count("captioned 1 shots") == 1


@pytest.mark.parametrize("role", ["planner", "captioner"])
def test_caption_and_llm_require_both_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, role: str
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "providers: {fake: {base_url: 'http://fake/v1'}}\n"
        f"roles: {{{role}: {{provider: fake, model: test}}}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("vlogcut.pipeline.cut", lambda *a, **k: pytest.fail("started cut"))
    assert (
        cli.main(
            [
                "cut",
                str(tmp_path),
                "--caption",
                "--planner",
                "llm",
                "--config",
                str(config),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "missing" in err and "role" in err and err.count("\n") == 1


@pytest.mark.parametrize("planner", ["rules", "llm"])
def test_cut_planner_options_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, planner: str
) -> None:
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["cut", "day"])
    assert (args.planner, args.brief, args.config) == ("rules", None, Path("config.yaml"))
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers: {fake: {base_url: 'http://fake/v1'}}\n"
        "roles: {planner: {provider: fake, model: test}}\n",
        encoding="utf-8",
    )
    result = _cut_result(tmp_path, [])
    result.planner = planner
    result.repair_rounds = 1 if planner == "llm" else 0

    def cut(*args: object, **kwargs: object) -> CutResult:
        assert kwargs["planner"] == planner
        if planner == "llm":
            assert isinstance(kwargs["router"], ModelRouter)
            assert kwargs["router"].roles["planner"].model == "test"
            assert kwargs["brief"] == "A quiet morning"
        else:
            assert kwargs["router"] is None and kwargs["brief"] is None
        return result

    monkeypatch.setattr("vlogcut.pipeline.cut", cut)
    argv = ["cut", str(tmp_path), "--no-render"]
    if planner == "llm":
        argv += ["--planner", "llm", "--brief", "A quiet morning"]
    else:
        monkeypatch.setattr(ModelRouter, "from_yaml", lambda *a: pytest.fail("read config"))
    assert cli.main(argv) == 0
    assert (
        f"planner   {planner}, repair rounds: {result.repair_rounds}, fallback to rules: no\n"
        in capsys.readouterr().out
    )
    if planner == "llm":
        result.planner, result.fallback, result.repair_rounds = "rules", True, 0
        assert cli.main(argv) == 0
        assert (
            "planner   rules, repair rounds: 0, fallback to rules: yes\n" in capsys.readouterr().out
        )
