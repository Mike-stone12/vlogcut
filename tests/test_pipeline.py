import json
import logging
from pathlib import Path

import pytest

from vlogcut import pipeline
from vlogcut.schemas import (
    EDL,
    Clip,
    DurationStats,
    MediaAsset,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.validate import validate_edl


def _shots() -> list[Shot]:
    return [
        Shot(
            shot_id="s", asset_id="a", start_s=0, end_s=10,
            quality=QualityFlags(sharpness=200, exposure_clip=0, shake=0),
        )
    ]  # fmt: skip


def _edl(shot_id: str = "s") -> EDL:
    return EDL(
        variant_id="test", target_duration_s=1,
        clips=[Clip(shot_id=shot_id, in_s=0, out_s=1)],
    )  # fmt: skip


def test_validate_and_write_valid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "edl" / "coverage_1s.json"
    with caplog.at_level(logging.INFO, logger="vlogcut.pipeline"):
        assert pipeline.validate_and_write(_edl(), _shots(), path, target_s=1) == []
    assert json.loads(path.with_suffix(".violations.json").read_text(encoding="utf-8")) == []
    assert [(r.levelname, r.message) for r in caplog.records] == [("INFO", "validation: OK")]


def test_validate_and_write_unknown_then_overwrite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "edl" / "coverage_1s_fast.json"
    violations = pipeline.validate_and_write(_edl("missing"), _shots(), path, target_s=1)
    report = path.with_suffix(".violations.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data == [v.model_dump(mode="json") for v in violations]
    assert data == [
        {"code": "unknown_shot", "clip_index": 0, "message": "Shot 'missing' is absent from shots"}
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "unknown_shot" in caplog.records[0].message
    assert pipeline.validate_and_write(_edl(), _shots(), path, target_s=1) == []
    assert json.loads(report.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("styled", [False, True])
@pytest.mark.parametrize("render", [False, True])
def test_cut_validates_with_requested_target_and_style_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    styled: bool,
    render: bool,
) -> None:
    shots, edl = _shots(), _edl()
    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=10, fps=30, width=320, height=240)
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    monkeypatch.setattr("vlogcut.plan.rough_cut", lambda *args: edl)
    monkeypatch.setattr(pipeline, "repair_edl", lambda *args: pytest.fail("unexpected repair"))
    profile = StyleProfile(shot_duration_s=DurationStats(p10=2, median=3, p90=4), cuts_per_min=20)
    style = tmp_path / "fast.json" if styled else None
    if style is not None:
        style.write_text(profile.model_dump_json(), encoding="utf-8")
    stem = "coverage_3s_fast" if styled else "coverage_3s"
    report = tmp_path / "edl" / f"{stem}.violations.json"
    rendered = []

    def render_preview(*args: object, **kwargs: object) -> None:
        assert json.loads(report.read_text(encoding="utf-8"))
        rendered.append(args[0])

    monkeypatch.setattr("vlogcut.render.render_preview", render_preview)
    result = pipeline.cut(tmp_path, tmp_path, target_s=3, style=style, render=render)
    expected = ["clip_too_short", "total_duration"] if styled else ["total_duration"]
    assert [v.code for v in result.violations] == expected
    assert result.violations[-1].clip_index is None
    assert len(caplog.records) == len(expected)
    assert all(r.levelname == "WARNING" for r in caplog.records)
    assert result.profile == (profile if styled else None)
    assert result.edl_path == report.with_name(f"{stem}.json")
    assert json.loads(report.read_text(encoding="utf-8")) == [
        v.model_dump(mode="json") for v in result.violations
    ]
    assert rendered == ([edl] if render else [])
    assert not result.repaired


@pytest.mark.parametrize("case", ["clean", "mixed", "all_unknown"])
def test_cut_repair_updates_files_and_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    shots = _shots()
    rejected = shots[0].model_copy(deep=True)
    rejected.shot_id = "rejected"
    rejected.quality.rejected = True
    shots.append(rejected)
    retained = Clip(shot_id="s", in_s=2, out_s=4, speed=2, audio="duck")
    clips = [retained]
    if case == "mixed":
        clips += [Clip(shot_id="rejected", in_s=0, out_s=1)]
    if case != "clean":
        clips += [Clip(shot_id="missing", in_s=0, out_s=1)]
    if case == "all_unknown":
        clips = clips[-1:]
    edl = EDL(variant_id="test", target_duration_s=len(clips), clips=clips)
    original_json = edl.model_dump_json(indent=1)
    target_s = edl.total_duration_s
    original_violations = validate_edl(edl, shots, target_s=target_s)
    # Baseline serialization uses the same platform newline handling as cut.
    baseline = tmp_path / "original.json"
    baseline.write_text(original_json, encoding="utf-8")
    pipeline.validate_and_write(edl, shots, baseline, target_s=target_s)
    original_report = baseline.with_suffix(".violations.json").read_bytes()
    caplog.clear()

    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=10, fps=30, width=320, height=240)
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    monkeypatch.setattr("vlogcut.plan.rough_cut", lambda *args: edl)
    if case == "clean":
        monkeypatch.setattr(
            pipeline, "repair_edl", lambda *args: pytest.fail("clean EDL must bypass repair")
        )
    edl_path = tmp_path / "edl" / f"coverage_{target_s:.0f}s.json"
    report = edl_path.with_suffix(".violations.json")
    rendered: list[EDL] = []

    def render_preview(final_edl: EDL, *args: object, **kwargs: object) -> None:
        assert EDL.model_validate_json(edl_path.read_text(encoding="utf-8")) == final_edl
        assert json.loads(report.read_text(encoding="utf-8")) == [
            v.model_dump(mode="json") for v in validate_edl(final_edl, shots, target_s=target_s)
        ]
        rendered.append(final_edl)

    monkeypatch.setattr("vlogcut.render.render_preview", render_preview)
    result = pipeline.cut(tmp_path, tmp_path, target_s=target_s)
    assert result.repaired == (case == "mixed")
    assert rendered == [result.edl]
    assert json.loads(report.read_text(encoding="utf-8")) == [
        v.model_dump(mode="json") for v in result.violations
    ]
    assert edl.model_dump_json(indent=1) == original_json
    assert not edl_path.with_suffix(".json.part").exists()
    if case == "mixed":
        assert {v.code for v in original_violations} == {"unknown_shot", "rejected_shot"}
        assert result.edl.clips == [retained]  # in/out, speed and audio all preserved
        assert [v.code for v in result.violations] == ["total_duration"]
    else:
        assert result.edl == edl
        assert result.violations == original_violations
        assert edl_path.read_bytes() == baseline.read_bytes()
        assert report.read_bytes() == original_report
    if case == "all_unknown":
        assert any(
            r.levelname == "WARNING" and "keeping original EDL and violations" in r.message
            for r in caplog.records
        )


@pytest.mark.parametrize(
    ("dissolve_share", "duration", "code", "expected"),
    [
        (0.0, 0.5, "transition_not_allowed", Transition()),
        (1.0, 4.0, "dissolve_too_long", Transition(kind="dissolve", duration_s=1)),
    ],
)
def test_repair_if_needed_fixes_transitions(
    dissolve_share: float, duration: float, code: str, expected: Transition
) -> None:
    shots = _shots()
    shots.append(shots[0].model_copy(update={"shot_id": "s2"}))
    edl = _edl()
    edl.clips.append(
        Clip(
            shot_id="s2",
            in_s=1,
            out_s=2,
            transition_in=Transition(kind="dissolve", duration_s=duration),
        )
    )
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=1, p90=1),
        cuts_per_min=60,
        transitions=TransitionMix(hard_cut=1 - dissolve_share, dissolve=dissolve_share),
    )
    violations = validate_edl(edl, shots, profile, target_s=2)
    assert code in {v.code for v in violations}
    final_edl, repaired = pipeline.repair_if_needed(edl, shots, violations, profile)
    assert repaired
    assert final_edl.clips[1].transition_in == expected
    assert validate_edl(final_edl, shots, profile, target_s=2) == []
    assert [(c.in_s, c.out_s) for c in final_edl.clips] == [(0, 1), (1, 2)]
