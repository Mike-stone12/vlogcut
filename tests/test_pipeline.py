import json
import logging
from pathlib import Path

import pytest

from vlogcut import pipeline
from vlogcut.schemas import EDL, Clip, DurationStats, MediaAsset, QualityFlags, Shot, StyleProfile


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
