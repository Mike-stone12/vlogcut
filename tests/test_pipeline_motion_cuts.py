import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from vlogcut import pipeline, plan
from vlogcut.analyze import pauses, quality
from vlogcut.analyze.shots import ShotSpan
from vlogcut.schemas import CutPoint, MediaAsset, QualityFlags, Shot


class LegacyQualityCache(BaseModel):
    asset_id: str
    params: quality.QualityParams
    spans: list[tuple[str, float, float]]
    shots: list[Shot]


@pytest.mark.parametrize("workers", [1, 2])
def test_motion_candidates_merge_and_cache_toggle_preserves_default_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=20, fps=2, width=320, height=240)
    spans = [
        ShotSpan(shot_id=f"s{i}", asset_id="a", start_s=start, end_s=start + 4)
        for i, start in enumerate((10, 14))
    ]
    flags = QualityFlags(sharpness=200, exposure_clip=0, shake=0)
    legacy_shots = [Shot(**span.model_dump(), quality=flags) for span in spans]
    cache_path = tmp_path / "shots" / "a.quality.json"
    cache_path.parent.mkdir()
    cache_path.write_text(
        LegacyQualityCache(
            asset_id="a",
            params=quality.QualityParams(),
            spans=[(s.shot_id, s.start_s, s.end_s) for s in spans],
            shots=legacy_shots,
        ).model_dump_json(indent=1),
        encoding="utf-8",
    )
    legacy_bytes = cache_path.read_bytes()
    measured: list[str] = []

    def measure(
        asset: MediaAsset, span: ShotSpan, params: quality.QualityParams
    ) -> quality.ShotMetrics:
        measured.append(span.shot_id)
        return quality.ShotMetrics(
            fps=2,
            sharpness=[200] * 10,
            dark=[0] * 10,
            bright=[0] * 10,
            motion=[(v, 0) for v in (8, 0, 8, 1, 8, 0, 8, 0, 8)],
        )

    monkeypatch.setattr(pipeline, "find_footage", lambda *a: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **kw: ([asset], []))
    monkeypatch.setattr("vlogcut.analyze.detect_shots", lambda *a: spans)
    monkeypatch.setattr(quality, "measure_shot", measure)
    monkeypatch.setattr(quality, "assess", lambda *a: flags.model_copy(deep=True))
    midpoints = [18, 16.05, 14.5, 14.5, 14, 12.05, 10.5, 10.5, 10]
    monkeypatch.setattr(
        pauses, "analyze_pauses", lambda *a: [SimpleNamespace(mid_s=t) for t in midpoints]
    )
    expected = [
        s.model_copy(
            update={
                "kind": "broll",
                "cut_points": pauses.cut_points_for_span(midpoints, s.start_s, s.end_s),
            }
        )
        for s in legacy_shots
    ]
    baseline_edl = plan.rough_cut(expected, [asset], plan.RoughCutParams(target_s=3))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(baseline_edl.model_dump_json(indent=1), encoding="utf-8")
    result = pipeline.cut(tmp_path, tmp_path, target_s=3, render=False, workers=workers)
    assert measured == []  # Legacy cache hit: no marker required.
    assert result.shots == expected
    assert result.edl_path.read_bytes() == baseline_path.read_bytes()
    assert cache_path.read_bytes() == legacy_bytes
    shot_bytes = (tmp_path / "shots.json").read_bytes()
    prompt_bytes = json.dumps(plan.build_planner_messages(expected, target_s=3)).encode()

    for _ in range(2):
        result = pipeline.cut(
            tmp_path, tmp_path, target_s=3, render=False, workers=workers, motion_cuts=True
        )
        assert measured == ["s0", "s1"]  # Second opt-in run hits the cache.
        assert json.loads(cache_path.read_bytes())["motion_cuts"] is True
        for shot in result.shots:
            start = shot.start_s
            assert shot.cut_points == [
                CutPoint(time_s=start + 0.5, kind="pause"),
                CutPoint(time_s=start + 1, kind="motion"),
                CutPoint(time_s=start + 2.05, kind="pause"),
                CutPoint(time_s=start + 3, kind="motion"),
            ]
            assert Shot.model_validate_json(shot.model_dump_json()) == shot

    for _ in range(2):
        result = pipeline.cut(
            tmp_path, tmp_path, target_s=3, render=False, workers=workers, motion_cuts=False
        )
        assert measured == ["s0", "s1", "s0", "s1"]
        assert result.shots == expected
        assert cache_path.read_bytes() == legacy_bytes
        assert (tmp_path / "shots.json").read_bytes() == shot_bytes
        assert result.edl_path.read_bytes() == baseline_path.read_bytes()
        assert (
            json.dumps(plan.build_planner_messages(result.shots, target_s=3)).encode()
            == prompt_bytes
        )
