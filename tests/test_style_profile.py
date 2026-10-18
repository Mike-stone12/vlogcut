from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vlogcut.analyze import shots as shots_mod
from vlogcut.schemas import MediaAsset
from vlogcut.style import ReferenceParams, StyleError, build_profile, load_profile
from vlogcut.style import profile as profile_mod
from vlogcut.style.profile import (
    ReferenceCuts,
    aggregate,
    analyze_reference,
    duration_stats,
    shot_durations,
)

# ---------------------------------------------------------------------------
# pure aggregation
# ---------------------------------------------------------------------------


def test_shot_durations_merges_slivers_and_covers_the_clip():
    # cuts at 2.0 and 2.1: the 0.1 s sliver joins its shorter neighbour (the 2 s one)
    durations = shot_durations([2.0, 2.1, 5.0], 8.0, min_shot_s=0.25)
    assert durations == pytest.approx([2.1, 2.9, 3.0])
    assert sum(durations) == pytest.approx(8.0)
    assert shot_durations([], 4.0, min_shot_s=0.25) == [4.0]


def test_duration_stats_quantiles():
    stats = duration_stats([float(i) for i in range(1, 11)])
    assert (stats.p10, stats.median, stats.p90) == pytest.approx((1.9, 5.5, 9.1))


def test_duration_stats_needs_at_least_two_shots():
    with pytest.raises(StyleError, match="cut"):
        duration_stats([5.0])


def _ref(
    name: str, duration: float, cuts: list[float], dissolves: list[float] | None = None
) -> ReferenceCuts:
    return ReferenceCuts(
        asset_id=name, source=f"{name}.mp4", duration_s=duration, params=ReferenceParams(),
        cuts=cuts, dissolve_cuts=dissolves or [],
    )  # fmt: skip


def test_aggregate_pools_shots_and_rates_by_total_time():
    a = _ref("a", 60.0, [i * 6.0 for i in range(1, 10)])  # 9 cuts: ten 6 s shots
    b = _ref("b", 120.0, [30.0, 60.0, 90.0])  # 3 cuts: four 30 s shots
    profile = aggregate([a, b], ReferenceParams())
    assert profile.cuts_per_min == pytest.approx(4.0)  # 12 cuts over 3 min, not mean(9, 1.5)
    assert profile.shot_duration_s.median == pytest.approx(6.0)
    assert profile.shot_duration_s.p90 == pytest.approx(30.0)
    assert profile.source_refs == ["a.mp4", "b.mp4"]
    assert profile.beat_aligned_ratio is None and profile.speech_ratio is None
    assert "2 references" in profile.notes and "4.0 cuts/min" in profile.notes


def test_aggregate_without_references_raises():
    with pytest.raises(StyleError, match="reference"):
        aggregate([], ReferenceParams())


def test_aggregate_pools_transition_counts():
    a = _ref("a", 60, [10, 20, 30], [10, 20])
    b = _ref("b", 120, [60])
    p = aggregate([a, b], ReferenceParams())
    assert p.transitions.hard_cut == pytest.approx(0.5)
    assert p.transitions.dissolve == pytest.approx(2 / 4)
    assert p.cuts_per_min == pytest.approx(4 / 3)
    assert "50.0% dissolves" in p.notes


@pytest.mark.parametrize("disabled_reference", [False, True])
def test_aggregate_transitions_unmeasured_when_detection_disabled(disabled_reference):
    a = _ref("a", 60, [10], [10])
    b = _ref("b", 60, [20])
    params = ReferenceParams(dissolves=disabled_reference)
    if disabled_reference:
        b.params = ReferenceParams(dissolves=False)
    p = aggregate([a, b], params)
    assert p.transitions is None
    assert "dissolves" not in p.notes


def test_aggregate_transition_counts_use_surviving_rounded_boundaries():
    # The 2 s boundary disappears with its sliver; rounding retains the other dissolve.
    ref = _ref("a", 8, [2, 2.1, 5.0000001], [2, 5.0000001])
    p = aggregate([ref], ReferenceParams())
    assert p.cuts_per_min == pytest.approx(2 / (8 / 60))
    assert p.transitions.dissolve == pytest.approx(1 / 2)


def test_reference_dissolves_must_be_a_subset_of_cuts():
    with pytest.raises(ValidationError, match="subset"):
        _ref("a", 60, [10], [20])


@pytest.mark.parametrize("cuts", [[], [0.1]])
@pytest.mark.parametrize("detection", [False, True])
def test_aggregate_zero_surviving_cuts_is_unmeasured(cuts, detection):
    ref = _ref("a", 60, cuts, cuts)
    for refs in ([ref], [ref, _ref("b", 120, [])]):
        p = aggregate(refs, ReferenceParams(dissolves=detection))
        assert p.cuts_per_min == 0
        assert p.transitions is None
        assert "dissolves" not in p.notes
        if len(refs) == 1:
            assert p.shot_duration_s.model_dump() == {"p10": 60, "median": 60, "p90": 60}


@pytest.mark.parametrize("detection", [False, True])
def test_old_reference_cache_is_recomputed(tmp_path, monkeypatch, detection):
    params = ReferenceParams(dissolves=detection)
    old = _ref("a", 60, [10]).model_dump(exclude={"dissolve_cuts"})
    old["params"] = params.model_dump()
    cache = tmp_path / "style" / "a.cuts.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(old), encoding="utf-8")
    asset = MediaAsset(
        asset_id="a",
        path="a.mp4",
        proxy_path="proxy.mp4",
        duration_s=60,
        fps=30,
        width=64,
        height=64,
    )
    monkeypatch.setattr(profile_mod, "probe_asset", lambda path: asset)
    monkeypatch.setattr(profile_mod, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [10, 20])
    monkeypatch.setattr(
        profile_mod.transitions,
        "load_frames",
        lambda path: None if detection else pytest.fail("dissolve detection disabled"),
    )
    monkeypatch.setattr(profile_mod.transitions, "find_dissolves", lambda *a: [(19, 21)])
    record = analyze_reference("a.mp4", tmp_path, params)
    assert record.cuts == [10, 20]
    assert record.dissolve_cuts == ([20] if detection else [])
    assert ReferenceCuts.model_validate_json(cache.read_text(encoding="utf-8")) == record
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: pytest.fail("re-detected"))
    assert analyze_reference("a.mp4", tmp_path, params) == record


def test_load_profile_round_trips_and_rejects_bad_files(tmp_path):
    profile = aggregate([_ref("a", 60.0, [10.0, 20.0, 30.0])], ReferenceParams())
    path = tmp_path / "p.json"
    path.write_text(profile.model_dump_json(indent=1), encoding="utf-8")
    assert load_profile(path) == profile
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StyleError, match="invalid style profile"):
        load_profile(path)
    path.write_text('{"cuts_per_min": 0}', encoding="utf-8")
    with pytest.raises(StyleError, match="shot_duration_s") as info:
        load_profile(path)
    assert "\n" not in str(info.value)  # one line, so the CLI can print it as is


# ---------------------------------------------------------------------------
# real detector on generated "finished" clips
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_fast_and_slow_references_give_different_profiles(make_edited_clip, tmp_path):
    fast = make_edited_clip("fast.mp4", shot_lengths=[0.5, 1.0, 0.5, 0.75, 1.0, 0.5, 0.75, 1.0])
    slow = make_edited_clip("slow.mp4", shot_lengths=[2.0, 3.0, 12.0, 3.0])
    ws = tmp_path / "ws"
    quick = build_profile([fast], ws)
    calm = build_profile([slow], ws)
    assert quick.cuts_per_min == pytest.approx(70.0, abs=10.0)  # 7 cuts in 6 s, +-1 cut
    assert calm.cuts_per_min == pytest.approx(9.0, abs=3.0)  # 3 cuts in 20 s, +-1 cut
    assert quick.shot_duration_s.median < 1.1 < 2.5 < calm.shot_duration_s.median
    assert quick.source_refs == [str(fast)]
    # no split_long: the 12 s shot survives whole, so p90 of [2, 3, 3, 12] is 9.3
    # (the footage detector would split it into 3 x 4 s and p90 would be 4)
    assert calm.shot_duration_s.p90 == pytest.approx(9.3, abs=0.2)


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_reference_cuts_are_cached_by_params(make_edited_clip, tmp_path, monkeypatch):
    ref = make_edited_clip("ref.mp4", shot_lengths=[1.0, 1.0, 1.0])
    ws = tmp_path / "ws"
    first = analyze_reference(ref, ws, ReferenceParams())
    assert len(first.cuts) == 2
    assert (ws / "style" / f"{first.asset_id}.cuts.json").is_file()

    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: pytest.fail("re-detected"))
    assert analyze_reference(ref, ws, ReferenceParams()) == first

    monkeypatch.undo()
    other = analyze_reference(ref, ws, ReferenceParams(threshold=2.5))
    assert other.params.threshold == 2.5 and len(other.cuts) == 2
