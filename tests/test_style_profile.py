from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from vlogcut.analyze import beats as beats_mod
from vlogcut.analyze import look as look_mod
from vlogcut.analyze import pauses as pauses_mod
from vlogcut.analyze import shots as shots_mod
from vlogcut.ingest import ffmpeg
from vlogcut.schemas import LookStats, MediaAsset
from vlogcut.style import ReferenceParams, StyleError, build_profile, load_profile
from vlogcut.style import profile as profile_mod
from vlogcut.style.profile import (
    ReferenceCuts,
    aggregate,
    analyze_reference,
    combine_cuts,
    duration_stats,
    intro_pace_ratio,
    kind_alternation,
    montage_share,
    pooled_beat_ratio,
    pooled_bpm,
    shot_durations,
    transition_mix,
)

# ---------------------------------------------------------------------------
# pure aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("beat_lists", "expected"),
    [
        ([[0, 0.5, 1, 1.5]], 120.0),
        ([[0, 0.5], [100, 101]], 80.0),  # median(.5, 1), with no inter-reference gap
        ([[0, 0.5, 1, 1.5], [0, 1]], 120.0),  # pool intervals, not reference BPMs
        ([[], [50], [0, 0.5]], 120.0),
        ([[1, 0, 0.5, 0.5]], 120.0),  # sort without mutating; skip zero intervals
        ([[1, 1]], None),
        ([[], [50]], None),
        ([[], []], None),
        ([], None),
    ],
)
def test_pooled_bpm(beat_lists: list[list[float]], expected: float | None) -> None:
    original = [beats[:] for beats in beat_lists]
    assert pooled_bpm(beat_lists) == expected
    assert beat_lists == original


def test_aggregate_music_bpm_pools_reference_intervals() -> None:
    a = _ref("a", 60, [], beats=[0, 0.5, 1, 1.5])
    b = _ref("b", 120, [20], beats=[100, 101])
    beatless = _ref("c", 60, [10])
    single = _ref("d", 60, [10], beats=[50])
    assert aggregate([a, b, beatless, single], ReferenceParams()).music_bpm == 120.0
    assert aggregate([beatless, single], ReferenceParams()).music_bpm is None


def test_aggregate_references_leave_kind_alternation_unmeasured() -> None:
    profile = aggregate([_ref("a", 60, [10, 20, 30, 40])], ReferenceParams())
    assert profile.structure is not None
    assert profile.structure.kind_alternation is None


def test_aggregate_look_weights_sampled_frames() -> None:
    a, b, unmeasured = [_ref(name, 60, [20]) for name in ("a", "b", "c")]
    a.look = LookStats(brightness=1, saturation=0, warmth=-1, frames=1)
    b.look = LookStats(brightness=0, saturation=1, warmth=1, frames=3)
    result = aggregate([a, b, unmeasured], ReferenceParams()).look
    assert result == LookStats(brightness=0.25, saturation=0.75, warmth=0.5, frames=4)
    assert aggregate([unmeasured], ReferenceParams()).look is None
    assert aggregate([a, b], ReferenceParams(look=False)).look is None
    b.params.look = False
    assert aggregate([a, b], ReferenceParams()).look is None


@pytest.fixture
def stub_reference(monkeypatch: pytest.MonkeyPatch) -> MediaAsset:
    asset = MediaAsset(
        asset_id="a", path="a.mp4", proxy_path="proxy.mp4", duration_s=120,
        fps=30, width=64, height=64, has_audio=False,
    )  # fmt: skip
    monkeypatch.setattr(profile_mod, "probe_asset", lambda path: asset)
    monkeypatch.setattr(profile_mod, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [10, 20])
    return asset


@pytest.mark.parametrize("empty", [False, True])
def test_build_look_cache_hit_without_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_reference: MediaAsset, empty: bool
) -> None:
    measured = None if empty else LookStats(brightness=0.5, saturation=0, warmth=0, frames=3)
    frame_batch = object()

    def load(path: str, params: look_mod.LookParams, *, max_frames: int) -> object:
        assert path == stub_reference.proxy_path
        assert params.size == 64
        assert params.sample_fps * stub_reference.duration_s == pytest.approx(32)
        assert max_frames == 32
        return frame_batch

    def measure(frames: object) -> LookStats | None:
        assert frames is frame_batch
        return measured

    monkeypatch.setattr(look_mod, "load_rgb_frames", load)
    monkeypatch.setattr(look_mod, "look_stats", measure)
    params = ReferenceParams(dissolves=False, beats=False)
    first = build_profile(["a.mp4"], tmp_path, params)
    assert first.speech_ratio is None
    assert '"speech_ratio":null' in first.model_dump_json()
    assert (first.look is None) == empty
    if not empty:
        assert first.look.brightness == 0.5
        assert first.look.frames == 3
    cache = tmp_path / "style/a.cuts.json"
    stored = ReferenceCuts.model_validate_json(cache.read_text(encoding="utf-8"))
    assert stored.look == first.look
    monkeypatch.setattr(look_mod, "load_rgb_frames", lambda *a, **k: pytest.fail("resampled"))
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: pytest.fail("redetected"))
    assert build_profile(["a.mp4"], tmp_path, params) == first


@pytest.mark.parametrize("missing", ["stats", "flag", "both"])
def test_old_look_cache_and_flag_changes_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_reference: MediaAsset, missing: str
) -> None:
    params = ReferenceParams(dissolves=False, beats=False)
    old = _ref("a", 120, [10]).model_dump()
    old["params"] = params.model_dump()
    if missing in ("stats", "both"):
        old.pop("look")
    if missing in ("flag", "both"):
        old["params"].pop("look")
    cache = tmp_path / "style/a.cuts.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(look_mod, "load_rgb_frames", lambda *a, **k: object())
    monkeypatch.setattr(
        look_mod,
        "look_stats",
        lambda frames: LookStats(brightness=0, saturation=0, warmth=0, frames=2),
    )
    first = build_profile(["a.mp4"], tmp_path, params)
    assert first.look == LookStats(brightness=0, saturation=0, warmth=0, frames=2)
    assert json.loads(cache.read_text(encoding="utf-8"))["params"]["look"] is True
    with monkeypatch.context() as disabled:
        disabled.setattr(look_mod, "load_rgb_frames", lambda *a, **k: pytest.fail("disabled"))
        assert (
            build_profile(["a.mp4"], tmp_path, params.model_copy(update={"look": False})).look
            is None
        )
        assert json.loads(cache.read_text(encoding="utf-8"))["params"]["look"] is False
    assert build_profile(["a.mp4"], tmp_path, params).look == first.look


@pytest.mark.ffmpeg
def test_bright_and_dark_reference_look(
    make_clip: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numpy")
    source = make_clip(duration=2, audio=False)
    # Isolate RGB analysis from scene detection; both references have two known shots.
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [1])
    profiles = []
    for name, brightness in [("bright", 1), ("dark", -1)]:
        path = tmp_path / f"{name}.mp4"
        ffmpeg.run(
            "ffmpeg",
            [
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                f"eq=brightness={brightness}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                str(path),
            ],
        )
        profiles.append(build_profile([path], tmp_path / "ws", ReferenceParams(dissolves=False)))
    bright, dark = [p.look for p in profiles]
    assert bright is not None and dark is not None
    assert bright.brightness > dark.brightness + 0.8
    assert 1 <= bright.frames <= 32 and 1 <= dark.frames <= 32


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


@pytest.mark.parametrize(
    ("per_ref", "expected"),
    [
        ([[2, 3, 4]], 0.0),
        ([[0.2, 0.2, 0.2, 2, 2], [0.2, 0.2, 2]], 0.375),
        ([[2, 0.2, 0.2], [0.2, 0.2, 2]], 0.0),
        ([[0.2, 0.2, 1, 0.2, 0.2]], 0.0),  # equality breaks a run
        ([[0.2] * 4 + [2] + [0.2] * 3], 7 / 8),  # whole runs, including tail
        ([[0.2, 2, 0.2, 2, 0.2]], 0.0),  # timeline order, never sorted
        ([], None),
        ([[], []], None),
    ],
)
def test_montage_share(per_ref: list[list[float]], expected: float | None) -> None:
    assert montage_share(per_ref, median_s=2) == expected


@pytest.mark.parametrize(
    ("median_s", "rel", "min_run"),
    [(0, 0.5, 3), (-1, 0.5, 3), (2, 0, 3), (2, -0.5, 3), (2, 0.5, 1), (2, 0.5, 0)],
)
def test_montage_share_rejects_invalid_args(median_s: float, rel: float, min_run: int) -> None:
    with pytest.raises(ValueError):
        montage_share([], median_s, rel, min_run)


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        ([], None),
        (["talking"], None),
        (["talking", "broll", "talking"], 1.0),
        (["broll", "broll", "talking", "talking"], 1 / 3),
        (["talking", None, "broll"], None),
        (["talking", "broll", None, "broll", "broll"], 0.5),
        (["broll", "broll"], 0.0),
        ([None, None], None),
    ],
)
def test_kind_alternation(
    kinds: list[Literal["talking", "broll"] | None], expected: float | None
) -> None:
    assert kind_alternation(kinds) == expected


def test_montage_share_custom_threshold_and_minimum_run() -> None:
    assert montage_share([[0.5, 0.5, 2]], median_s=2, rel=0.3, min_run=2) == 2 / 3
    assert montage_share([[0.5, 0.5, 2]], median_s=2, rel=0.25, min_run=2) == 0


@pytest.mark.parametrize(
    ("sequences", "expected"),
    [
        ([[4, 6, 12, 8, 16]], 0.5),  # crossing shot stays whole: median(4,6,12) / 12
        ([[10, 10, 4, 6]], 2.0),  # shot starting exactly at 20 s belongs to the body
        ([[5, 15, 2], [20, 8]], 3.0),  # pooled median(5,15,20) / median(2,8)
        ([[5, 15, 2, 4], [2, 2, 2, 2, 12, 8, 12]], 1 / 3),  # pool shots, not ratios
        ([[20, 4, 6]], None),  # only one intro shot
        ([[10, 10, 4]], None),  # only one body shot
        ([[4, 6]], None),  # no body
        ([], None),
        ([[], []], None),
    ],
)
def test_intro_pace_ratio(sequences: list[list[float]], expected: float | None) -> None:
    assert intro_pace_ratio(sequences) == pytest.approx(expected)


def test_intro_pace_ratio_custom_window() -> None:
    assert intro_pace_ratio([[2, 3, 8, 12]], intro_s=5) == 0.25


@pytest.mark.parametrize("intro_s", [0, -1])
def test_intro_pace_ratio_rejects_non_positive_window(intro_s: float) -> None:
    with pytest.raises(ValueError, match="intro_s"):
        intro_pace_ratio([], intro_s=intro_s)


def _ref(
    name: str,
    duration: float,
    cuts: list[float],
    dissolves: list[float] | None = None,
    *,
    beats: list[float] | None = None,
) -> ReferenceCuts:
    return ReferenceCuts(
        asset_id=name, source=f"{name}.mp4", duration_s=duration, params=ReferenceParams(),
        cuts=cuts, dissolve_cuts=dissolves or [], beats=beats or [],
    )  # fmt: skip


@pytest.mark.parametrize(
    ("hard_cuts", "dissolves", "expected_cuts", "expected_dissolves"),
    [
        pytest.param([10.0, 30.0], [(9.0, 11.0)], [10.0, 30.0], [10.0], id="coincident"),
        pytest.param([10.02, 30.0], [(9.0, 11.0)], [10.0, 30.0], [10.0], id="nearby"),
        pytest.param(
            [30.0],
            [(9.0, 11.0), (9.5, 10.5)],
            [10.0, 10.0, 30.0],
            [10.0],
            id="overlap-shared-midpoint",
        ),
        pytest.param(
            [30.0],
            [(9.0, 11.0), (10.0, 12.0)],
            [10.0, 11.0, 30.0],
            [10.0, 11.0],
            id="overlap-distinct-midpoints",
        ),
        pytest.param([9.3, 30.0], [(9.0, 11.0)], [10.0, 30.0], [10.0], id="inside"),
        pytest.param([10.0, 30.0], [], [10.0, 30.0], [], id="no-dissolves"),
    ],
)
def test_combine_cuts_tracks_merged_dissolves(
    hard_cuts: list[float],
    dissolves: list[tuple[float, float]],
    expected_cuts: list[float],
    expected_dissolves: list[float],
) -> None:
    cuts, dissolve_cuts = combine_cuts(hard_cuts, dissolves, min_gap_s=0.25)
    assert cuts == profile_mod.transitions.merge_cuts(hard_cuts, dissolves, 0.25)
    assert cuts == expected_cuts
    assert dissolve_cuts == expected_dissolves
    assert set(dissolve_cuts) <= set(cuts)
    ref = _ref("a", 60.0, cuts, dissolve_cuts)
    # Shared midpoints count once even though merge_cuts preserves duplicates.
    profile = aggregate([ref], ReferenceParams())
    assert profile.transitions.dissolve == pytest.approx(len(dissolve_cuts) / len(set(cuts)))


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


def test_aggregate_intro_pace_uses_ordered_post_merge_durations() -> None:
    # The sliver merges left: [4.1, 5.9, 12, 8, 16], so intro/body medians are 5.9/12.
    profile = aggregate([_ref("a", 46, [4, 4.1, 10, 22, 30])], ReferenceParams())
    assert profile.structure is not None
    assert profile.structure.intro_pace_ratio == pytest.approx(5.9 / 12)
    assert profile.notes == (
        "1 reference, 0.8 min, 4 cuts; median shot 8.0 s "
        "(p10 4.8, p90 14.4); 5.2 cuts/min; 0.0% dissolves"
    )


def test_aggregate_without_references_raises():
    with pytest.raises(StyleError, match="reference"):
        aggregate([], ReferenceParams())


def test_aggregate_measures_bursts_after_merging_with_pooled_median() -> None:
    # After merging the sliver: [2, 2, .5, .5, .5] and [.5, .5, 2, 2, 2, 2].
    # The first reference's own median is .5, but the pooled median is 2.
    a = _ref("a", 5.5, [2, 4, 4.1, 4.5, 5])
    b = _ref("b", 9, [0.5, 1, 3, 5, 7])
    profile = aggregate([a, b], ReferenceParams())
    assert profile.shot_duration_s.median == 2
    assert profile.structure is not None
    assert profile.structure.montage_share == 3 / 11


@pytest.mark.parametrize(
    ("beats", "expected"),
    [([10, 20], 1.0), ([10], 0.5), ([9.92, 20.081], 0.5), ([5], 0.0), ([], None)],
)
def test_aggregate_beat_alignment(beats: list[float], expected: float | None) -> None:
    ref = _ref("a", 60, [10, 20], beats=beats)
    profile = aggregate([ref], ReferenceParams())
    assert profile.beat_aligned_ratio == expected
    if expected is not None:
        assert f"{expected:.1%} beat-aligned cuts" in profile.notes


def test_aggregate_pools_beat_counts_and_excludes_beatless_references() -> None:
    a = _ref("a", 60, [10, 20, 30], beats=[10, 20, 30])
    b = _ref("b", 60, [10], beats=[20])
    beatless = _ref("c", 60, [10, 20, 30, 40])
    profile = aggregate([a, b, beatless], ReferenceParams())
    assert profile.beat_aligned_ratio == 0.75  # 3/4, not mean(1, 0) or 3/8


@pytest.mark.parametrize("disabled_reference", [False, True])
def test_aggregate_beats_disabled(disabled_reference: bool) -> None:
    a = _ref("a", 60, [10], beats=[10, 10.5])
    b = _ref("b", 60, [20], beats=[20, 20.5])
    params = ReferenceParams(beats=disabled_reference)
    if disabled_reference:
        b.params.beats = False
    assert aggregate([a, b], params).beat_aligned_ratio is None
    assert aggregate([a, b], params).music_bpm is None


def test_aggregate_beats_use_surviving_rounded_boundaries() -> None:
    ref = _ref("a", 8, [2, 2.1, 5.0000001], [5.0000001], beats=[2, 5.08])
    assert aggregate([ref], ReferenceParams()).beat_aligned_ratio == 0.5


def test_beat_pool_without_surviving_cuts_is_unmeasured() -> None:
    ref = _ref("a", 60, [], beats=[10])
    assert pooled_beat_ratio([ref], [[]]) is None
    # A beatless reference can supply pacing while the measured pool has no cuts.
    beatless = _ref("b", 60, [20])
    assert aggregate([ref, beatless], ReferenceParams()).beat_aligned_ratio is None


@pytest.mark.parametrize("dissolves", [False, True])
def test_unmeasured_beats_preserve_notes(dissolves: bool) -> None:
    ref = _ref("a", 60, [20, 40])
    expected = "1 reference, 1.0 min, 2 cuts; median shot 20.0 s (p10 20.0, p90 20.0); 2.0 cuts/min"
    if dissolves:
        expected += "; 0.0% dissolves"
    assert aggregate([ref], ReferenceParams(dissolves=dissolves)).notes == expected
    ref.beats = [20, 40]
    ref.params.beats = False
    assert aggregate([ref], ReferenceParams(dissolves=dissolves)).notes == expected


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


def test_transition_mix_handles_unmeasured_and_pooled_counts():
    assert transition_mix(0, 0) is None
    assert transition_mix(None, 5) is None
    mix = transition_mix(1, 4)
    assert mix.dissolve == pytest.approx(0.25)
    assert mix.hard_cut == pytest.approx(0.75)


@pytest.mark.parametrize("cuts", [[], [0.1]])
@pytest.mark.parametrize("detection", [False, True])
def test_aggregate_zero_surviving_cuts_raises(cuts, detection):
    ref = _ref("a", 60, cuts, cuts)
    for refs in ([ref], [ref, _ref("b", 120, [])]):
        with pytest.raises(StyleError, match="two shots"):
            aggregate(refs, ReferenceParams(dissolves=detection))


def test_analyze_reference_records_only_surviving_dissolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = MediaAsset(
        asset_id="a",
        path="a.mp4",
        proxy_path="proxy.mp4",
        has_audio=False,
        duration_s=60,
        fps=30,
        width=64,
        height=64,
    )
    monkeypatch.setattr(profile_mod, "probe_asset", lambda path: asset)
    monkeypatch.setattr(profile_mod, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [10])
    monkeypatch.setattr(profile_mod.transitions, "load_frames", lambda path: None)
    found = [(10.05, 10.15), (19, 21)]
    monkeypatch.setattr(profile_mod.transitions, "find_dissolves", lambda *a: found)

    def merge_cuts(hard, dissolves, min_gap_s):
        assert hard == [10] and dissolves == found
        assert abs(sum(dissolves[0]) / 2 - hard[0]) < min_gap_s
        # Simulate a removed midpoint; surviving midpoints stay unchanged.
        return [10, 20]

    monkeypatch.setattr(profile_mod.transitions, "merge_cuts", merge_cuts)
    record = analyze_reference("a.mp4", tmp_path, ReferenceParams(look=False))
    assert record.cuts == [10, 20]
    assert record.dissolve_cuts == [record.cuts[1]]
    cached = tmp_path / "style" / "a.cuts.json"
    assert ReferenceCuts.model_validate_json(
        cached.read_text(encoding="utf-8")
    ) == record.model_copy(update={"source": str(Path("a.mp4").resolve())})


@pytest.mark.parametrize("detection", [False, True])
@pytest.mark.parametrize("invalid_subset", [False, True])
def test_old_reference_cache_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detection: bool, invalid_subset: bool
) -> None:
    params = ReferenceParams(dissolves=detection, look=False)
    old = _ref("a", 60, [10]).model_dump(exclude={"dissolve_cuts"})
    if invalid_subset:
        old["dissolve_cuts"] = [20]
    old["params"] = params.model_dump()
    cache = tmp_path / "style" / "a.cuts.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(old), encoding="utf-8")
    asset = MediaAsset(
        asset_id="a",
        path="a.mp4",
        proxy_path="proxy.mp4",
        has_audio=False,
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
    assert ReferenceCuts.model_validate_json(
        cache.read_text(encoding="utf-8")
    ) == record.model_copy(update={"source": str(Path("a.mp4").resolve())})
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: pytest.fail("re-detected"))
    assert analyze_reference("a.mp4", tmp_path, params) == record


@pytest.mark.parametrize("missing", ["times", "flag", "both"])
def test_old_beat_cache_is_recomputed_and_beats_switch_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    params = ReferenceParams(dissolves=False, look=False)
    old = _ref("a", 60, [10]).model_dump()
    old["params"] = params.model_dump()
    if missing in ("times", "both"):
        old.pop("beats")
    if missing in ("flag", "both"):
        old["params"].pop("beats")
    cache = tmp_path / "style" / "a.cuts.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps(old), encoding="utf-8")
    asset = MediaAsset(
        asset_id="a", path="a.mp4", proxy_path="proxy.mp4", duration_s=60,
        fps=30, width=64, height=64,
    )  # fmt: skip
    monkeypatch.setattr(profile_mod, "probe_asset", lambda path: asset)
    monkeypatch.setattr(profile_mod, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [10, 20])
    samples = [0.0, 1.0, 0.0]

    def decode(path: str, sample_rate: int) -> list[float]:
        assert path == asset.proxy_path and sample_rate == 22050
        return samples

    def detect(audio: list[float], sample_rate: int) -> beats_mod.BeatResult:
        assert audio is samples and sample_rate == 22050
        return beats_mod.BeatResult(120, [20, 10])

    monkeypatch.setattr(pauses_mod, "load_mono_audio", decode)
    monkeypatch.setattr(beats_mod, "detect_beats", detect)
    record = analyze_reference("a.mp4", tmp_path, params)
    assert record.beats == [10, 20]
    assert ReferenceCuts.model_validate_json(
        cache.read_text(encoding="utf-8")
    ) == record.model_copy(update={"source": str(Path("a.mp4").resolve())})
    monkeypatch.setattr(beats_mod, "detect_beats", lambda *a: pytest.fail("re-detected beats"))
    monkeypatch.setattr(pauses_mod, "load_mono_audio", lambda *a: pytest.fail("re-decoded"))
    assert analyze_reference("a.mp4", tmp_path, params) == record
    assert build_profile(["a.mp4"], tmp_path, params).beat_aligned_ratio == 1.0
    # Pool beat intervals, rather than using the detector's separate BPM estimate.
    assert build_profile(["a.mp4"], tmp_path, params).music_bpm == 6.0

    disabled = params.model_copy(update={"beats": False})
    record = analyze_reference("a.mp4", tmp_path, disabled)
    assert record.params == disabled and record.beats == []
    assert build_profile(["a.mp4"], tmp_path, disabled).music_bpm is None
    monkeypatch.setattr(pauses_mod, "load_mono_audio", decode)
    monkeypatch.setattr(beats_mod, "detect_beats", detect)
    assert analyze_reference("a.mp4", tmp_path, params).beats == [10, 20]


@pytest.mark.parametrize("has_audio", [False, True])
def test_reference_without_audio_has_no_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_audio: bool
) -> None:
    asset = MediaAsset(
        asset_id="a", path="a.mp4", proxy_path="proxy.mp4", duration_s=60,
        fps=30, width=64, height=64, has_audio=has_audio,
    )  # fmt: skip
    monkeypatch.setattr(profile_mod, "probe_asset", lambda path: asset)
    monkeypatch.setattr(profile_mod, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots_mod, "detect_cuts", lambda *a, **k: [10, 20])
    monkeypatch.setattr(
        pauses_mod,
        "load_mono_audio",
        lambda *a: None if has_audio else pytest.fail("decoded asset without audio"),
    )
    monkeypatch.setattr(beats_mod, "detect_beats", lambda *a: pytest.fail("detected no audio"))
    record = analyze_reference("a.mp4", tmp_path, ReferenceParams(dissolves=False, look=False))
    assert record.beats == []


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
@pytest.mark.parametrize("phase_s", [0.0, 0.25])
def test_reference_pulse_audio_alignment(
    make_edited_clip: Callable[..., Path], tmp_path: Path, phase_s: float
) -> None:
    from vlogcut.ingest import ffmpeg

    video = make_edited_clip("cuts.mp4", shot_lengths=[1.0] * 8, size="64x64")
    reference = tmp_path / "with_audio.mp4"
    # 120 BPM pulses: integer-second cuts align at phase 0, miss at phase 0.25.
    pulse = f"aevalsrc='0.8*sin(2*PI*1000*t)*lt(mod(t+{phase_s},0.5),0.02)':s=22050:d=8"
    ffmpeg.run(
        "ffmpeg",
        [
            "-v", "error", "-y", "-i", str(video), "-f", "lavfi", "-i", pulse,
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(reference),
        ],
    )  # fmt: skip
    profile = build_profile([reference], tmp_path / "ws", ReferenceParams(dissolves=False))
    assert profile.beat_aligned_ratio is not None
    if phase_s == 0:
        assert profile.beat_aligned_ratio > 0.8
    else:
        assert profile.beat_aligned_ratio < 0.2


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
