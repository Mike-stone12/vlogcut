from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vlogcut.plan import RoughCutError, RoughCutParams, rough_cut, score_shot
from vlogcut.plan.rules import (
    DEFAULT_PACING,
    assign_lengths,
    assign_transitions,
    expected_length,
    fit_lengths,
    place_on_cut_points,
    sample_lengths,
)
from vlogcut.schemas import (
    CutPoint,
    DurationStats,
    MediaAsset,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.style.profile import duration_stats

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def _asset(aid: str, minutes: float | None, path: str | None = None) -> MediaAsset:
    created = None if minutes is None else T0 + timedelta(minutes=minutes)
    return MediaAsset(
        asset_id=aid, path=path or f"{aid}.mp4", duration_s=600, fps=30, width=1080,
        height=1920, created_at=created,
    )  # fmt: skip


def _shot(aid: str, i: int, dur: float = 6.0, **q) -> Shot:
    quality = QualityFlags(**{"sharpness": 200.0, "exposure_clip": 0.0, "shake": 0.5, **q})
    return Shot(
        shot_id=f"{aid}_{i:03d}", asset_id=aid, start_s=i * dur, end_s=(i + 1) * dur,
        quality=quality,
    )  # fmt: skip


def _day(n_per_asset: int = 20) -> tuple[list[Shot], list[MediaAsset]]:
    # recorded order: morning (b), noon (a), evening (c); one asset without a date
    assets = [_asset("a", 180), _asset("b", 0), _asset("c", 600), _asset("z", None, "zz.mp4")]
    shots = [_shot(a.asset_id, i) for a in assets for i in range(n_per_asset)]
    return shots, assets


def _by_id(shots: list[Shot]) -> dict[str, Shot]:
    return {s.shot_id: s for s in shots}


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [4.123456, 7.123456])
def test_rough_cut_aligns_rounded_window_to_nearby_candidate(offset: float) -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, 9.0))
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=3.001)
    plain = rough_cut([shot], assets, params).clips[0]
    candidate = shot.start_s + offset
    shot.cut_points = [CutPoint(time_s=candidate, kind="pause")]
    aligned = rough_cut([shot], assets, params).clips[0]
    assert (
        min(abs(shot.start_s + edge - candidate) for edge in (aligned.in_s, aligned.out_s)) < 1e-6
    )
    assert aligned.timeline_duration_s == pytest.approx(plain.timeline_duration_s, abs=1e-12)
    assert (aligned.in_s, aligned.out_s) != (plain.in_s, plain.out_s)


def test_rough_cut_max_shift_and_unbounded_placement() -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, 9.0))
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=3)
    plain = rough_cut([shot], assets, params)
    shot.cut_points = [CutPoint(time_s=108.5, kind="pause")]
    assert rough_cut([shot], assets, params).model_dump_json() == plain.model_dump_json()
    for limit in (2.0, None):
        clip = rough_cut([shot], assets, params, max_shift_s=limit).clips[0]
        assert shot.start_s + clip.out_s == 108.5
        assert clip.timeline_duration_s == plain.clips[0].timeline_duration_s


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_rough_cut_placement_preserves_multi_shot_invariants(strategy: str) -> None:
    assets = [_asset("a", 0), _asset("b", 60)]
    shots = [_shot(a.asset_id, i, dur=8, usable_range=(1.0, 7.0)) for a in assets for i in range(6)]
    shots[2].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=24, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=0.3, median=2, p90=5),
        cuts_per_min=30,
        transitions=TransitionMix(hard_cut=0.5, dissolve=0.5),
    )
    plain = rough_cut(shots, assets, params, profile)
    by_id = _by_id(shots)
    for i, clip in enumerate(plain.clips):
        shot = by_id[clip.shot_id]
        if i % 3 != 0 or "kept_for_audio" in shot.quality.warnings:
            shot.cut_points = [
                CutPoint(time_s=shot.start_s + offset, kind="pause")
                for offset in (0.5, clip.in_s + 0.125, 7.5)
            ]
    aligned = rough_cut(shots, assets, params, profile)
    assert any(a.in_s != b.in_s for a, b in zip(plain.clips, aligned.clips, strict=True))
    assert aligned.model_dump(exclude={"clips"}) == plain.model_dump(exclude={"clips"})
    assert aligned.total_duration_s == pytest.approx(plain.total_duration_s, abs=1e-12)
    for a, b in zip(plain.clips, aligned.clips, strict=True):
        assert a.model_dump(exclude={"in_s", "out_s"}) == b.model_dump(exclude={"in_s", "out_s"})
        assert b.timeline_duration_s == pytest.approx(a.timeline_duration_s, abs=1e-12)
        shot = by_id[b.shot_id]
        lo, hi = shot.quality.usable_range
        assert shot.start_s <= shot.start_s + b.in_s < shot.start_s + b.out_s <= shot.end_s
        assert lo <= b.in_s < b.out_s <= hi
        if not shot.cut_points:
            assert b == a


@pytest.mark.parametrize("usable_end,expected_in", [(9.0, 1.25), (4.0, 1.0)])
def test_rough_cut_places_audio_shots_unless_window_is_full(
    usable_end: float, expected_in: float
) -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, usable_end), warnings=["kept_for_audio"])
    shot.cut_points = [CutPoint(time_s=101.25, kind="pause")]
    clip = rough_cut([shot], [_asset("a", 0)], RoughCutParams(target_s=3)).clips[0]
    assert clip.in_s == expected_in
    assert clip.timeline_duration_s == 3
    assert clip.audio == "keep"


@pytest.mark.parametrize("limit", [0.0, 1.0, None])
def test_rough_cut_no_candidates_matches_legacy_json(
    monkeypatch: pytest.MonkeyPatch, limit: float | None
) -> None:
    shots, assets = _day(5)
    # Include an awkward full-window rounding case from the previous behaviour.
    shots[0].quality.usable_range = (0.0004, 4.0004)
    shots[0].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=180)
    actual = rough_cut(shots, assets, params, max_shift_s=limit).model_dump_json()
    monkeypatch.setattr("vlogcut.plan.rules._place_clip", lambda *args, **kwargs: None)
    legacy = rough_cut(shots, assets, params).model_dump_json()
    assert actual.encode() == legacy.encode()


def test_place_on_cut_points_no_candidates_returns_default() -> None:
    assert place_on_cut_points(100, 110, 3, 103.125, []) == 103.125


@pytest.mark.parametrize("candidate,expected", [(107.25, 104.25), (104.25, 104.25)])
def test_place_on_cut_points_snaps_end_or_start(candidate: float, expected: float) -> None:
    start = place_on_cut_points(100, 110, 3, 104, [candidate])
    assert start == expected
    assert candidate in (start, start + 3)


@pytest.mark.parametrize("candidates", [[99, 111], [102, 108], [99, 102, 108, 111]])
def test_place_on_cut_points_ignores_infeasible_placements(candidates: list[float]) -> None:
    # Interior candidates also fail: neither a start nor an end can fit 9 s.
    assert place_on_cut_points(100, 110, 9, 100.5, candidates) == 100.5


@pytest.mark.parametrize("limit,expected", [(None, 105), (0, 103), (1.5, 103), (2, 105)])
def test_place_on_cut_points_max_shift(limit: float | None, expected: float) -> None:
    assert place_on_cut_points(100, 110, 3, 103, [108], max_shift_s=limit) == expected


def test_place_on_cut_points_zero_shift_allows_default_candidate() -> None:
    assert place_on_cut_points(100, 110, 3, 103, [106], max_shift_s=0) == 103


@pytest.mark.parametrize("candidates", [[106, 108, 104], [104, 108, 106]])
def test_place_on_cut_points_tie_prefers_both_ends(candidates: list[float]) -> None:
    # Starts 102 and 104 tie in distance, but only [104, 108] aligns both ends.
    assert place_on_cut_points(100, 110, 4, 103, candidates) == 104


@pytest.mark.parametrize("candidates", [[104, 102], [102, 104]])
def test_place_on_cut_points_tie_prefers_earlier_start(candidates: list[float]) -> None:
    assert place_on_cut_points(100, 110, 4, 103, candidates) == 102


def test_place_on_cut_points_smallest_shift_beats_both_ends() -> None:
    assert place_on_cut_points(100, 110, 4, 103, [103.25, 104, 108]) == 103.25


def test_place_on_cut_points_preserves_length_and_input() -> None:
    candidates = [108, 102, 107.5, 102]
    original = candidates.copy()
    length = 3.5
    start = place_on_cut_points(100, 110, length, 103.5, candidates)
    end = start + length
    assert (start, end) == (104, 107.5)
    assert end - start == length
    assert 100 <= start < end <= 110
    assert candidates == original
    assert place_on_cut_points(100, 110, length, 103.5, tuple(candidates)) == start


@pytest.mark.parametrize("candidate", [100, 110])
def test_place_on_cut_points_accepts_window_boundaries(candidate: float) -> None:
    assert place_on_cut_points(100, 110, 10, 100, [candidate]) == 100


@pytest.mark.parametrize("candidate,expected", [(-5e-10, -5e-10), (10 + 5e-10, 7 + 5e-10)])
def test_place_on_cut_points_window_tolerance(candidate: float, expected: float) -> None:
    start = place_on_cut_points(0, 10, 3, 4, [candidate])
    assert start == expected
    assert -1e-9 <= start and start + 3 <= 10 + 1e-9


def test_place_on_cut_points_short_window_within_tolerance() -> None:
    assert place_on_cut_points(0, 3 - 5e-10, 3, 0, [0]) == 0


@pytest.mark.parametrize(
    "lo,hi,length,limit,message",
    [
        (0, 10, 0, None, "length_s must be positive"),
        (0, 10, -1, None, "length_s must be positive"),
        (0, 2, 3, None, "window is shorter"),
        (0, 3 - 2e-9, 3, None, "window is shorter"),
        (10, 0, 3, None, "window is shorter"),
        (0, 10, 3, -0.1, "max_shift_s must be non-negative"),
    ],
)
def test_place_on_cut_points_invalid_arguments(
    lo: float, hi: float, length: float, limit: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        place_on_cut_points(lo, hi, length, 0, [], max_shift_s=limit)


def test_sample_lengths_small_counts_and_expected_length():
    assert sample_lengths(DEFAULT_PACING, 1) == [3.0]
    assert sample_lengths(DEFAULT_PACING, 2) == [1.5, 5.0]
    assert expected_length(DEFAULT_PACING) == pytest.approx(3.15)
    flat = DurationStats(p10=2, median=2, p90=2)
    assert sample_lengths(flat, 12) == [2.0] * 12


@pytest.mark.parametrize("k", [12, 19])
@pytest.mark.parametrize("pace", [DEFAULT_PACING, DurationStats(p10=0.5, median=0.77, p90=1)])
def test_sample_lengths_reproduce_profile(pace, k):
    lengths = sample_lengths(pace, k)
    assert len(lengths) == k and lengths == sorted(lengths)
    assert all(pace.p10 <= length <= pace.p90 for length in lengths)
    measured = duration_stats(lengths)
    for field in ("p10", "median", "p90"):
        assert getattr(measured, field) == pytest.approx(getattr(pace, field), rel=0.03)
    assert sum(lengths) / k == pytest.approx(expected_length(pace), rel=0.02)


def test_assign_lengths_by_capacity_score_and_chronology():
    targets, scores = [0.5, 1, 2], [0.5, 0.9, 0.9]
    assert assign_lengths(targets, [3, 1, 3], scores, [False] * 3) == [1, 0.5, 2]
    assert assign_lengths(targets, [3, 1, 1.1], scores, [False, False, True]) == [2, 0.5, 1.1]
    assert assign_lengths(targets, [3] * 3, [0.9] * 3, [False] * 3) == [2, 1, 0.5]
    assert assign_lengths(targets, [3, 1, 1.1], scores, [True] * 3) == [3, 1, 1.1]
    assert targets == [0.5, 1, 2]  # assignment does not consume the caller's samples


def test_fit_lengths_hits_target_within_bounds():
    out = fit_lengths([3, 3, 3], [1.5, 1.5, 1.5], [5, 5, 2], 11)
    assert sum(out) == pytest.approx(11, abs=1e-3)
    assert out[2] == 2 and all(1.5 <= x <= 5 for x in out)


@pytest.mark.parametrize(
    "lengths,share", [([2.0] * 5, None), ([2.0], 1.0), ([], 0.5), ([2.0] * 5, 0)]
)
def test_assign_transitions_all_cuts(lengths: list[float], share: float | None):
    assert assign_transitions(lengths, share) == [Transition() for _ in lengths]


@pytest.mark.parametrize("share,indices", [(0.5, [2, 4]), (1.0, [1, 2, 3, 4]), (0.625, [2, 4])])
def test_assign_transitions_even_and_deterministic(share: float, indices: list[int]):
    lengths = [2.0] * 5
    transitions = assign_transitions(lengths, share)
    assert [i for i, t in enumerate(transitions) if t.kind == "dissolve"] == indices
    assert all(transitions[i].duration_s == 0.5 for i in indices)
    assert transitions == assign_transitions(lengths, share)
    assert lengths == [2.0] * 5


def test_assign_transitions_caps_duration_by_both_neighbours():
    transitions = assign_transitions([4, 0.3, 2, 4], 1.0)
    assert [t.duration_s for t in transitions] == pytest.approx([0, 0.15, 0.15, 0.5])
    assert assign_transitions([4, 4], 1.0, duration_s=0.8)[1].duration_s == 0.8


def test_rough_cut_transition_mix_preserves_clip_times():
    shots, assets = _day()
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=0.3, median=0.7, p90=1.3), cuts_per_min=80
    )
    params = RoughCutParams(target_s=30)
    plain = rough_cut(shots, assets, params, profile=profile)
    mixed_profile = profile.model_copy(
        update={"transitions": TransitionMix(hard_cut=0.75, dissolve=0.25)}
    )
    mixed = rough_cut(shots, assets, params, profile=mixed_profile)
    assert all(c.transition_in == Transition() for c in plain.clips)
    k = len(mixed.clips)
    dissolves = sum(c.transition_in.kind == "dissolve" for c in mixed.clips)
    assert dissolves / (k - 1) == pytest.approx(0.25, abs=1 / (k - 1))
    assert mixed.clips[0].transition_in == Transition()
    for i, (a, b) in enumerate(zip(plain.clips, mixed.clips, strict=True)):
        assert a.model_dump(exclude={"transition_in"}) == b.model_dump(exclude={"transition_in"})
        assert a.timeline_duration_s == b.timeline_duration_s
        if b.transition_in.kind == "dissolve":
            assert b.transition_in.duration_s == min(
                0.5, 0.5 * min(mixed.clips[i - 1].timeline_duration_s, b.timeline_duration_s)
            )
    assert plain.total_duration_s == mixed.total_duration_s


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("target", [30, 60, 120])
def test_duration_within_tolerance_and_basic_invariants(strategy, target):
    shots, assets = _day()
    edl = rough_cut(shots, assets, RoughCutParams(target_s=target, strategy=strategy))
    assert edl.total_duration_s == pytest.approx(target, rel=0.1)
    ids = [c.shot_id for c in edl.clips]
    assert len(ids) == len(set(ids))  # each shot used once
    for clip in edl.clips:
        assert 1.5 - 1e-6 <= clip.out_s - clip.in_s <= 5 + 1e-6
        assert clip.transition_in == Transition()


def test_clips_are_chronological_and_undated_assets_go_last():
    shots, assets = _day()
    edl = rough_cut(shots, assets, RoughCutParams(target_s=90))
    asset_seq = [c.shot_id.split("_")[0] for c in edl.clips]
    first_seen = list(dict.fromkeys(asset_seq))
    assert first_seen == ["b", "a", "c", "z"]
    shot_index = [int(c.shot_id.split("_")[1]) for c in edl.clips]
    for aid in "bacz":
        idx = [i for a, i in zip(asset_seq, shot_index, strict=True) if a == aid]
        assert idx == sorted(idx)


def test_rejected_shots_skipped_and_usable_range_respected():
    shots, assets = _day(10)
    by_id = _by_id(shots)
    for s in shots[::2]:
        s.quality.rejected, s.quality.reject_reasons = True, ["too_dark"]
    for s in shots[1::4]:
        s.quality.usable_range = (2.0, 5.5)
    edl = rough_cut(shots, assets, RoughCutParams(target_s=40))
    for clip in edl.clips:
        q = by_id[clip.shot_id].quality
        assert not q.rejected
        lo, hi = q.usable_range or (0, 6)
        assert lo - 1e-6 <= clip.in_s < clip.out_s <= hi + 1e-6


def test_coverage_spreads_over_the_day_highlights_concentrates():
    shots, assets = _day(25)  # 100 shots
    for s in shots[:25]:  # asset "a" (noon) is much sharper than the rest
        s.quality.sharpness = 400.0
    for s in shots[25:]:
        s.quality.sharpness, s.quality.warnings = 60.0, ["blurry"]
    params = dict(target_s=60)
    cover = rough_cut(shots, assets, RoughCutParams(**params, strategy="coverage"))
    best = rough_cut(shots, assets, RoughCutParams(**params, strategy="highlights"))
    assert {c.shot_id[0] for c in cover.clips} == {"a", "b", "c", "z"}
    assert {c.shot_id[0] for c in best.clips} == {"a"}


def test_coverage_keeps_every_moment_even_next_to_a_long_one_take():
    assets = [_asset("a", 0), _asset("b", 60), _asset("c", 120)]
    shots = [_shot("a", i) for i in range(2)] + [_shot("b", i) for i in range(12)]
    shots += [_shot("c", 0, sharpness=50.0, warnings=["dark"])]  # the last, weaker scene
    edl = rough_cut(shots, assets, RoughCutParams(target_s=15))
    assert [c.shot_id[0] for c in edl.clips][-1] == "c"
    assert {c.shot_id[0] for c in edl.clips} == {"a", "b", "c"}
    tiny = rough_cut(shots, assets, RoughCutParams(target_s=5))  # fewer slots than moments
    assert len(tiny.clips) <= 2


def test_talking_shot_starts_at_usable_start_and_runs_long():
    shots, assets = _day(10)
    talk = shots[13]  # asset "b"
    talk.quality.warnings = ["shaky", "kept_for_audio"]
    talk.quality.usable_range = (0.5, 6.0)
    edl = rough_cut(shots, assets, RoughCutParams(target_s=40))
    clip = next(c for c in edl.clips if c.shot_id == talk.shot_id)
    assert clip.in_s == 0.5
    assert clip.out_s - clip.in_s >= 4.0
    assert score_shot(talk) > score_shot(shots[12])


def test_style_profile_pacing_changes_the_cut():
    shots, assets = _day()
    fast = StyleProfile(
        shot_duration_s=DurationStats(p10=0.8, median=1.2, p90=2.0),
        cuts_per_min=50,
        beat_aligned_ratio=0.5,
    )
    slow = rough_cut(shots, assets, RoughCutParams(target_s=60))
    quick = rough_cut(shots, assets, RoughCutParams(target_s=60), profile=fast)
    assert len(quick.clips) > 2 * len(slow.clips)
    assert all(c.out_s - c.in_s <= 2.0 + 1e-6 for c in quick.clips)
    for edl, pace in ((quick, fast.shot_duration_s), (slow, DEFAULT_PACING)):
        measured = duration_stats([c.timeline_duration_s for c in edl.clips])
        for field in ("p10", "median", "p90"):
            assert getattr(measured, field) == pytest.approx(getattr(pace, field), rel=0.1)


def test_short_footage_uses_everything_and_says_so():
    assets = [_asset("a", 0)]
    shots = [_shot("a", i, dur=4.0) for i in range(3)]
    edl = rough_cut(shots, assets, RoughCutParams(target_s=180))
    assert len(edl.clips) == 3
    assert edl.total_duration_s == pytest.approx(12.0)
    assert "too short" in edl.notes


def test_nothing_usable_raises():
    assets = [_asset("a", 0)]
    shots = [_shot("a", 0, rejected=True, reject_reasons=["too_dark"]), _shot("a", 1, dur=1.0)]
    with pytest.raises(RoughCutError):
        rough_cut(shots, assets)
