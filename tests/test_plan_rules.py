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
    sample_lengths,
)
from vlogcut.schemas import (
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
