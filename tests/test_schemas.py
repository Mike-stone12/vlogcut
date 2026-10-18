from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlogcut.schemas import (
    EDL,
    Clip,
    DurationStats,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)


def test_shot_duration_and_bounds():
    q = QualityFlags(sharpness=120.0, exposure_clip=0.01, shake=0.2)
    shot = Shot(shot_id="s1", asset_id="a1", start_s=3.0, end_s=7.5, quality=q)
    assert shot.duration_s == pytest.approx(4.5)
    with pytest.raises(ValidationError):
        Shot(shot_id="s2", asset_id="a1", start_s=5, end_s=5, quality=q)


def test_quality_usable_range_must_fit_the_shot():
    q = QualityFlags(sharpness=50, exposure_clip=0, shake=0.1, usable_range=(0.6, 4.0))
    assert Shot(shot_id="s", asset_id="a", start_s=10, end_s=14.5, quality=q)
    with pytest.raises(ValidationError):
        Shot(shot_id="s", asset_id="a", start_s=10, end_s=13, quality=q)
    with pytest.raises(ValidationError):
        QualityFlags(sharpness=50, exposure_clip=0, shake=0, usable_range=(2.0, 1.0))
    assert QualityFlags(sharpness=1, exposure_clip=0, shake=0).warnings == []


def test_clip_speed_changes_timeline_duration():
    clip = Clip(shot_id="s1", in_s=0, out_s=4, speed=2.0)
    assert clip.timeline_duration_s == pytest.approx(2.0)


def test_edl_requires_clips():
    with pytest.raises(ValidationError):
        EDL(variant_id="v", target_duration_s=10, clips=[])


def test_old_edl_json_defaults_to_hard_cuts():
    edl = EDL.model_validate_json(
        '{"variant_id": "v", "target_duration_s": 4, "clips": ['
        '{"shot_id": "s1", "in_s": 0, "out_s": 2},'
        '{"shot_id": "s2", "in_s": 0, "out_s": 2}]}'
    )
    assert all(c.transition_in == Transition(kind="cut", duration_s=0) for c in edl.clips)
    assert edl.clips[0].transition_in is not edl.clips[1].transition_in


@pytest.mark.parametrize(
    "kind,duration", [("cut", 0.5), ("dissolve", 0), ("dissolve", -0.5), ("fade", 0.5)]
)
def test_transition_rejects_invalid_kind_or_duration(kind: str, duration: float):
    with pytest.raises(ValidationError):
        Transition.model_validate({"kind": kind, "duration_s": duration})


def test_edl_rejects_first_clip_dissolve():
    clip = Clip(
        shot_id="s", in_s=0, out_s=2, transition_in=Transition(kind="dissolve", duration_s=0.5)
    )
    with pytest.raises(ValidationError, match="first clip"):
        EDL(variant_id="v", target_duration_s=2, clips=[clip])


def test_dissolve_preserves_edl_timing_and_round_trips():
    clips = [Clip(shot_id=f"s{i}", in_s=1, out_s=5, speed=2) for i in range(2)]
    clips[1].transition_in = Transition(kind="dissolve", duration_s=0.5)
    edl = EDL(variant_id="v", target_duration_s=4, clips=clips)
    assert edl.total_duration_s == 4
    assert all(c.timeline_duration_s == 2 and (c.in_s, c.out_s) == (1, 5) for c in clips)
    assert EDL.model_validate_json(edl.model_dump_json()) == edl


def test_style_profile_duration_stats_ordered():
    with pytest.raises(ValidationError):
        DurationStats(p10=3, median=2, p90=4)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=0.8, median=1.6, p90=3.2),
        cuts_per_min=32,
        beat_aligned_ratio=0.7,
    )
    assert profile.version == 1


def test_style_profile_unmeasured_ratios_are_none():
    stats = DurationStats(p10=1.0, median=2.0, p90=4.0)
    profile = StyleProfile(shot_duration_s=stats, cuts_per_min=20)
    assert profile.beat_aligned_ratio is None and profile.speech_ratio is None
    with pytest.raises(ValidationError):
        StyleProfile(shot_duration_s=stats, cuts_per_min=20, speech_ratio=1.5)


@pytest.mark.parametrize("hard,dissolve", [(0.2, 0.3), (-0.1, 1.1), (1.1, -0.1), (0.5, 0.500002)])
def test_transition_mix_rejects_invalid_shares(hard, dissolve):
    with pytest.raises(ValidationError):
        TransitionMix(hard_cut=hard, dissolve=dissolve)


@pytest.mark.parametrize("hard,dissolve", [(1, 0), (0, 1), (0.75, 0.25), (0.5, 0.5000005)])
def test_transition_mix_accepts_shares_within_tolerance(hard, dissolve):
    assert TransitionMix(hard_cut=hard, dissolve=dissolve).dissolve == dissolve


def test_old_profile_json_has_unmeasured_transitions():
    profile = StyleProfile.model_validate_json(
        '{"shot_duration_s": {"p10": 1, "median": 2, "p90": 4}, "cuts_per_min": 20}'
    )
    assert profile.transitions is None


@pytest.mark.parametrize("rate", [0, -1])
def test_style_profile_requires_positive_cut_rate(rate):
    stats = DurationStats(p10=60, median=60, p90=60)
    with pytest.raises(ValidationError):
        StyleProfile(shot_duration_s=stats, cuts_per_min=rate)
