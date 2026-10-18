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


def test_style_profile_cut_rate_accepts_zero_but_not_negative():
    stats = DurationStats(p10=60, median=60, p90=60)
    assert StyleProfile(shot_duration_s=stats, cuts_per_min=0).transitions is None
    with pytest.raises(ValidationError):
        StyleProfile(shot_duration_s=stats, cuts_per_min=-1)
