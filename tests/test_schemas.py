from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlogcut.schemas import EDL, Clip, DurationStats, QualityFlags, Shot, StyleProfile


def test_shot_duration_and_bounds():
    q = QualityFlags(sharpness=120.0, exposure_clip=0.01, shake=0.2)
    shot = Shot(shot_id="s1", asset_id="a1", start_s=3.0, end_s=7.5, quality=q)
    assert shot.duration_s == pytest.approx(4.5)
    with pytest.raises(ValidationError):
        Shot(shot_id="s2", asset_id="a1", start_s=5, end_s=5, quality=q)


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
