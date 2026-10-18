from __future__ import annotations

import pytest

from vlogcut.pipeline import cut
from vlogcut.schemas import EDL, Clip, DurationStats, StyleProfile
from vlogcut.style import StyleError, build_profile, profile_distance, profile_from_edl
from vlogcut.style.distance import format_diff, format_profile


def _profile(p10: float, median: float, p90: float, cpm: float, **extra) -> StyleProfile:
    stats = DurationStats(p10=p10, median=median, p90=p90)
    return StyleProfile(shot_duration_s=stats, cuts_per_min=cpm, **extra)


def _edl(lengths: list[float], speed: float = 1.0) -> EDL:
    clips = [
        Clip(shot_id=f"s{i}", in_s=0.0, out_s=length * speed, speed=speed)
        for i, length in enumerate(lengths)
    ]
    return EDL(variant_id="t", target_duration_s=sum(lengths), clips=clips)


# ---------------------------------------------------------------------------
# pure
# ---------------------------------------------------------------------------


def test_distance_is_zero_for_identical_and_one_octave_for_double_speed():
    a = _profile(1.0, 2.0, 4.0, 20.0)
    assert profile_distance(a, a).total == 0.0
    b = _profile(0.5, 1.0, 2.0, 40.0)  # everything twice as fast
    d = profile_distance(a, b)
    assert d.total == pytest.approx(1.0)
    assert d.per_dim == pytest.approx({"p10": 1.0, "median": 1.0, "p90": 1.0, "cuts_per_min": 1.0})
    assert profile_distance(b, a).total == pytest.approx(d.total)  # symmetric
    assert d.unmeasured == ["beat_aligned_ratio", "speech_ratio", "dissolve"]


def test_ratio_dims_count_only_when_measured_on_both_sides():
    a = _profile(1.0, 2.0, 4.0, 20.0, beat_aligned_ratio=0.8, speech_ratio=0.5)
    b = _profile(1.0, 2.0, 4.0, 20.0, beat_aligned_ratio=0.3)
    d = profile_distance(a, b)
    assert d.per_dim["beat_aligned_ratio"] == pytest.approx(0.5)
    assert d.unmeasured == ["speech_ratio", "dissolve"]
    assert d.total == pytest.approx(0.5 / 5)  # four pacing dims at 0 plus one ratio dim


def test_profile_from_edl_uses_timeline_durations():
    p = profile_from_edl(_edl([1.0, 2.0, 1.0]))
    assert p.cuts_per_min == pytest.approx(30.0)  # 2 cuts in 4 s
    assert p.shot_duration_s.median == pytest.approx(1.0)
    assert p.source_refs == ["edl"]
    assert p.transitions is None
    sped_up = profile_from_edl(_edl([1.0, 2.0, 1.0], speed=2.0))  # same timeline, 2x source spans
    assert sped_up.shot_duration_s == p.shot_duration_s
    with pytest.raises(StyleError, match="two shots"):
        profile_from_edl(_edl([3.0]))


def test_tables_name_every_dimension():
    a, b = _profile(0.6, 1.8, 5.2, 28.8), _profile(2.1, 3.4, 9.3, 9.0)
    text = format_diff("fast.json", a, "slow.json", b, profile_distance(a, b))
    for word in ("p10", "median", "p90", "cuts/min", "octaves", "fast.json", "slow.json"):
        assert word in text
    assert "not measured" in text  # the ratio dims
    shown = format_profile("fast.json", a)
    assert "cuts/min" in shown and "28.80" in shown and "beat" not in shown


def test_dissolve_distance_is_absolute_difference_and_has_a_diff_row():
    a = _profile(1, 2, 4, 20, transitions={"hard_cut": 0.8, "dissolve": 0.2})
    b = _profile(1, 2, 4, 20, transitions={"hard_cut": 0.3, "dissolve": 0.7})
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim["dissolve"] == pytest.approx(0.5)
        assert d.total == pytest.approx(0.5 / 5)
    text = format_diff("a", a, "b", b, profile_distance(a, b))
    row = next(line for line in text.splitlines() if line.startswith("dissolve"))
    assert row.split() == ["dissolve", "0.20", "0.70", "0.50"]


def test_dissolve_is_unmeasured_when_either_profile_is_unmeasured():
    a = _profile(1, 2, 4, 20, transitions={"hard_cut": 1, "dissolve": 0})
    b = _profile(1, 2, 4, 20)
    for left, right in [(a, b), (b, a), (b, b)]:
        d = profile_distance(left, right)
        assert "dissolve" not in d.per_dim
        assert "dissolve" in d.unmeasured


# ---------------------------------------------------------------------------
# round trip: reference -> profile -> cut of unrelated footage -> profile of the cut
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_round_trip_lands_near_the_reference(make_edited_clip, tmp_path):
    fast_ref = make_edited_clip(
        "fast.mp4", shot_lengths=[0.5, 1.0, 0.75, 0.5, 1.0, 0.75, 0.5, 1.0, 0.75, 0.5]
    )
    slow_ref = make_edited_clip("slow.mp4", shot_lengths=[2.0, 3.0, 4.0, 5.0])
    refs_ws = tmp_path / "refs"
    fast, slow = build_profile([fast_ref], refs_ws), build_profile([slow_ref], refs_ws)
    d_refs = profile_distance(fast, slow)
    assert d_refs.total > 1.5  # the two creators are octaves apart

    # unrelated footage: three recordings of four ~3 s shots (different lengths, so the
    # content fingerprints differ), nothing like either reference
    day = tmp_path / "day"
    day.mkdir()
    for i, lengths in enumerate([[3.0] * 4, [3.0, 3.5, 3.0, 2.5], [2.5, 3.0, 3.5, 3.0]]):
        make_edited_clip(f"IMG_{i}.mp4", shot_lengths=lengths).rename(day / f"IMG_{i}.mp4")
    fast_json, slow_json = tmp_path / "fast.json", tmp_path / "slow.json"
    fast_json.write_text(fast.model_dump_json(), encoding="utf-8")
    slow_json.write_text(slow.model_dump_json(), encoding="utf-8")
    ws = tmp_path / "ws"

    quick = cut(day, ws, target_s=9.0, style=fast_json, render=True, progress=lambda _m: None)
    d_edl = profile_distance(fast, profile_from_edl(quick.edl))
    assert d_edl.total < d_refs.total / 3, (d_edl, d_refs)
    assert d_edl.total < 0.1, d_edl
    # the path users take: profile the rendered preview with the real detector
    d_video = profile_distance(fast, build_profile([quick.preview_path], refs_ws))
    assert d_video.total < d_refs.total / 3, (d_video, d_refs)
    assert d_video.total < 0.2, d_video
    assert d_video.per_dim["cuts_per_min"] < 0.3  # the detector keeps our own cuts

    calm = cut(day, ws, target_s=9.0, style=slow_json, render=False, progress=lambda _m: None)
    measured = profile_from_edl(calm.edl)
    assert profile_distance(slow, measured).total < profile_distance(fast, measured).total
    assert len(calm.edl.clips) < len(quick.edl.clips)
