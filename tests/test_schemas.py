from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlogcut.schemas import (
    EDL,
    Clip,
    CutPoint,
    DurationStats,
    Feedback,
    LookStats,
    PreferencePair,
    QualityFlags,
    Shot,
    StructureStats,
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


def test_old_shot_json_defaults_to_independent_empty_cut_points() -> None:
    payload = (
        '{"shot_id":"s", "asset_id":"a", "start_s":10, "end_s":14,'
        '"quality":{"sharpness":50, "exposure_clip":0, "shake":0}}'
    )
    first, second = Shot.model_validate_json(payload), Shot.model_validate_json(payload)
    assert first.cut_points == second.cut_points == []
    assert first.cut_points is not second.cut_points


def test_shot_cut_points_round_trip_independently_of_usable_range() -> None:
    points = [CutPoint(time_s=t, kind="pause") for t in [10.5, 12, 13.5]]
    shot = Shot(
        shot_id="s",
        asset_id="a",
        start_s=10,
        end_s=14,
        quality=QualityFlags(sharpness=50, exposure_clip=0, shake=0, usable_range=(1, 3)),
        cut_points=points,
    )
    assert shot.cut_points == points
    assert Shot.model_validate_json(shot.model_dump_json()) == shot


@pytest.mark.parametrize(
    "times",
    [[12, 11], [11, 11], [10], [14], [9.9], [14.1]],
    ids=["unsorted", "duplicate", "start", "end", "before", "after"],
)
def test_shot_rejects_invalid_cut_points(times: list[float]) -> None:
    with pytest.raises(ValidationError, match="cut_points"):
        Shot(
            shot_id="s",
            asset_id="a",
            start_s=10,
            end_s=14,
            quality=QualityFlags(sharpness=50, exposure_clip=0, shake=0),
            cut_points=[CutPoint(time_s=t, kind="pause") for t in times],
        )


def test_cut_point_rejects_other_kinds() -> None:
    with pytest.raises(ValidationError, match="kind"):
        CutPoint.model_validate({"time_s": 12, "kind": "motion"})


def test_clip_speed_changes_timeline_duration():
    clip = Clip(shot_id="s1", in_s=0, out_s=4, speed=2.0)
    assert clip.timeline_duration_s == pytest.approx(2.0)


def test_clip_audio_defaults_and_old_json():
    assert Clip(shot_id="s", in_s=0, out_s=2).audio == "keep"
    assert Clip.model_validate_json('{"shot_id":"s","in_s":0,"out_s":2}').audio == "keep"


@pytest.mark.parametrize("mode", ["keep", "mute", "duck"])
def test_clip_audio_round_trips(mode: str):
    clip = Clip.model_validate({"shot_id": "s", "in_s": 0, "out_s": 2, "audio": mode})
    assert Clip.model_validate_json(clip.model_dump_json()).audio == mode


@pytest.mark.parametrize("mode", ["silence", "KEEP", "", None, 0])
def test_clip_audio_rejects_invalid_modes(mode: object):
    with pytest.raises(ValidationError, match="audio"):
        Clip.model_validate({"shot_id": "s", "in_s": 0, "out_s": 2, "audio": mode})


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
    assert all(c.audio == "keep" for c in edl.clips)
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


def test_style_profile_cold_open_default_and_old_json() -> None:
    profile = StyleProfile(shot_duration_s=DurationStats(p10=1, median=2, p90=4), cuts_per_min=20)
    assert profile.cold_open_clips == 0
    old = StyleProfile.model_validate_json(
        '{"shot_duration_s": {"p10": 1, "median": 2, "p90": 4}, "cuts_per_min": 20}'
    )
    assert old.cold_open_clips == 0


def test_style_profile_cold_open_rejects_negative() -> None:
    with pytest.raises(ValidationError, match="cold_open_clips"):
        StyleProfile(
            shot_duration_s=DurationStats(p10=1, median=2, p90=4),
            cuts_per_min=20,
            cold_open_clips=-1,
        )


def test_style_profile_cold_open_round_trip() -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=2, p90=4),
        cuts_per_min=20,
        cold_open_clips=2,
    )
    restored = StyleProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert restored.cold_open_clips == 2


def test_style_profile_unmeasured_ratios_are_none():
    stats = DurationStats(p10=1.0, median=2.0, p90=4.0)
    profile = StyleProfile(shot_duration_s=stats, cuts_per_min=20)
    assert profile.beat_aligned_ratio is None and profile.speech_ratio is None
    assert profile.look is None
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
    assert profile.structure is None
    assert profile.look is None


@pytest.mark.parametrize(
    "value", [None, {}, {"brightness": 0, "saturation": 0, "warmth": 0, "frames": 1}]
)
def test_profile_look_round_trip_and_legacy_empty_dict(value: object) -> None:
    profile = StyleProfile.model_validate(
        {"shot_duration_s": {"p10": 1, "median": 2, "p90": 4}, "cuts_per_min": 20, "look": value}
    )
    assert (profile.look is None) == (value is None or value == {})
    assert StyleProfile.model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.parametrize(
    "field,value",
    [
        ("brightness", -0.01),
        ("brightness", 1.01),
        ("saturation", -0.01),
        ("saturation", 1.01),
        ("warmth", -1.01),
        ("warmth", 1.01),
        ("frames", 0),
        ("brightness", float("nan")),
        ("warmth", float("inf")),
    ],
)
def test_look_stats_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        LookStats.model_validate(
            {"brightness": 0, "saturation": 0, "warmth": 0, "frames": 1, field: value}
        )


@pytest.mark.parametrize("brightness,saturation,warmth", [(0, 0, -1), (1, 1, 1)])
def test_look_stats_inclusive_bounds(brightness: float, saturation: float, warmth: float) -> None:
    assert LookStats(brightness=brightness, saturation=saturation, warmth=warmth, frames=1)


@pytest.mark.parametrize("share", [-0.1, 1.1])
def test_structure_stats_rejects_invalid_share(share: float) -> None:
    with pytest.raises(ValidationError, match="montage_share"):
        StructureStats(montage_share=share)


def test_old_structure_json_has_unmeasured_intro_pace() -> None:
    profile = StyleProfile.model_validate_json(
        '{"shot_duration_s": {"p10": 1, "median": 2, "p90": 4}, '
        '"cuts_per_min": 20, "structure": {"montage_share": 0.5}}'
    )
    assert profile.structure is not None
    assert profile.structure.intro_pace_ratio is None


@pytest.mark.parametrize("ratio", [0, -0.5])
def test_structure_rejects_non_positive_intro_pace(ratio: float) -> None:
    with pytest.raises(ValidationError, match="intro_pace_ratio"):
        StructureStats(montage_share=0, intro_pace_ratio=ratio)


@pytest.mark.parametrize("ratio", [None, 0.5, 1, 2])
def test_structure_intro_pace_round_trip(ratio: float | None) -> None:
    structure = StructureStats(montage_share=0.5, intro_pace_ratio=ratio)
    assert StructureStats.model_validate_json(structure.model_dump_json()) == structure


@pytest.mark.parametrize("share", [0, 0.375, 1])
def test_style_structure_round_trip(share: float) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=2, p90=4),
        cuts_per_min=20,
        structure=StructureStats(montage_share=share),
    )
    assert StyleProfile.model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.parametrize("rate", [0, -1])
def test_style_profile_requires_positive_cut_rate(rate):
    stats = DurationStats(p10=60, median=60, p90=60)
    with pytest.raises(ValidationError):
        StyleProfile(shot_duration_s=stats, cuts_per_min=rate)


@pytest.mark.parametrize(
    "fields",
    [
        {"verdict": "pick"},
        {"verdict": "reject"},
        {"text": "  Keep more of the cafe  "},
        {"verdict": "pick", "text": "Good pacing"},
    ],
)
def test_feedback_valid_and_round_trips(fields: dict[str, str]) -> None:
    feedback = Feedback.model_validate({"variant_id": "coverage_180s_fast", **fields})
    assert feedback.verdict == fields.get("verdict")
    assert feedback.text == fields.get("text", "")
    assert Feedback.model_validate_json(feedback.model_dump_json()) == feedback


@pytest.mark.parametrize("fields", [{}, {"text": ""}, {"text": " \t\n"}])
def test_feedback_requires_verdict_or_text(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="verdict or non-blank text"):
        Feedback.model_validate({"variant_id": "v", **fields})


@pytest.mark.parametrize("variant_id", ["", " \t\n"])
def test_feedback_rejects_blank_variant_id(variant_id: str) -> None:
    with pytest.raises(ValidationError, match="variant_id"):
        Feedback(variant_id=variant_id, verdict="pick")


def test_feedback_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError, match="verdict"):
        Feedback.model_validate({"variant_id": "v", "verdict": "maybe"})


def test_preference_pair_rejects_same_variant() -> None:
    with pytest.raises(ValidationError, match="different variants"):
        PreferencePair(context="", chosen="v", rejected="v")


@pytest.mark.parametrize("field", ["chosen", "rejected"])
@pytest.mark.parametrize("variant_id", ["", " \t\n"])
def test_preference_pair_rejects_blank_ids(field: str, variant_id: str) -> None:
    fields = {"context": "", "chosen": "a", "rejected": "b", field: variant_id}
    with pytest.raises(ValidationError, match="must not be blank"):
        PreferencePair.model_validate(fields)


@pytest.mark.parametrize("context", ["", "A day at the cafe"])
def test_preference_pair_round_trips(context: str) -> None:
    pair = PreferencePair(context=context, chosen="fast", rejected="slow")
    assert PreferencePair.model_validate_json(pair.model_dump_json()) == pair
