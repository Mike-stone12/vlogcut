import pytest

from vlogcut.schemas import (
    EDL,
    Clip,
    DurationStats,
    MusicCue,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.validate import repair_edl, validate_edl


def _shots(n: int = 3) -> list[Shot]:
    return [
        Shot(
            shot_id=str(i),
            asset_id="a",
            start_s=100 + i * 10,
            end_s=110 + i * 10,
            quality=QualityFlags(sharpness=200, exposure_clip=0, shake=0),
        )
        for i in range(n)
    ]


def _edl(*lengths: float) -> EDL:
    return EDL(
        variant_id="test",
        target_duration_s=100,
        clips=[Clip(shot_id=str(i), in_s=0, out_s=length) for i, length in enumerate(lengths)],
        notes="Keep the requested target and music timing",
        music=[MusicCue(track_id="music", start_s=2, gain_db=-6)],
    )


def _profile(share: float | None = None) -> StyleProfile:
    return StyleProfile(
        shot_duration_s=DurationStats(p10=2, median=3, p90=4),
        cuts_per_min=20,
        transitions=None if share is None else TransitionMix(hard_cut=1 - share, dissolve=share),
    )


def _dissolve(edl: EDL, index: int, duration: float) -> None:
    edl.clips[index].transition_in = Transition(kind="dissolve", duration_s=duration)


def _assert_repaired(edl: EDL, shots: list[Shot], profile: StyleProfile | None = None) -> None:
    codes = {v.code for v in validate_edl(edl, shots, profile, target_s=edl.target_duration_s)}
    assert not codes & {
        "unknown_shot",
        "rejected_shot",
        "transition_not_allowed",
        "dissolve_too_long",  # Also used for overlapping half-windows.
    }
    assert edl.clips[0].transition_in == Transition()
    assert EDL.model_validate_json(edl.model_dump_json()) == edl


def test_valid_edl_round_trips_as_an_independent_copy() -> None:
    shots, edl, profile = _shots(), _edl(3, 3, 3), _profile(0.5)
    edl.target_duration_s = 9
    _dissolve(edl, 1, 1)
    assert validate_edl(edl, shots, profile, target_s=9) == []
    result = repair_edl(edl, shots, profile)
    assert result == edl and result is not edl
    assert result.clips[0] is not edl.clips[0]
    assert result.music[0] is not edl.music[0]
    assert validate_edl(result, shots, profile, target_s=9) == []
    _assert_repaired(result, shots, profile)


@pytest.mark.parametrize("rejected_first", [False, True])
def test_drop_unknown_and_rejected_preserves_retained_clips_and_metadata(
    rejected_first: bool,
) -> None:
    shots, edl = _shots(4), _edl(3, 5, 3, 7)
    unknown, rejected = (2, 0) if rejected_first else (0, 2)
    edl.clips[unknown].shot_id = "missing"
    shots[rejected].quality.rejected = True
    for i in (1, 3):
        edl.clips[i].in_s = 1
        edl.clips[i].speed = 2
        edl.clips[i].subtitle = "Retain this"
        edl.clips[i].reason = "Selected for speech"
        _dissolve(edl, i, 1)
    edl.clips[1].audio = "mute"
    edl.clips[3].audio = "duck"
    before = [item.model_dump_json() for item in [edl, *shots]]
    assert {v.code for v in validate_edl(edl, shots)} >= {"unknown_shot", "rejected_shot"}
    result = repair_edl(edl, shots)
    expected = edl.model_copy(deep=True)
    expected.clips = [expected.clips[1], expected.clips[3]]
    expected.clips[0].transition_in = Transition()
    assert result == expected
    assert result.total_duration_s == 5
    assert [item.model_dump_json() for item in [edl, *shots]] == before
    _assert_repaired(result, shots)


@pytest.mark.parametrize("share", [None, 0.0, 0.5])
def test_profile_mix_and_input_immutability(share: float | None) -> None:
    shots, edl, profile = _shots(), _edl(3, 3, 3), _profile(share)
    for i in (1, 2):
        _dissolve(edl, i, 1)
    before = [item.model_dump_json() for item in [edl, *shots, profile]]
    result = repair_edl(edl, shots, profile)
    if share == 0:
        assert all(c.transition_in == Transition() for c in result.clips)
    else:
        assert result == edl
    assert [item.model_dump_json() for item in [edl, *shots, profile]] == before
    _assert_repaired(result, shots, profile)


def test_none_profile_keeps_dissolves() -> None:
    shots, edl = _shots(), _edl(3, 3, 3)
    _dissolve(edl, 1, 1)
    result = repair_edl(edl, shots)
    assert result == edl
    _assert_repaired(result, shots)


@pytest.mark.parametrize("short_index", [0, 1])
def test_oversized_dissolve_fits_both_speed_adjusted_neighbours(short_index: int) -> None:
    shots, edl = _shots(), _edl(4, 4)
    edl.clips[short_index].speed = 4
    _dissolve(edl, 1, 2)
    before = edl.model_dump_json()
    assert [v.code for v in validate_edl(edl, shots)] == ["dissolve_too_long"]
    result = repair_edl(edl, shots)
    assert result.clips[1].transition_in == Transition(kind="dissolve", duration_s=1)
    assert edl.model_dump_json() == before
    _assert_repaired(result, shots)


def test_overlapping_windows_are_shrunk_to_touch_and_repair_is_idempotent() -> None:
    shots, edl = _shots(), _edl(4, 1, 4)
    for i in (1, 2):
        _dissolve(edl, i, 1.5)
    assert any("halves" in v.message for v in validate_edl(edl, shots))
    result = repair_edl(edl, shots)
    assert [c.transition_in.duration_s for c in result.clips] == [0, 1, 1]
    assert repair_edl(result, shots) == result
    _assert_repaired(result, shots)


def test_no_positive_timeline_space_converts_dissolve_to_cut() -> None:
    # Valid positive source duration can underflow to zero at a higher speed.
    # Ordinary positive timeline durations always permit some dissolve duration.
    shots, edl = _shots(), _edl(float.fromhex("0x0.0000000000001p-1022"), 3)
    edl.clips[0].speed = 4
    assert edl.clips[0].timeline_duration_s == 0
    _dissolve(edl, 1, 1)
    result = repair_edl(edl, shots)
    assert result.clips[1].transition_in == Transition()
    assert result.clips[0] == edl.clips[0]
    _assert_repaired(result, shots)


def test_unrepaired_violations_and_selections_remain() -> None:
    shots, edl, profile = _shots(), _edl(1, 12, 3), _profile()
    for shot in shots:
        shot.dup_group = "same"
    edl.clips[0].shot_id = "2"
    edl.clips[1].shot_id = "0"
    edl.clips[2].shot_id = "0"
    violations = validate_edl(edl, shots, profile, target_s=100)
    assert {v.code for v in violations} == {
        "out_of_bounds",
        "not_chronological",
        "duplicate_neighbours",
        "clip_too_short",
        "clip_too_long",
        "total_duration",
    }
    result = repair_edl(edl, shots, profile)
    assert result == edl
    assert validate_edl(result, shots, profile, target_s=100) == violations
    _assert_repaired(result, shots, profile)


@pytest.mark.parametrize("unknown", [False, True])
def test_dropping_everything_raises_without_mutation(unknown: bool) -> None:
    shots, edl = _shots(1), _edl(3)
    if unknown:
        shots = []
    else:
        shots[0].quality.rejected = True
    before = [item.model_dump_json() for item in [edl, *shots]]
    with pytest.raises(ValueError, match="no known, unrejected clips remain"):
        repair_edl(edl, shots)
    assert [item.model_dump_json() for item in [edl, *shots]] == before
