import pytest

from vlogcut.plan.rules import RoughCutParams, rough_cut
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
from vlogcut.validate import Violation, validate_edl


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
    )


def _profile(share: float | None = None) -> StyleProfile:
    return StyleProfile(
        shot_duration_s=DurationStats(p10=2, median=3, p90=4),
        cuts_per_min=20,
        transitions=None if share is None else TransitionMix(hard_cut=1 - share, dissolve=share),
    )


def _dissolve(edl: EDL, index: int, duration: float) -> None:
    edl.clips[index].transition_in = Transition(kind="dissolve", duration_s=duration)


def test_clean_is_deterministic_and_does_not_mutate_inputs() -> None:
    shots, edl, profile = _shots(), _edl(3, 3, 3), _profile(0.5)
    shots[0].quality.usable_range = (4, 6)  # Suggestion, not a hard constraint.
    for clip, mode in zip(edl.clips, ("keep", "mute", "duck"), strict=True):
        clip.audio = mode
    _dissolve(edl, 1, 1)
    before = [item.model_dump_json() for item in [*shots, edl, profile]]
    for _ in range(2):
        assert validate_edl(edl, shots, profile, target_s=9) == []
    assert [item.model_dump_json() for item in [*shots, edl, profile]] == before


def test_unknown_shot_gets_only_one_violation_and_total_is_still_checked() -> None:
    edl = _edl(3, 100, 3)
    edl.clips[1].shot_id = "missing\nshot"
    _dissolve(edl, 1, 200)
    violations = validate_edl(edl, _shots(), _profile(0), target_s=9)
    unknown = [v for v in violations if v.clip_index == 1]
    assert len(unknown) == 1 and unknown[0].code == "unknown_shot"
    assert "missing" in unknown[0].message and "\n" not in unknown[0].message
    assert violations[-1].code == "total_duration"
    assert validate_edl(_edl(3), []) == [
        Violation(code="unknown_shot", clip_index=0, message="Shot '0' is absent from shots")
    ]


@pytest.mark.parametrize(
    "end,expected", [(10, []), (10 + 0.5e-6, []), (10 + 2e-6, ["out_of_bounds"])]
)
def test_shot_relative_bounds_tolerance(end: float, expected: list[str]) -> None:
    assert [v.code for v in validate_edl(_edl(end), _shots())] == expected


def test_start_beyond_shot_and_rejected_shot() -> None:
    shots, edl = _shots(), _edl(13)
    edl.clips[0].in_s = 11
    shots[0].quality.rejected = True
    assert [v.code for v in validate_edl(edl, shots)] == ["out_of_bounds", "rejected_shot"]


@pytest.mark.parametrize(
    "length,expected", [(8.9, ["total_duration"]), (9, []), (11, []), (11.1, ["total_duration"])]
)
def test_target_tolerance(length: float, expected: list[str]) -> None:
    edl = _edl(length * 0.5)
    edl.clips[0].speed = 0.5  # Timeline, not source duration.
    violations = validate_edl(edl, _shots(), target_s=10)
    assert [v.code for v in violations] == expected
    assert all(v.clip_index is None for v in violations)


@pytest.mark.parametrize(
    "length,expected", [(1.49, ["clip_too_short"]), (1.5, []), (5, []), (5.01, ["clip_too_long"])]
)
def test_profile_length_tolerance(length: float, expected: list[str]) -> None:
    edl = _edl(length * 0.5)
    edl.clips[0].speed = 0.5
    assert [v.code for v in validate_edl(edl, _shots(), _profile())] == expected


def test_none_skips_target_and_profile_and_custom_tolerances_apply() -> None:
    edl, shots = _edl(1, 6), _shots()
    _dissolve(edl, 1, 0.5)
    assert validate_edl(edl, shots, profile=None, target_s=None) == []
    assert (
        validate_edl(
            edl, shots, _profile(), target_s=10, duration_tolerance=0.3, length_tolerance=0.5
        )
        == []
    )
    assert [v.code for v in validate_edl(_edl(3), shots, target_s=4, duration_tolerance=0)] == [
        "total_duration"
    ]
    assert [v.code for v in validate_edl(_edl(1.9), shots, _profile(), length_tolerance=0)] == [
        "clip_too_short"
    ]


def test_duplicate_neighbours_only_non_none_adjacent_groups() -> None:
    shots, edl = _shots(), _edl(3, 3, 3)
    shots[0].dup_group = shots[2].dup_group = ""  # Empty string is still non-None.
    assert validate_edl(edl, shots) == []
    shots[1].dup_group = ""
    assert [(v.code, v.clip_index) for v in validate_edl(edl, shots)] == [
        ("duplicate_neighbours", 1),
        ("duplicate_neighbours", 2),
    ]
    edl.clips[1].shot_id = "missing"
    assert [v.code for v in validate_edl(edl, shots)] == ["unknown_shot"]


@pytest.mark.parametrize("ids", [("1", "0"), ("0", "0"), ("1", "missing", "0")])
def test_chronology_rejects_backwards_repeated_and_cold_open(ids: tuple[str, ...]) -> None:
    edl = _edl(*([3] * len(ids)))
    for clip, shot_id in zip(edl.clips, ids, strict=True):
        clip.shot_id = shot_id
    violations = validate_edl(edl, _shots())
    assert violations[-1].code == "not_chronological"
    assert violations[-1].clip_index == len(ids) - 1


def test_chronology_uses_supplied_order_not_timestamps() -> None:
    shots, edl = _shots(2)[::-1], _edl(3, 3)
    edl.clips.reverse()
    assert validate_edl(edl, shots) == []


def test_cold_open_allowance_preserves_default_violation() -> None:
    edl, shots, profile = _edl(3, 3, 3), _shots(), _profile()
    for clip, shot_id in zip(edl.clips, ("2", "0", "1"), strict=True):
        clip.shot_id = shot_id
    expected = [
        Violation(
            code="not_chronological",
            clip_index=1,
            message="Shot '0' position 0 is not after previous known position 2",
        )
    ]
    assert validate_edl(edl, shots, None) == expected
    assert validate_edl(edl, shots, profile) == expected
    profile.cold_open_clips = 1
    assert validate_edl(edl, shots, profile) == []


@pytest.mark.parametrize(
    "allowance,ids",
    [
        (1, ("2", "0", "1", "2")),  # The hook can be used again later.
        (1, ("2", "2")),  # Even immediately after the hook.
        (2, ("2", "1", "0", "1", "2")),
        (2, ("1", "2", "0", "1", "2")),
        (2, ("2", "2", "0", "1")),
        (10, ("2", "1", "0", "0")),
    ],
)
def test_cold_open_exempts_leading_clips(allowance: int, ids: tuple[str, ...]) -> None:
    edl, profile = _edl(*([3] * len(ids))), _profile()
    profile.cold_open_clips = allowance
    for clip, shot_id in zip(edl.clips, ids, strict=True):
        clip.shot_id = shot_id
    assert validate_edl(edl, _shots(), profile) == []


@pytest.mark.parametrize(
    "ids,expected",
    [
        (("2", "1", "0"), [("not_chronological", 2)]),
        (("2", "0", "0"), [("not_chronological", 2)]),
        (("2", "1", "missing", "0"), [("unknown_shot", 2), ("not_chronological", 3)]),
        (("missing", "1", "0"), [("unknown_shot", 0), ("not_chronological", 2)]),
        (("2", "missing", "0"), [("unknown_shot", 1)]),
    ],
)
def test_chronology_after_cold_open(ids: tuple[str, ...], expected: list[tuple[str, int]]) -> None:
    edl, profile = _edl(*([3] * len(ids))), _profile()
    profile.cold_open_clips = 1
    for clip, shot_id in zip(edl.clips, ids, strict=True):
        clip.shot_id = shot_id
    violations = validate_edl(edl, _shots(), profile)
    assert [(v.code, v.clip_index) for v in violations] == expected


def test_cold_open_chronology_uses_supplied_order() -> None:
    edl, profile = _edl(3, 3, 3, 3), _profile()
    profile.cold_open_clips = 1
    for clip, shot_id in zip(edl.clips, ("0", "2", "1", "0"), strict=True):
        clip.shot_id = shot_id
    assert validate_edl(edl, _shots()[::-1], profile) == []


@pytest.mark.parametrize("allowance", [1, 2, 10])
def test_cold_open_preserves_all_other_checks(allowance: int) -> None:
    edl, shots, profile = _edl(11, 0.5, 3, 3), _shots(), _profile(0)
    for clip, shot_id in zip(edl.clips, ("2", "0", "missing", "1"), strict=True):
        clip.shot_id = shot_id
    shots[2].quality.rejected = True
    shots[0].dup_group = shots[2].dup_group = "same"
    _dissolve(edl, 1, 12)
    _dissolve(edl, 3, 4)
    expected = [
        v for v in validate_edl(edl, shots, profile, target_s=100) if v.code != "not_chronological"
    ]
    assert {v.code for v in expected} == {
        "unknown_shot",
        "out_of_bounds",
        "rejected_shot",
        "clip_too_short",
        "clip_too_long",
        "duplicate_neighbours",
        "transition_not_allowed",
        "dissolve_too_long",
        "total_duration",
    }
    profile.cold_open_clips = allowance
    assert validate_edl(edl, shots, profile, target_s=100) == expected


@pytest.mark.parametrize("share,expected", [(None, []), (0.5, []), (0, ["transition_not_allowed"])])
def test_transition_mix(share: float | None, expected: list[str]) -> None:
    edl = _edl(3, 3)
    _dissolve(edl, 1, 1)
    assert [v.code for v in validate_edl(edl, _shots(), _profile(share))] == expected


@pytest.mark.parametrize("short_index", [0, 1])
def test_dissolve_must_fit_each_neighbours_timeline(short_index: int) -> None:
    edl = _edl(4, 4)
    edl.clips[short_index].speed = 4  # A 1 s timeline clip despite 4 s of source.
    _dissolve(edl, 1, 1.5)
    violations = validate_edl(edl, _shots())
    assert [(v.code, v.clip_index) for v in violations] == [("dissolve_too_long", 1)]


def test_adjacent_dissolve_windows_overlap_and_touching_is_allowed() -> None:
    edl = _edl(4, 1, 4)
    for i in (1, 2):
        _dissolve(edl, i, 1)
    assert validate_edl(edl, _shots()) == []  # Exact fit, halves touch.
    for i in (1, 2):
        _dissolve(edl, i, 1.5)
    violations = validate_edl(edl, _shots())
    assert [(v.code, v.clip_index) for v in violations] == [
        ("dissolve_too_long", 1),
        ("dissolve_too_long", 1),
        ("dissolve_too_long", 2),
    ]
    assert "halves 0.75 s + 0.75 s" in violations[1].message


def test_violations_order_and_serializable_messages() -> None:
    shots, edl = _shots(), _edl(0.5, 11, 3)
    shots[0].quality.rejected = True
    edl.clips[2].shot_id = "missing"
    violations = validate_edl(edl, shots, _profile(), target_s=100)
    assert [v.clip_index for v in violations] == [0, 0, 1, 1, 2, None]
    for violation in violations:
        assert Violation.model_validate_json(violation.model_dump_json()) == violation
        assert violation.message and "\n" not in violation.message


@pytest.mark.parametrize("styled", [False, True])
def test_rules_rough_cut_passes(styled: bool) -> None:
    shots = _shots(20)  # Chronological, long, unrejected, no duplicate groups.
    profile = _profile(0.5) if styled else None
    edl = rough_cut(shots, [], RoughCutParams(target_s=30), profile=profile)
    assert validate_edl(edl, shots, profile, target_s=30) == []
    if styled:
        assert any(clip.transition_in.kind == "dissolve" for clip in edl.clips)
