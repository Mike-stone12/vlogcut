from itertools import accumulate

import pytest

from vlogcut.plan import snap_to_beats
from vlogcut.plan.rules import RoughCutParams, rough_cut
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
from vlogcut.validate.checks import validate_edl


def _shots(count: int = 4) -> list[Shot]:
    return [
        Shot(
            shot_id=f"s{i}",
            asset_id="asset",
            start_s=10.0 * i,
            end_s=10.0 * (i + 1),
            quality=QualityFlags(sharpness=100, exposure_clip=0, shake=0),
        )
        for i in range(count)
    ]


def _edl(lengths: tuple[float, ...] = (2.0, 2.0, 2.0)) -> EDL:
    return EDL(
        variant_id="test",
        target_duration_s=sum(lengths),
        clips=[
            Clip(shot_id=f"s{i}", in_s=1.0, out_s=1.0 + length) for i, length in enumerate(lengths)
        ],
        notes="preserve these notes",
        music=[MusicCue(track_id="music", start_s=0.7)],
    )


def _ends(edl: EDL) -> list[float]:
    return list(accumulate(clip.timeline_duration_s for clip in edl.clips))


def test_snap_is_local_and_preserves_metadata_and_inputs() -> None:
    edl = _edl()
    edl.clips[1].transition_in = Transition(kind="dissolve", duration_s=0.5)
    edl.clips[0].audio = "mute"
    edl.clips[1].audio = "duck"
    edl.clips[1].reason = "keep speech"
    edl.clips[1].subtitle = "hello"
    before = edl.model_dump()
    shots = _shots()
    shot_data = [shot.model_dump() for shot in shots]
    beats = [8.0, 2.05, 0.0]
    result = snap_to_beats(edl, shots, beats)

    assert _ends(result) == pytest.approx([2.05, 4.0, 6.0], abs=1e-12)
    assert result.total_duration_s == pytest.approx(edl.total_duration_s, abs=1e-12)
    assert result.clips[0].out_s == pytest.approx(3.05)
    assert result.clips[1].in_s == pytest.approx(1.05)
    # Compare every field except the two edited offsets, including nested metadata.
    exclude = {"clips": {0: {"out_s"}, 1: {"in_s"}}}
    assert result.model_dump(exclude=exclude) == edl.model_dump(exclude=exclude)
    assert edl.model_dump() == before
    assert [shot.model_dump() for shot in shots] == shot_data
    assert beats == [8.0, 2.05, 0.0]
    assert result is not edl and result.clips[2] is not edl.clips[2]
    assert result.music[0] is not edl.music[0]


@pytest.mark.parametrize(
    ("beat", "expected"),
    [(2.080001, 2.0), (2.08, 2.08), (1.92, 1.92), (2.08 + 5e-13, 2.08 + 5e-13)],
)
def test_inclusive_tolerance(beat: float, expected: float) -> None:
    result = snap_to_beats(_edl(), _shots(), [beat])
    assert _ends(result) == pytest.approx([expected, 4.0, 6.0], abs=1e-12)


@pytest.mark.parametrize(
    ("beats", "expected"),
    [([2.07, 1.98, 2.04], 1.98), ([2.0625, 1.9375], 1.9375), ([2.05, 1.95], 1.95)],
)
def test_nearest_beat_and_earlier_tie(beats: list[float], expected: float) -> None:
    result = snap_to_beats(_edl(), _shots(), beats)
    assert _ends(result)[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("speed_index", [0, 1])
def test_speed_scales_source_shift(speed_index: int) -> None:
    edl = _edl()
    edl.clips[speed_index].speed = 2.0
    beat = _ends(edl)[0] + 0.05
    result = snap_to_beats(edl, _shots(), [beat])
    assert result.clips[0].out_s - edl.clips[0].out_s == pytest.approx(0.05 * edl.clips[0].speed)
    assert result.clips[1].in_s - edl.clips[1].in_s == pytest.approx(0.05 * edl.clips[1].speed)
    assert _ends(result) == pytest.approx([beat, *_ends(edl)[1:]], abs=1e-12)


@pytest.mark.parametrize("overflow", ["out", "in"])
def test_skips_shot_bound_overflow(overflow: str) -> None:
    edl = _edl()
    if overflow == "out":
        edl.clips[0].out_s = 10.0
        delta = 0.05
    else:
        edl.clips[1].in_s = 0.0
        delta = -0.05
    assert snap_to_beats(edl, _shots(), [_ends(edl)[0] + delta]) == edl


@pytest.mark.parametrize("index", [0, 1])
def test_skips_below_minimum_timeline_length(index: int) -> None:
    edl = _edl()
    edl.clips[index].speed = 2.0
    edl.clips[index].out_s = edl.clips[index].in_s + 0.5
    delta = -0.05 if index == 0 else 0.05
    assert snap_to_beats(edl, _shots(), [_ends(edl)[0] + delta]) == edl


def test_minimum_length_is_inclusive_and_configurable() -> None:
    edl = _edl((1.0, 0.5))
    assert snap_to_beats(edl, _shots(), [1.25], tolerance_s=0.25) != edl
    assert snap_to_beats(edl, _shots(), [1.25], tolerance_s=0.25, min_clip_s=0.3) == edl


def test_zero_tolerance_and_infeasible_nearest_beat() -> None:
    edl = _edl()
    assert snap_to_beats(edl, _shots(), [2.0], tolerance_s=0) == edl
    assert snap_to_beats(edl, _shots(), [2.01], tolerance_s=0) == edl
    edl.clips[0].out_s = 10.0
    # The nearer beat overflows; do not fall back to the feasible earlier beat.
    assert snap_to_beats(edl, _shots(), [8.94, 9.04]) == edl


@pytest.mark.parametrize(
    ("transition_index", "delta"), [(1, -0.05), (2, -0.05), (2, 0.05), (3, 0.05)]
)
def test_skips_dissolves_at_edited_and_adjacent_boundaries(
    transition_index: int, delta: float
) -> None:
    edl = _edl((2.0, 2.0, 2.0, 2.0))
    edl.clips[transition_index].transition_in = Transition(kind="dissolve", duration_s=1.98)
    assert validate_edl(edl, _shots()) == []
    assert snap_to_beats(edl, _shots(), [4.0 + delta]) == edl


def test_skips_overlapping_dissolve_half_windows() -> None:
    edl = _edl()
    for clip in edl.clips[1:]:
        clip.transition_in = Transition(kind="dissolve", duration_s=2.0)
    assert validate_edl(edl, _shots()) == []
    assert snap_to_beats(edl, _shots(), [2.05]) == edl


def test_unknown_shot_skips_both_sides_but_not_other_boundaries() -> None:
    edl = _edl((2.0, 2.0, 2.0, 2.0))
    shots = [shot for shot in _shots() if shot.shot_id != "s1"]
    result = snap_to_beats(edl, shots, [2.05, 4.05, 6.05])
    assert result.clips[:2] == edl.clips[:2]
    assert _ends(result) == pytest.approx([2.0, 4.0, 6.05, 8.0], abs=1e-12)


def test_consecutive_snaps_use_updated_clips() -> None:
    result = snap_to_beats(_edl(), _shots(), [2.04, 4.03])
    assert _ends(result) == pytest.approx([2.04, 4.03, 6.0], abs=1e-12)
    # The first snap leaves only 0.26 s in the middle clip; the second must skip.
    result = snap_to_beats(_edl((1.0, 0.3, 1.0)), _shots(), [1.04, 1.26])
    assert _ends(result) == pytest.approx([1.04, 1.3, 2.3], abs=1e-12)


@pytest.mark.parametrize(("lengths", "beats"), [((2.0, 2.0), []), ((2.0,), [2.05])])
def test_noop_returns_equal_new_edl(lengths: tuple[float, ...], beats: list[float]) -> None:
    edl = _edl(lengths)
    result = snap_to_beats(edl, _shots(), beats)
    assert result == edl and result is not edl


@pytest.mark.parametrize(("tolerance", "minimum"), [(-0.01, 0.25), (0.08, 0.0), (0.08, -0.1)])
def test_invalid_arguments_even_for_noop(tolerance: float, minimum: float) -> None:
    with pytest.raises(ValueError):
        snap_to_beats(_edl((2.0,)), _shots(), [], tolerance_s=tolerance, min_clip_s=minimum)


def test_rules_edl_stays_valid_after_snapping() -> None:
    shots = _shots()
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=2, median=2, p90=2),
        cuts_per_min=30,
        transitions=TransitionMix(hard_cut=0, dissolve=1),
    )
    edl = rough_cut(shots, [], RoughCutParams(target_s=8), profile)
    assert validate_edl(edl, shots, profile, target_s=8) == []
    beats = [boundary + 0.04 for boundary in _ends(edl)[:-1]]
    result = snap_to_beats(edl, shots, beats)
    assert _ends(result) == pytest.approx([*beats, edl.total_duration_s], abs=1e-12)
    assert validate_edl(result, shots, profile, target_s=8) == []
