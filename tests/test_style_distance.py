from __future__ import annotations

from typing import Literal

import pytest

from vlogcut.pipeline import cut
from vlogcut.schemas import (
    EDL,
    Clip,
    DurationStats,
    LookStats,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.style import (
    ReferenceParams,
    StyleError,
    build_profile,
    profile_distance,
    profile_from_edl,
)
from vlogcut.style.distance import format_diff, format_profile, speech_ratio


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
    assert d.unmeasured == [
        "intro_pace",
        "beat_aligned_ratio",
        "speech_ratio",
        "dissolve",
        "montage_share",
        "kind_alternation",
        "brightness",
        "saturation",
        "warmth",
    ]


@pytest.mark.parametrize(("a_bpm", "b_bpm", "octaves"), [(120, 60, 1), (60, 120, 1), (120, 120, 0)])
@pytest.mark.parametrize("beat_ratio", [None, 0.5])
def test_music_bpm_distance_and_diff_row(
    a_bpm: float, b_bpm: float, octaves: float, beat_ratio: float | None
) -> None:
    a = _profile(1, 2, 4, 20, music_bpm=a_bpm, beat_aligned_ratio=beat_ratio)
    b = _profile(1, 2, 4, 20, music_bpm=b_bpm, beat_aligned_ratio=beat_ratio)
    d = profile_distance(a, b)
    assert d.per_dim["music_bpm"] == octaves
    assert "music_bpm" not in d.unmeasured
    assert d.total == pytest.approx(octaves / (5 if beat_ratio is None else 6))
    assert profile_distance(b, a) == d
    text = format_diff("a", a, "b", b, d)
    row = next(line for line in text.splitlines() if line.startswith("music_bpm"))
    assert row.split() == ["music_bpm", f"{a_bpm:.1f}", f"{b_bpm:.1f}", f"{octaves:.2f}"]
    if beat_ratio is not None:
        assert "octaves (pacing, tempo) and ratio points, mixed" in text


def test_music_bpm_skips_missing_on_either_side() -> None:
    a = _profile(1, 2, 4, 20, beat_aligned_ratio=0.8)
    b = _profile(0.5, 1, 2, 40, beat_aligned_ratio=0.3)
    baseline = profile_distance(a, b)
    a.music_bpm = 120
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.total == baseline.total
        assert d.per_dim == baseline.per_dim
        assert d.unmeasured == ["intro_pace", "music_bpm", *baseline.unmeasured[1:]]
        lines = format_diff("a", left, "b", right, d).splitlines()
        assert not any(line.startswith("music_bpm") for line in lines)
        assert "music_bpm" in lines[-1]


def test_music_bpm_absent_on_both_sides_preserves_distance_and_diff() -> None:
    a = _profile(1, 2, 4, 20)
    b = _profile(0.5, 1, 2, 40)
    d = profile_distance(a, b)
    # Exact distance and text before music_bpm became a distance dimension.
    assert d.model_dump() == {
        "total": 1.0,
        "per_dim": {"p10": 1.0, "median": 1.0, "p90": 1.0, "cuts_per_min": 1.0},
        "unmeasured": [
            "intro_pace", "beat_aligned_ratio", "speech_ratio", "dissolve", "montage_share",
            "kind_alternation", "brightness", "saturation", "warmth",
        ],
    }  # fmt: skip
    assert format_diff("a", a, "b", b, d) == (
        "                      a          b     delta\n"
        "p10 shot s         1.00       0.50      1.00\n"
        "median shot s      2.00       1.00      1.00\n"
        "p90 shot s         4.00       2.00      1.00\n"
        "cuts/min          20.00      40.00      1.00\n"
        "distance      1.00 octaves; mean over 4 dims\n"
        "not measured on both sides: intro_pace, beat_aligned_ratio, speech_ratio, "
        "dissolve, montage_share, kind_alternation, brightness, saturation, warmth"
    )


def test_ratio_dims_count_only_when_measured_on_both_sides():
    a = _profile(1.0, 2.0, 4.0, 20.0, beat_aligned_ratio=0.8, speech_ratio=0.5)
    b = _profile(1.0, 2.0, 4.0, 20.0, beat_aligned_ratio=0.3)
    d = profile_distance(a, b)
    assert d.per_dim["beat_aligned_ratio"] == pytest.approx(0.5)
    assert d.unmeasured == [
        "intro_pace", "speech_ratio", "dissolve", "montage_share", "kind_alternation",
        "brightness", "saturation", "warmth",
    ]  # fmt: skip
    assert d.total == pytest.approx(0.5 / 5)  # four pacing dims at 0 plus one ratio dim


@pytest.mark.parametrize(("a_value", "b_value"), [(0.2, 0.7), (0.0, 0.0), (0.0, 1.0)])
def test_speech_ratio_distance(a_value: float, b_value: float) -> None:
    a = _profile(1, 2, 4, 20, speech_ratio=a_value)
    b = _profile(1, 2, 4, 20, speech_ratio=b_value)
    for left, right in [(a, b), (b, a)]:
        distance = profile_distance(left, right)
        assert distance.per_dim["speech_ratio"] == pytest.approx(abs(a_value - b_value))
        assert "speech_ratio" not in distance.unmeasured
        assert distance.total == pytest.approx(abs(a_value - b_value) / 5)


def test_speech_ratio_distance_skips_unmeasured_on_either_side() -> None:
    a = _profile(1, 2, 4, 20, speech_ratio=0.0)
    b = _profile(1, 2, 4, 20)
    for left, right in [(a, b), (b, a), (b, b)]:
        distance = profile_distance(left, right)
        assert "speech_ratio" not in distance.per_dim
        assert "speech_ratio" in distance.unmeasured
        assert distance.total == 0
        lines = format_diff("a", left, "b", right, distance).splitlines()
        assert not any(line.startswith("speech ratio") for line in lines)
        assert "speech_ratio" in lines[-1]
    assert "speech ratio" not in format_profile("b", b)


def test_speech_ratio_weights_speed_adjusted_durations_and_excludes_unknowns() -> None:
    clips = [
        Clip(shot_id="talk", in_s=1, out_s=9, speed=2),  # 4 s
        Clip(shot_id="broll", in_s=0, out_s=3, speed=0.5),  # 6 s
        Clip(shot_id="talk", in_s=2, out_s=4),  # 2 s, repeated shot counts again
        Clip(shot_id="unknown", in_s=0, out_s=100),
        Clip(shot_id="missing", in_s=0, out_s=100),
    ]
    assert speech_ratio(clips, {"talk": "talking", "broll": "broll", "unknown": None}) == 0.5


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        ({"s0": "broll", "s1": "broll"}, 0.0),
        ({"s0": "talking", "s1": "talking"}, 1.0),
        ({"s0": None, "s1": None}, None),
        ({}, None),
    ],
)
def test_speech_ratio_known_and_unmeasured_kinds(
    kinds: dict[str, Literal["talking", "broll"] | None], expected: float | None
) -> None:
    assert speech_ratio(_edl([2, 4]).clips, kinds) == expected
    assert speech_ratio([], kinds) is None


def test_identical_looks_have_zero_distance() -> None:
    a = _profile(1, 2, 4, 20, look=dict(brightness=0.3, saturation=0.4, warmth=-0.2, frames=3))
    d = profile_distance(a, a)
    for dim in ("brightness", "saturation", "warmth"):
        assert d.per_dim[dim] == 0
        assert dim not in d.unmeasured
    assert d.total == 0


@pytest.mark.parametrize(("warm_a", "warm_b", "delta"), [(-0.2, 0.2, 0.2), (-1, 1, 1)])
def test_look_distance_known_values_and_diff_rows(
    warm_a: float,
    warm_b: float,
    delta: float,
) -> None:
    a = _profile(1, 2, 4, 20, look=dict(brightness=0.3, saturation=0.4, warmth=warm_a, frames=3))
    b = _profile(0.5, 1, 2, 40, look=dict(brightness=0.8, saturation=0.1, warmth=warm_b, frames=9))
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim == pytest.approx(
            {
                "p10": 1,
                "median": 1,
                "p90": 1,
                "cuts_per_min": 1,
                "brightness": 0.5,
                "saturation": 0.3,
                "warmth": delta,
            }
        )
        assert d.total == pytest.approx((4 + 0.5 + 0.3 + delta) / 7)
    text = format_diff("a", a, "b", b, profile_distance(a, b))
    for expected in [
        ["brightness", "0.30", "0.80", "0.50"],
        ["saturation", "0.40", "0.10", "0.30"],
        ["warmth", f"{warm_a:.2f}", f"{warm_b:.2f}", f"{delta:.2f}"],
    ]:
        row = next(line for line in text.splitlines() if line.startswith(expected[0]))
        assert row.split() == expected
    assert "mixed; mean over 7 dims" in text


@pytest.mark.parametrize("beat_ratio", [None, 0.8])
def test_look_distance_skips_missing_look_on_either_side(beat_ratio: float | None) -> None:
    a = _profile(1, 2, 4, 20, beat_aligned_ratio=beat_ratio)
    b = _profile(0.5, 1, 2, 40, beat_aligned_ratio=0.3)
    baseline = profile_distance(a, b)
    measured = a.model_copy(
        update={
            "look": LookStats(brightness=0.3, saturation=0.4, warmth=-0.2, frames=3),
        }
    )
    for left, right in [(measured, b), (b, measured), (a, b)]:
        d = profile_distance(left, right)
        assert d.per_dim == baseline.per_dim
        assert d.total == baseline.total
        for dim in ("brightness", "saturation", "warmth"):
            assert dim in d.unmeasured
            assert dim not in d.per_dim


def test_profile_from_edl_uses_timeline_durations():
    p = profile_from_edl(_edl([1.0, 2.0, 1.0]))
    assert p.cuts_per_min == pytest.approx(30.0)  # 2 cuts in 4 s
    assert p.shot_duration_s.median == pytest.approx(1.0)
    assert p.source_refs == ["edl"]
    assert p.look is None
    assert p.music_bpm is None
    assert p.energy_curve is None
    reference = p.model_copy(update={"music_bpm": 120})
    distance = profile_distance(reference, p)
    assert "music_bpm" in distance.unmeasured
    assert distance.per_dim == profile_distance(p, p).per_dim
    assert distance.total == 0
    assert p.transitions == TransitionMix(hard_cut=1, dissolve=0)
    sped_up = profile_from_edl(_edl([1.0, 2.0, 1.0], speed=2.0))  # same timeline, 2x source spans
    assert sped_up.shot_duration_s == p.shot_duration_s
    with pytest.raises(StyleError, match="two shots"):
        profile_from_edl(_edl([3.0]))


def test_profile_from_edl_measures_dissolves_over_boundaries():
    edl = _edl([1, 2, 1])
    edl.clips[1].transition_in = Transition(kind="dissolve", duration_s=0.5)
    profile = profile_from_edl(edl)
    assert profile.transitions == TransitionMix(hard_cut=0.5, dissolve=0.5)
    assert profile.cuts_per_min == 30


def test_profile_from_edl_without_shots_preserves_previous_json_bytes() -> None:
    edl = _edl([1, 2, 1])
    edl.clips[1].transition_in = Transition(kind="dissolve", duration_s=0.5)
    # Captured before adding kind_alternation, including all existing null fields.
    previous = (
        '{"version":1,"source_refs":["custom"],'
        '"shot_duration_s":{"p10":1.0,"median":1.0,"p90":1.8},"cuts_per_min":30.0,'
        '"beat_aligned_ratio":null,"bpm_range":null,"intro_s":null,"cold_open_clips":0,'
        '"speech_ratio":null,"transitions":{"hard_cut":0.5,"dissolve":0.5},'
        '"structure":{"montage_share":0.0,"intro_pace_ratio":null},"look":null,'
        '"subtitle":{"present":false,"position":null,"relative_size":null},'
        '"notes":"1 reference, 0.1 min, 2 cuts; median shot 1.0 s '
        '(p10 1.0, p90 1.8); 30.0 cuts/min"}'
    )
    assert profile_from_edl(edl, "custom").model_dump_json() == previous
    for supplied in (None, []):
        profile = profile_from_edl(edl, "custom", shots=supplied)
        assert profile.speech_ratio is None
        assert profile.structure.kind_alternation is None
        assert profile.model_dump_json() == previous


@pytest.fixture
def kind_shots() -> list[Shot]:
    quality = QualityFlags(sharpness=50, exposure_clip=0, shake=0)
    return [
        Shot(shot_id="s0", asset_id="a", start_s=0, end_s=4, quality=quality, kind="talking"),
        Shot(shot_id="s1", asset_id="a", start_s=4, end_s=8, quality=quality, kind="broll"),
        Shot(
            shot_id="s2", asset_id="a", start_s=8, end_s=12,
            quality=quality.model_copy(update={"warnings": ["kept_for_audio"]}),
        ),
        Shot(shot_id="s3", asset_id="a", start_s=12, end_s=16, quality=quality),
    ]  # fmt: skip


def test_profile_from_edl_speech_ratio_uses_classifier_and_clip_durations(
    kind_shots: list[Shot],
) -> None:
    edl = _edl([4, 3, 2, 2])
    edl.clips[0].speed = 2
    edl.clips[1].speed = 0.5
    original_edl = edl.model_dump_json()
    original_shots = [shot.model_dump_json() for shot in kind_shots]
    profile = profile_from_edl(edl, shots=list(reversed(kind_shots)))
    assert profile.speech_ratio == pytest.approx(4 / 12)
    assert edl.model_dump_json() == original_edl
    assert [shot.model_dump_json() for shot in kind_shots] == original_shots
    assert StyleProfile.model_validate_json(profile.model_dump_json()) == profile
    edl.clips = [edl.clips[1], edl.clips[3]]
    profile = profile_from_edl(edl, shots=kind_shots)
    assert profile.speech_ratio == 0.0
    assert '"speech_ratio":0.0' in profile.model_dump_json()


def test_profile_from_edl_kind_alternation_uses_classifier_and_clip_order(
    kind_shots: list[Shot],
) -> None:
    edl = _edl([2] * 4)
    original_shots = [shot.model_dump_json() for shot in kind_shots]
    original_edl = edl.model_dump_json()
    baseline = profile_from_edl(edl)
    measured = profile_from_edl(edl, shots=list(reversed(kind_shots)))
    assert measured.structure.kind_alternation == 1.0
    assert measured.structure.montage_share == baseline.structure.montage_share
    assert measured.structure.intro_pace_ratio == baseline.structure.intro_pace_ratio
    assert [shot.model_dump_json() for shot in kind_shots] == original_shots
    assert edl.model_dump_json() == original_edl
    # A one-sided measurement is shown, but remains unmeasured in the distance.
    assert profile_distance(baseline, measured) == profile_distance(baseline, baseline)
    assert "kind_alternation" in format_profile("p", measured)
    assert format_diff("a", baseline, "b", measured, profile_distance(baseline, measured)) == (
        format_diff("a", baseline, "b", baseline, profile_distance(baseline, baseline))
    )
    edl.clips = [edl.clips[i] for i in (0, 2, 1, 3)]
    assert profile_from_edl(edl, shots=kind_shots).structure.kind_alternation == 1 / 3


@pytest.mark.parametrize("unknown", ["missing", "rejected"])
def test_profile_from_edl_unknown_kind_breaks_adjacency(
    kind_shots: list[Shot],
    unknown: str,
) -> None:
    edl = _edl([2] * 3)
    edl.clips[1].shot_id = unknown
    edl.clips[2].shot_id = "s1"
    kind_shots.append(
        Shot(
            shot_id="rejected", asset_id="a", start_s=16, end_s=20,
            quality=QualityFlags(sharpness=0, exposure_clip=0, shake=0, rejected=True),
        )
    )  # fmt: skip
    assert profile_from_edl(edl, shots=kind_shots).structure.kind_alternation is None
    assert profile_from_edl(edl, shots=kind_shots).speech_ratio == 0.5
    edl.clips.append(Clip(shot_id="s1", in_s=0, out_s=2))
    assert profile_from_edl(edl, shots=kind_shots).structure.kind_alternation == 0.0
    assert profile_from_edl(edl, shots=kind_shots).speech_ratio == pytest.approx(1 / 3)
    for clip in edl.clips:
        clip.shot_id = unknown
    assert profile_from_edl(edl, shots=kind_shots).speech_ratio is None


def test_profile_from_edl_measures_bursts_in_speed_adjusted_timeline_order() -> None:
    edl = _edl([2] * 7)
    for clip in edl.clips[:3]:
        clip.speed = 4  # source spans are all 2 s; timeline starts with three .5 s shots
    profile = profile_from_edl(edl)
    assert profile.shot_duration_s.median == 2
    assert profile.structure is not None
    assert profile.structure.montage_share == 3 / 7
    edl.clips[1], edl.clips[4] = edl.clips[4], edl.clips[1]
    assert profile_from_edl(edl).structure.montage_share == 0


@pytest.mark.parametrize(("a_value", "b_value", "delta"), [(0.2, 0.7, 0.5), (0.0, 0.0, 0.0)])
def test_kind_alternation_distance_and_diff_row(
    a_value: float, b_value: float, delta: float
) -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0, "kind_alternation": a_value})
    b = _profile(1, 2, 4, 20, structure={"montage_share": 0, "kind_alternation": b_value})
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim["kind_alternation"] == pytest.approx(delta, abs=1e-9)
        assert "kind_alternation" not in d.unmeasured
        assert len(d.per_dim) == 6  # four pacing dims, montage and alternation
        assert d.total == pytest.approx(delta / 6, abs=1e-9)
    text = format_diff("a", a, "b", b, profile_distance(a, b))
    row = next(line for line in text.splitlines() if line.startswith("kind_alternation"))
    assert row.split() == ["kind_alternation", f"{a_value:.2f}", f"{b_value:.2f}", f"{delta:.2f}"]


@pytest.mark.parametrize("structure", [None, {"montage_share": 0.5, "kind_alternation": None}])
def test_kind_alternation_skips_unmeasured_on_either_side(
    structure: dict[str, float | None] | None,
) -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0.5})
    b = _profile(0.5, 1, 2, 40, structure=structure)
    baseline = profile_distance(a, b)
    a.structure.kind_alternation = 0.7
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim == baseline.per_dim
        assert d.total == baseline.total
        assert "kind_alternation" not in d.per_dim
        assert "kind_alternation" in d.unmeasured
        lines = format_diff("a", left, "b", right, d).splitlines()
        assert not any(line.startswith("kind_alternation") for line in lines)
        assert "kind_alternation" in lines[-1]
    assert "kind_alternation" not in format_profile("b", b)
    if structure is None:
        assert baseline.total == profile_distance(a.model_copy(update={"structure": None}), b).total


def test_montage_distance_and_diff_row() -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0.2})
    b = _profile(1, 2, 4, 20, structure={"montage_share": 0.7})
    assert profile_distance(a, a).per_dim["montage_share"] == 0
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim["montage_share"] == pytest.approx(0.5)
        assert d.total == pytest.approx(0.5 / 5)
    text = format_diff("a", a, "b", b, profile_distance(a, b))
    row = next(line for line in text.splitlines() if line.startswith("montage_share"))
    assert row.split() == ["montage_share", "0.20", "0.70", "0.50"]
    assert "montage_share" in format_profile("a", a)


def test_montage_distance_skips_unmeasured_structure_on_either_side() -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0.5})
    b = _profile(0.5, 1, 2, 40)
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert "montage_share" not in d.per_dim
        assert "montage_share" in d.unmeasured
        assert d.total == pytest.approx(1.0)  # mean over just the four pacing dimensions


def test_profile_from_edl_intro_pace_respects_speed_and_order() -> None:
    edl = _edl([20] * 6)
    for clip in edl.clips[:4]:
        clip.speed = 4  # intro: four 5 s shots; body: two 20 s shots
    assert profile_from_edl(edl).structure.intro_pace_ratio == 0.25
    edl.clips[0], edl.clips[4] = edl.clips[4], edl.clips[0]
    assert profile_from_edl(edl).structure.intro_pace_ratio is None  # just one intro shot


@pytest.mark.parametrize(("ratio", "octaves"), [(0.25, 0), (0.5, 1), (1, 2)])
def test_intro_pace_distance_and_tables(ratio: float, octaves: float) -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0, "intro_pace_ratio": 0.25})
    b = _profile(1, 2, 4, 20, structure={"montage_share": 0, "intro_pace_ratio": ratio})
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert d.per_dim["intro_pace"] == pytest.approx(octaves)
        assert d.total == pytest.approx(octaves / 6)
    text = format_diff("a", a, "b", b, profile_distance(a, b))
    row = next(line for line in text.splitlines() if line.startswith("intro_pace"))
    assert row.split() == ["intro_pace", "0.25", f"{ratio:.2f}", f"{octaves:.2f}"]
    assert "intro_pace" in format_profile("a", a)


@pytest.mark.parametrize("structure", [None, {"montage_share": 0.5}])
def test_intro_pace_distance_skips_unmeasured_on_either_side(
    structure: dict[str, float] | None,
) -> None:
    a = _profile(1, 2, 4, 20, structure={"montage_share": 0.5, "intro_pace_ratio": 2})
    b = _profile(0.5, 1, 2, 40, structure=structure)
    for left, right in [(a, b), (b, a)]:
        d = profile_distance(left, right)
        assert "intro_pace" not in d.per_dim
        assert "intro_pace" in d.unmeasured
        assert d.total == pytest.approx(4 / (5 if structure else 4))
        assert "intro_pace" in format_diff("a", left, "b", right, d).splitlines()[-1]
    assert "intro_pace" not in format_profile("b", b)


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
    # Evaluate pacing: synthetic footage differs in look, which planners do not match.
    params = ReferenceParams(look=False)
    fast = build_profile([fast_ref], refs_ws, params=params)
    slow = build_profile([slow_ref], refs_ws, params=params)
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
    d_video = profile_distance(fast, build_profile([quick.preview_path], refs_ws, params=params))
    assert d_video.total < d_refs.total / 3, (d_video, d_refs)
    assert d_video.total < 0.2, d_video
    assert d_video.per_dim["cuts_per_min"] < 0.3  # the detector keeps our own cuts

    calm = cut(day, ws, target_s=9.0, style=slow_json, render=False, progress=lambda _m: None)
    measured = profile_from_edl(calm.edl)
    assert profile_distance(slow, measured).total < profile_distance(fast, measured).total
    assert len(calm.edl.clips) < len(quick.edl.clips)
