from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from vlogcut.analyze.kind import classify_kind
from vlogcut.plan import RoughCutError, RoughCutParams, rough_cut, score_shot
from vlogcut.plan.rules import (
    DEFAULT_PACING,
    _Candidate,
    _estimated_speech_ratio,
    _lengths,
    _ranker_percentiles,
    apply_cold_open,
    apply_intro_pace,
    apply_kind_alternation,
    apply_montage,
    apply_speech_ratio,
    assign_lengths,
    assign_transitions,
    avoid_adjacent_duplicates,
    expected_length,
    fit_lengths,
    place_on_cut_points,
    sample_lengths,
)
from vlogcut.schemas import (
    Clip,
    CutPoint,
    DurationStats,
    MediaAsset,
    QualityFlags,
    Shot,
    StructureStats,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.style.distance import profile_from_edl, speech_ratio
from vlogcut.style.profile import duration_stats, kind_alternation, montage_share
from vlogcut.validate.checks import validate_edl

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


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("coverage", "90dd4166fadd5d1c393363e68b0d63688ef35bb7fc6d558cb7849f98821abf80"),
        ("highlights", "5bf72aaec4502e19047514e9f27e20b340af7ca009430793332ef2e559ee2a52"),
    ],
)
def test_shot_kind_preserves_legacy_edl_bytes(strategy: str, expected: str) -> None:
    from vlogcut.analyze.kind import classify_kind

    shots, assets = _day(4)
    shots[0].quality.warnings = ["shaky", "kept_for_audio"]
    shots[2].quality.rejected = True
    params = RoughCutParams(target_s=24, strategy=strategy)
    # Digests captured from the planner before adding Shot.kind.
    legacy = rough_cut(shots, assets, params).model_dump_json().encode()
    assert sha256(legacy).hexdigest() == expected
    for shot in shots:
        shot.kind = classify_kind(shot)
    assert rough_cut(shots, assets, params).model_dump_json().encode() == legacy


@pytest.mark.parametrize("talking", [False, True])
def test_explicit_kind_controls_talking_placement_and_score(talking: bool) -> None:
    shot = _shot("a", 0, sharpness=0, warnings=[] if talking else ["kept_for_audio"])
    shot.kind = "talking" if talking else "broll"
    candidate = _Candidate(shot=shot, lo=0, hi=6, order=0, score=score_shot(shot))
    assert candidate.talking is talking
    assert score_shot(shot) == pytest.approx(0.79 if talking else 0.49)
    edl = rough_cut([shot], [_asset("a", 0)], RoughCutParams(target_s=3))
    assert edl.clips[0].in_s == (0 if talking else 1.5)


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("target", [0.0, 1.0])
def test_rough_cut_kind_alternation_moves_toward_target(strategy: str, target: float) -> None:
    shots = [_shot("a", i) for i in range(6)]
    for i, shot in enumerate(shots):
        shot.kind = "talking" if i % 2 == 0 else "broll"
    if target == 0:
        shots[2].kind, shots[3].kind = "broll", "talking"
    scores = {s.shot_id: float(i % 2 == 0) for i, s in enumerate(shots)}
    profile = StyleProfile(shot_duration_s=DEFAULT_PACING, cuts_per_min=20)
    params = RoughCutParams(target_s=9.45, strategy=strategy)
    assets = [_asset("a", 0)]

    def score(shot: Shot) -> float:
        return scores[shot.shot_id]

    plain = rough_cut(shots, assets, params, profile, score)
    profile.structure = StructureStats(montage_share=0, kind_alternation=target)
    edl = rough_cut(shots, assets, params, profile, score)
    by_id = _by_id(shots)
    before = kind_alternation([by_id[c.shot_id].kind for c in plain.clips])
    after = kind_alternation([by_id[c.shot_id].kind for c in edl.clips])
    assert before == 1 - target and after == target
    assert len(edl.clips) == len(plain.clips) == 3
    assert [c.shot_id for c in edl.clips] == sorted(c.shot_id for c in edl.clips)
    assert validate_edl(edl, shots, profile, target_s=params.target_s) == []
    assert (
        edl.model_dump_json() == rough_cut(shots, assets, params, profile, score).model_dump_json()
    )


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_unmeasured_kind_alternation_preserves_edl_bytes(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    shots, assets = _day(4)
    params = RoughCutParams(target_s=24, strategy=strategy)
    profile = StyleProfile(shot_duration_s=DEFAULT_PACING, cuts_per_min=20)
    absent = profile.model_copy(update={"structure": StructureStats(montage_share=0)})
    explicit_none = StyleProfile.model_validate(
        {
            **profile.model_dump(),
            "structure": {"montage_share": 0, "kind_alternation": None},
        }
    )
    expected = rough_cut(shots, assets, params, absent).model_dump_json().encode()

    def unexpected_swap(*args: object, **kwargs: object) -> list[_Candidate]:
        pytest.fail("unmeasured alternation must bypass the swap pass")

    monkeypatch.setattr("vlogcut.plan.rules.apply_kind_alternation", unexpected_swap)
    for unmeasured in (None, profile, absent, explicit_none):
        assert rough_cut(shots, assets, params, unmeasured).model_dump_json().encode() == expected


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("target", [0.0, 1.0])
@pytest.mark.parametrize("talking", [False, True])
def test_unimprovable_kind_alternation_preserves_edl_bytes(
    strategy: str, target: float, talking: bool
) -> None:
    shots = [_shot("a", i, usable_range=(0, 2)) for i in range(20)]
    for shot in shots:
        shot.kind = "talking" if talking else "broll"
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=24, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DEFAULT_PACING,
        cuts_per_min=20,
        structure=StructureStats(montage_share=0.4, intro_pace_ratio=0.5),
    )
    plain = rough_cut(shots, assets, params, profile)
    # Limited usable ranges force count fitting to add clips before the no-op swap.
    assert len(plain.clips) > round(params.target_s / expected_length(DEFAULT_PACING))
    assert "kind_alternation" not in profile.model_dump()["structure"]
    profile.structure = profile.structure.model_copy(update={"kind_alternation": target})
    measured = rough_cut(shots, assets, params, profile)
    assert measured.model_dump_json().encode() == plain.model_dump_json().encode()
    assert validate_edl(measured, shots, profile, target_s=params.target_s) == []


@pytest.mark.parametrize("ranked,expected", [(False, 1), (True, 3)])
def test_alternation_uses_blended_scores_and_filters(ranked: bool, expected: int) -> None:
    shots = [_shot("a", i) for i in range(8)]
    for i, shot in enumerate(shots):
        shot.kind = "talking" if i in (0, 2, 4) else "broll"
    shots[5].quality.rejected = True
    shots[6].quality.usable_range = (0, 0.5)
    shots[7].end_s = shots[7].start_s + 0.5
    scores = dict(zip((s.shot_id for s in shots), [1, 0.3, 1, 0.2, 1, 0.9, 0.9, 0.9], strict=True))
    ranks = {s.shot_id: r for s, r in zip(shots, [5, 0, 5, 1, 5, 100, 99, 98], strict=True)}
    profile = StyleProfile(
        shot_duration_s=DEFAULT_PACING,
        cuts_per_min=20,
        structure=StructureStats(montage_share=0, kind_alternation=1),
    )
    edl = rough_cut(
        shots,
        [_asset("a", 0)],
        RoughCutParams(target_s=9.45, strategy="highlights"),
        profile,
        lambda shot: scores[shot.shot_id],
        ranker_scores=ranks if ranked else None,
    )
    assert [c.shot_id for c in edl.clips] == [shots[i].shot_id for i in (0, expected, 4)]
    assert validate_edl(edl, shots, profile, target_s=9.45) == []


@pytest.mark.parametrize("tie", ["score", "start", "id"])
def test_alternation_swap_priority_and_purity(tie: str) -> None:
    cands = [_candidate(i, None) for i in range(5)]
    for i, c in enumerate(cands):
        c.shot.kind = "broll" if i in (1, 3) else "talking"
    if tie == "score":
        cands[3].score = 2
    elif tie == "id":
        cands[3].shot.start_s = cands[1].shot.start_s
        cands[3].shot.shot_id = "a_000_first"
    picked = cands[::2]
    before = [c.model_dump() for c in cands]
    result = apply_kind_alternation(picked, list(reversed(cands)), 1)
    assert result == [cands[0], cands[1 if tie == "start" else 3], cands[4]]
    assert [c.model_dump() for c in cands] == before
    assert picked == cands[::2]
    # A move from 0 to 0.5 does not strictly improve the gap to 0.25.
    assert apply_kind_alternation(picked, cands, 0.25) == picked


def test_alternation_checks_both_duplicate_neighbours_and_chronology() -> None:
    cands = [_candidate(i, None) for i in range(9)]
    for c in cands:
        c.shot.kind = "talking"
    cands[4].shot.kind = "broll"
    cands[1].shot.dup_group = cands[2].shot.dup_group = "x"
    cands[6].shot.dup_group = cands[7].shot.dup_group = ""
    cands[3].score = 0.5
    cands[5].shot.dup_group = "x"
    picked = [cands[i] for i in (1, 4, 7)]
    assert apply_kind_alternation(picked, cands, 0) == [cands[i] for i in (1, 3, 7)]


def test_alternation_prefers_score_over_largest_improvement() -> None:
    cands = [_candidate(i, None) for i in range(10)]
    for c in cands:
        c.shot.kind = "talking"
    cands[0].shot.kind = cands[5].shot.kind = "broll"
    cands[0].score = 2
    picked = [cands[i] for i in (1, 3, 6, 9)]
    # First take the higher-score endpoint (0 -> 1/3), then the interior (1/3 -> 1).
    # Taking the larger interior improvement first would block the endpoint swap.
    pool = [*picked, cands[0], cands[5]]
    assert apply_kind_alternation(picked, pool, 1) == [cands[i] for i in (0, 3, 5, 9)]


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("cold_open", [False, True])
def test_alternation_composes_with_later_planner_steps(strategy: str, cold_open: bool) -> None:
    shots, assets = _day()
    # Validation expects its supplied shot table in chronological asset order.
    shots.sort(key=lambda shot: ("bacz".index(shot.asset_id), shot.start_s))
    for i, shot in enumerate(shots):
        shot.kind = "talking" if i % 2 else "broll"
        shot.cut_points = [CutPoint(time_s=shot.start_s + 1, kind="pause")]
    params = RoughCutParams(target_s=60, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=0.75, median=4, p90=6),
        cuts_per_min=15,
        structure=StructureStats(montage_share=0.4, intro_pace_ratio=0.5, kind_alternation=1),
        cold_open_clips=1,
    )
    chronological = rough_cut(shots, assets, params, profile)
    edl = rough_cut(shots, assets, params, profile, cold_open=cold_open)
    assert {c.shot_id for c in edl.clips} == {c.shot_id for c in chronological.clips}
    assert validate_edl(edl, shots, profile, target_s=60) == []


@pytest.mark.parametrize("kinds", [["talking", None, "broll"], [None], []])
def test_alternation_without_known_pairs_does_nothing(
    monkeypatch: pytest.MonkeyPatch, kinds: list[str | None]
) -> None:
    # Eligible legacy shots normally get a fallback kind; isolate unknown classifier results.
    monkeypatch.setattr("vlogcut.plan.rules.classify_kind", lambda shot: shot.kind)
    picked = [_candidate(i, None) for i in range(len(kinds))]
    for c, kind in zip(picked, kinds, strict=True):
        c.shot.kind = kind
    alternative = _candidate(10, None)
    alternative.shot.kind = "talking"
    assert kind_alternation(kinds) is None
    assert apply_kind_alternation(picked, [*picked, alternative], 1) == picked


def test_alternation_unknown_candidates_are_skipped_and_unknown_picks_break_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vlogcut.plan.rules.classify_kind", lambda shot: shot.kind)
    cands = [_candidate(i, None) for i in range(5)]
    for c, kind in zip(cands, ["talking", None, "talking", None, "broll"], strict=True):
        c.shot.kind = kind
    picked = [cands[i] for i in (0, 2, 3, 4)]
    # The only known pair is talking/talking; do not bridge the unknown pick.
    assert apply_kind_alternation(picked, cands, 0) == picked
    assert apply_kind_alternation(picked, cands, 1) == picked


def test_alternation_preserves_capacity_for_duration_fit() -> None:
    shots = [_shot("a", i) for i in range(4)]
    for shot in shots:
        shot.kind = "talking"
    shots[1].kind = "broll"
    shots[1].quality.usable_range = (0, 1.5)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1.5, median=5, p90=5),
        cuts_per_min=12,
        structure=StructureStats(montage_share=0, kind_alternation=1),
    )
    edl = rough_cut(
        shots, [_asset("a", 0)], RoughCutParams(target_s=13, strategy="highlights"), profile
    )
    assert shots[1].shot_id not in {c.shot_id for c in edl.clips}
    assert validate_edl(edl, shots, profile, target_s=13) == []


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("with_profile", [False, True])
def test_unmeasured_speech_ratio_preserves_legacy_bytes(
    monkeypatch: pytest.MonkeyPatch, strategy: str, with_profile: bool
) -> None:
    shots, assets = _day(4)
    shots[0].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=24, strategy=strategy)
    profile = (
        StyleProfile(
            shot_duration_s=DEFAULT_PACING,
            cuts_per_min=20,
            structure=StructureStats(montage_share=0, kind_alternation=0.5),
        )
        if with_profile
        else None
    )
    actual = rough_cut(shots, assets, params, profile).model_dump_json().encode()
    # Bypass the pass to reproduce the previous planner with the same inputs.
    monkeypatch.setattr("vlogcut.plan.rules.apply_speech_ratio", lambda picked, *a, **kw: picked)
    assert actual == rough_cut(shots, assets, params, profile).model_dump_json().encode()

    def unexpected_swap(*args: object, **kwargs: object) -> list[_Candidate]:
        pytest.fail("unmeasured speech ratio must bypass the pass")

    monkeypatch.setattr("vlogcut.plan.rules.apply_speech_ratio", unexpected_swap)
    if profile:
        profile = profile.model_copy(update={"speech_ratio": None})
    assert actual == rough_cut(shots, assets, params, profile).model_dump_json().encode()


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("target", [0.0, 0.8])
@pytest.mark.parametrize("alternation", [None, 0.0, 0.5, 1.0])
def test_speech_ratio_moves_duration_share_and_preserves_invariants(
    strategy: str, target: float, alternation: float | None
) -> None:
    shots = [_shot("a", i) for i in range(24)]
    for i, shot in enumerate(shots):
        # Select the opposite kind from the speech target before swaps.
        if (i % 2 == 0) == (target == 0):
            shot.quality.warnings = ["kept_for_audio"]
        shot.dup_group = str(i // 2)
    profile = StyleProfile(
        shot_duration_s=DEFAULT_PACING,
        cuts_per_min=20,
        structure=StructureStats(montage_share=0, kind_alternation=alternation),
    )
    params = RoughCutParams(target_s=37.8, strategy=strategy)
    assets = [_asset("a", 0)]

    def score(shot: Shot) -> float:
        return float(shots.index(shot) % 2 == 0)

    plain = rough_cut(shots, assets, params, profile, score)
    profile.speech_ratio = target
    edl = rough_cut(shots, assets, params, profile, score)
    kinds = {shot.shot_id: classify_kind(shot) for shot in shots}
    before = speech_ratio(plain.clips, kinds)
    after = speech_ratio(edl.clips, kinds)
    assert before is not None and after is not None
    if alternation is None:
        assert abs(after - target) < abs(before - target)
    else:
        before_alt = kind_alternation([kinds[c.shot_id] for c in plain.clips])
        after_alt = kind_alternation([kinds[c.shot_id] for c in edl.clips])
        assert before_alt is not None and after_alt is not None
        assert abs(after_alt - alternation) <= abs(before_alt - alternation)
    assert len(edl.clips) == len(plain.clips)
    assert [c.shot_id for c in edl.clips] == sorted(c.shot_id for c in edl.clips)
    groups = [_by_id(shots)[c.shot_id].dup_group for c in edl.clips]
    assert all(a != b for a, b in zip(groups, groups[1:], strict=False))
    assert edl.total_duration_s == pytest.approx(params.target_s, rel=0.1)
    assert validate_edl(edl, shots, profile, target_s=params.target_s) == []
    assert (
        edl.model_dump_json() == rough_cut(shots, assets, params, profile, score).model_dump_json()
    )


def test_speech_estimate_uses_caps_mean_and_known_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vlogcut.plan.rules.classify_kind", lambda shot: shot.kind)
    cands = [_candidate(i, None) for i in range(4)]
    for c, kind in zip(cands, ["talking", "broll", "broll", None], strict=True):
        c.shot.kind = kind
    cands[0].lo, cands[0].hi = 1, 5  # usable duration, not whole shot duration
    cands[1].hi = 2
    pace = DurationStats(p10=1, median=3, p90=5)
    assert _estimated_speech_ratio(cands, pace) == pytest.approx(4 / (4 + 2 + 3))
    assert _estimated_speech_ratio([cands[-1]], pace) is None
    assert apply_speech_ratio([cands[-1]], cands, 0.8, pace) == [cands[-1]]
    assert apply_speech_ratio([], cands, 0.8, pace) == []


@pytest.mark.parametrize("ranked,expected", [(False, 1), (True, 3)])
def test_speech_swaps_use_blended_scores_and_eligible_pool(ranked: bool, expected: int) -> None:
    shots = [_shot("a", i) for i in range(8)]
    for i, shot in enumerate(shots):
        shot.kind = "talking" if i in (0, 2, 4) else "broll"
    shots[5].quality.rejected = True
    shots[6].quality.usable_range = (0, 0.5)
    shots[7].end_s = shots[7].start_s + 0.5
    scores = dict(zip((s.shot_id for s in shots), [1, 0.3, 1, 0.2, 1, 0.9, 0.9, 0.9], strict=True))
    ranks = {s.shot_id: r for s, r in zip(shots, [5, 0, 5, 1, 5, 100, 99, 98], strict=True)}
    profile = StyleProfile(
        shot_duration_s=DEFAULT_PACING,
        cuts_per_min=20,
        speech_ratio=10 / (10 + expected_length(DEFAULT_PACING)),
    )
    edl = rough_cut(
        shots,
        [_asset("a", 0)],
        RoughCutParams(target_s=9.45, strategy="highlights"),
        profile,
        lambda shot: scores[shot.shot_id],
        ranker_scores=ranks if ranked else None,
    )
    assert {c.shot_id for c in edl.clips} == {
        shots[i].shot_id for i in (expected, 2 if not ranked else 0, 4)
    }


def test_speech_swaps_prioritize_gap_and_preserve_inputs_and_capacity() -> None:
    cands = [_candidate(i, None) for i in range(5)]
    for c in cands:
        c.shot.kind = "talking"
    cands[1].shot.kind = cands[3].shot.kind = "broll"
    cands[1].hi, cands[1].score = 1.5, 2
    picked = cands[::2]
    before = [c.model_dump() for c in cands]
    target = 10 / (10 + expected_length(DEFAULT_PACING))
    # The larger improvement beats the higher incoming score; earlier slot wins ties.
    assert apply_speech_ratio(picked, cands, target, DEFAULT_PACING) == [
        cands[0],
        cands[3],
        cands[4],
    ]
    assert apply_speech_ratio(picked, cands, 1, DEFAULT_PACING) == picked
    assert apply_speech_ratio(picked, cands, 0, DEFAULT_PACING, min_total_s=15) == [
        cands[0],
        cands[3],
        cands[4],
    ]
    assert [c.model_dump() for c in cands] == before
    assert picked == cands[::2]


def test_speech_swap_can_improve_duration_share_without_changing_kinds() -> None:
    cands = [_candidate(i, None) for i in range(3)]
    cands[0].shot.kind = cands[1].shot.kind = "talking"
    cands[2].shot.kind = "broll"
    cands[1].hi = 2
    picked = [cands[0], cands[2]]
    result = apply_speech_ratio(picked, cands, 0.4, DEFAULT_PACING, alternation=1)
    assert result == [cands[1], cands[2]]
    # Preserve an existing capacity shortfall as well as a feasible target.
    assert apply_speech_ratio(picked, cands, 0.4, DEFAULT_PACING, min_total_s=20) == picked


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({}, [0.5] * 5),
        ({"b": 99}, [0.5] * 5),
        ({"a": 10, "b": 10, "c": 30, "d": 50}, [1 / 6, 1 / 6, 2 / 3, 1, 0.5]),
        ({"a": 8, "b": 8, "c": 8}, [0.5] * 5),
    ],
)
def test_ranker_percentiles_ties_single_and_missing(
    scores: dict[str, float], expected: list[float]
) -> None:
    before = scores.copy()
    result = _ranker_percentiles(list("abcde"), {**scores, "unknown": float("nan")})
    assert list(result.values()) == pytest.approx(expected)
    assert scores == before


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("cold_open", [False, True])
def test_absent_ranker_scores_preserve_edl_bytes(strategy: str, cold_open: bool) -> None:
    shots, assets = _day()
    params = RoughCutParams(target_s=60, strategy=strategy)
    profile = StyleProfile(shot_duration_s=DEFAULT_PACING, cuts_per_min=20, cold_open_clips=1)
    baseline = rough_cut(shots, assets, params, profile, cold_open=cold_open)
    for scores in (None, {}):
        actual = rough_cut(
            shots, assets, params, profile, cold_open=cold_open, ranker_scores=scores
        )
        assert actual.model_dump_json().encode() == baseline.model_dump_json().encode()


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_ranker_scores_change_selection(strategy: str) -> None:
    shots = [_shot("a", 0), _shot("a", 1, sharpness=0)]
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=3, strategy=strategy)
    baseline = rough_cut(shots, assets, params)
    assert [c.shot_id for c in baseline.clips] == [shots[0].shot_id]
    scores = {shots[0].shot_id: -1e300, shots[1].shot_id: 1e300}
    ranked = rough_cut(shots, assets, params, ranker_scores=scores)
    assert [c.shot_id for c in ranked.clips] == [shots[1].shot_id]
    assert validate_edl(ranked, shots, target_s=3) == []


def test_ranker_scores_break_length_assignment_ties() -> None:
    shots = [_shot("a", 0), _shot("a", 1, sharpness=0)]
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=6)
    baseline = rough_cut(shots, assets, params)
    ranked = rough_cut(
        shots, assets, params, ranker_scores={shots[0].shot_id: -100, shots[1].shot_id: 100}
    )
    assert [c.shot_id for c in ranked.clips] == [c.shot_id for c in baseline.clips]
    assert baseline.clips[0].timeline_duration_s > baseline.clips[1].timeline_duration_s
    assert ranked.clips[0].timeline_duration_s < ranked.clips[1].timeline_duration_s


def test_unknown_ranker_ids_are_ignored() -> None:
    shots, assets = _day()
    params = RoughCutParams(target_s=30)
    unknown = {"unknown": float("nan"), "also_unknown": float("inf")}
    for scores in ({}, {shots[0].shot_id: -10, shots[-1].shot_id: 10}):
        baseline = rough_cut(shots, assets, params, ranker_scores=scores)
        actual = rough_cut(shots, assets, params, ranker_scores={**scores, **unknown})
        assert actual.model_dump_json() == baseline.model_dump_json()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("status", ["eligible", "rejected", "too_short"])
def test_known_ranker_scores_must_be_finite(value: float, status: str) -> None:
    shot = _shot("a", 0, dur=0.5 if status == "too_short" else 6)
    shot.quality.rejected = status == "rejected"
    with pytest.raises(ValueError, match=shot.shot_id):
        rough_cut([shot], [_asset("a", 0)], ranker_scores={shot.shot_id: value})


def test_ranker_blend_uses_only_eligible_candidates_and_defaults_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shots = [_shot("a", i) for i in range(5)]
    shots[3].quality.rejected = True
    shots[4].quality.usable_range = (0, 0.5)
    scores = {shots[i].shot_id: value for i, value in enumerate([-10, 10, 0, 1000, -1000])}
    del scores[shots[2].shot_id]
    captured: list[float] = []

    def capture_lengths(
        targets: list[float], caps: list[float], scores: list[float], talking: list[bool]
    ) -> list[float]:
        captured.extend(scores)
        return assign_lengths(targets, caps, scores, talking)

    monkeypatch.setattr("vlogcut.plan.rules.assign_lengths", capture_lengths)
    edl = rough_cut(shots, [_asset("a", 0)], RoughCutParams(target_s=9), ranker_scores=scores)
    assert [c.shot_id for c in edl.clips] == [s.shot_id for s in shots[:3]]
    assert captured == pytest.approx([0.5 * score_shot(shots[0]) + 0.5 * p for p in (0, 1, 0.5)])
    assert validate_edl(edl, shots, target_s=9) == []


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("cold_open", [False, True])
def test_ranker_preserves_filters_order_and_cut_point_alignment(
    strategy: str, cold_open: bool
) -> None:
    shots = [_shot("a", i) for i in range(8)]
    shots[0].dup_group = shots[1].dup_group = "duplicate"
    shots[6].quality.rejected = True
    for shot in shots:
        shot.quality.usable_range = (1, 5)
        shot.cut_points = [CutPoint(time_s=shot.start_s + 1.25, kind="pause")]
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=2, median=2, p90=2),
        cuts_per_min=30,
        cold_open_clips=int(cold_open),
    )
    scores = {
        s.shot_id: value for s, value in zip(shots, [8, 7, 1, 6, 0, 5, 1000, -1], strict=True)
    }
    edl = rough_cut(
        shots,
        [_asset("a", 0)],
        RoughCutParams(target_s=8, strategy=strategy),
        profile,
        ranker_scores=scores,
        cold_open=cold_open,
    )
    ids = [c.shot_id for c in edl.clips]
    body = ids[1:] if cold_open else ids
    assert body == sorted(body)
    assert shots[6].shot_id not in ids
    assert all(c.in_s == 1.25 and c.out_s == 3.25 for c in edl.clips)
    assert validate_edl(edl, shots, profile, target_s=8) == []


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("enabled,allowance", [(False, 1), (True, 0), (True, None)])
def test_cold_open_disabled_preserves_json(
    monkeypatch: pytest.MonkeyPatch, strategy: str, enabled: bool, allowance: int | None
) -> None:
    shots, assets = _day()
    params = RoughCutParams(target_s=60, strategy=strategy)
    profile = (
        StyleProfile(shot_duration_s=DEFAULT_PACING, cuts_per_min=20, cold_open_clips=allowance)
        if allowance is not None
        else None
    )
    baseline = rough_cut(shots, assets, params).model_dump_json().encode()

    def unexpected_move(*args: object, **kwargs: object) -> list[Clip]:
        pytest.fail("disabled cold open must bypass the helper")

    monkeypatch.setattr("vlogcut.plan.rules.apply_cold_open", unexpected_move)
    assert rough_cut(shots, assets, params, profile).model_dump_json().encode() == baseline
    actual = rough_cut(shots, assets, params, profile, cold_open=enabled)
    assert actual.model_dump_json().encode() == baseline


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("allowance", [1, 3])
@pytest.mark.parametrize("custom_score,hook", [(False, 4), (True, 3)])
def test_rough_cut_cold_open_moves_highest_scoring_later_clip(
    strategy: str, allowance: int, custom_score: bool, hook: int
) -> None:
    shots = [_shot("a", i, sharpness=q) for i, q in enumerate([200, 200, 20, 40, 160, 80])]
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=12, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=2, median=2, p90=2),
        cuts_per_min=30,
        cold_open_clips=allowance,
        transitions=TransitionMix(hard_cut=0, dissolve=1),
    )

    def score(shot: Shot) -> float:
        return -shot.quality.sharpness if custom_score else score_shot(shot)

    plain = rough_cut(shots, assets, params, profile, score=score)
    moved = rough_cut(shots, assets, params, profile, score=score, cold_open=True)
    assert [c.shot_id for c in moved.clips] == [shots[hook].shot_id] + [
        s.shot_id for i, s in enumerate(shots) if i != hook
    ]
    assert moved.total_duration_s == plain.total_duration_s == 12
    before = {c.shot_id: c for c in plain.clips}
    for clip in moved.clips:
        original = before[clip.shot_id]
        assert clip.model_dump(exclude={"transition_in"}) == original.model_dump(
            exclude={"transition_in"}
        )
        expected = (
            Transition()
            if clip.shot_id in {shots[i].shot_id for i in (0, hook, hook + 1)}
            else original.transition_in
        )
        assert clip.transition_in == expected
    assert validate_edl(moved, shots, profile, target_s=12) == []
    assert "not_chronological" in {v.code for v in validate_edl(moved, shots)}


def _hook_clips(shots: list[Shot]) -> list[Clip]:
    return [
        Clip(
            shot_id=s.shot_id,
            in_s=0.25,
            out_s=2.25 + i / 4,
            speed=1 + i / 4,
            audio="duck" if i % 2 else "mute",
            transition_in=Transition(kind="dissolve", duration_s=0.2),
        )
        for i, s in enumerate(shots)
    ]


@pytest.mark.parametrize("count", [3, 6])
def test_apply_cold_open_ties_choose_last_and_preserve_inputs(count: int) -> None:
    shots = [_shot("a", i) for i in range(count)]
    clips = _hook_clips(shots)
    before_clips = [c.model_dump() for c in clips]
    before_shots = [s.model_dump() for s in shots]
    moved = apply_cold_open(clips, _by_id(shots))
    assert [c.shot_id for c in moved] == [clips[-1].shot_id] + [c.shot_id for c in clips[:-1]]
    assert sum(c.timeline_duration_s for c in moved) == pytest.approx(
        sum(c.timeline_duration_s for c in clips)
    )
    for position, original_index in enumerate([count - 1, *range(count - 1)]):
        expected = before_clips[original_index].copy()
        if position < 2:
            expected["transition_in"] = Transition().model_dump()
        assert moved[position].model_dump() == expected
    assert [c.model_dump() for c in clips] == before_clips
    assert [s.model_dump() for s in shots] == before_shots


@pytest.mark.parametrize("count", [0, 1, 2])
def test_apply_cold_open_short_timeline_is_unchanged(count: int) -> None:
    shots = [_shot("a", i) for i in range(count)]
    clips = _hook_clips(shots)
    assert apply_cold_open(clips, _by_id(shots)) == clips


@pytest.mark.parametrize(
    "groups,expected",
    [
        (["x", None, None, None, "x", None], 5),  # hook conflicts with first clip
        ([None, None, None, "x", None, "x"], 5),  # removal would join duplicates
        (["x", "y", "z", "x", "y", "x"], 1),  # all later-half moves conflict
        (["x", "y", "x"], None),  # every possible move conflicts
        (["", "y", ""], None),  # empty strings are non-null groups too
    ],
)
def test_apply_cold_open_duplicate_fallback(
    groups: list[str | None], expected: int | None, caplog: pytest.LogCaptureFixture
) -> None:
    shots = [_shot("a", i, sharpness=q) for i, q in enumerate([200, 200, 20, 40, 160, 80])]
    shots = shots[: len(groups)]
    for shot, group in zip(shots, groups, strict=True):
        shot.dup_group = group
    clips = _hook_clips(shots)
    before = [c.model_dump() for c in clips]
    with caplog.at_level("DEBUG", logger="vlogcut.plan.rules"):
        moved = apply_cold_open(clips, _by_id(shots))
    if expected is None:
        assert moved == clips
        assert "Skipping cold open" in caplog.text
    else:
        assert [c.shot_id for c in moved] == [shots[expected].shot_id] + [
            s.shot_id for i, s in enumerate(shots) if i != expected
        ]
        by_id = _by_id(shots)
        for left, right in zip(moved, moved[1:], strict=False):
            group = by_id[left.shot_id].dup_group
            assert group is None or group != by_id[right.shot_id].dup_group
    assert [c.model_dump() for c in clips] == before


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("layout", ["single", "moments", "pooled_moments"])
def test_rough_cut_avoids_adjacent_duplicates(
    monkeypatch: pytest.MonkeyPatch, strategy: str, layout: str
) -> None:
    asset_ids = {
        "single": ["a"] * 6,
        "moments": ["a", "a", "b", "b", "c", "c"],
        "pooled_moments": list("abcdef"),
    }[layout]
    assets = [_asset(aid, i * 60) for i, aid in enumerate(dict.fromkeys(asset_ids))]
    shots = [_shot(aid, i) for i, aid in enumerate(asset_ids)]
    shots[0].dup_group = shots[2].dup_group = "repeat"
    scores = dict(zip((s.shot_id for s in shots), [10, 1, 9, 2, 8, 1], strict=True))
    params = RoughCutParams(target_s=9.45, strategy=strategy)

    def score(shot: Shot) -> float:
        return scores[shot.shot_id]

    with monkeypatch.context() as patch:
        patch.setattr(
            "vlogcut.plan.rules.avoid_adjacent_duplicates", lambda picked, _: list(picked)
        )
        naive = rough_cut(shots, assets, params, score=score)
    assert [c.shot_id for c in naive.clips] == [shots[i].shot_id for i in (0, 2, 4)]
    assert "duplicate_neighbours" in {v.code for v in validate_edl(naive, shots)}

    edl = rough_cut(shots, assets, params, score=score)
    assert [c.shot_id for c in edl.clips] == [shots[i].shot_id for i in (0, 3, 4)]
    assert len(edl.clips) == len(naive.clips) == 3
    assert validate_edl(edl, shots, target_s=params.target_s) == []
    assert edl.model_dump_json() == rough_cut(shots, assets, params, score=score).model_dump_json()


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_rough_cut_keeps_duplicates_without_alternatives(strategy: str) -> None:
    shots = [_shot("a", i) for i in range(3)]
    for shot in shots:
        shot.dup_group = "repeat"
    edl = rough_cut(shots, [_asset("a", 0)], RoughCutParams(target_s=9.45, strategy=strategy))
    assert [c.shot_id for c in edl.clips] == [s.shot_id for s in shots]
    assert [v.code for v in validate_edl(edl, shots)] == ["duplicate_neighbours"] * 2


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("unique_groups", [False, True])
def test_rough_cut_without_duplicate_groups_matches_legacy_json(
    monkeypatch: pytest.MonkeyPatch, strategy: str, unique_groups: bool
) -> None:
    shots, assets = _day(5)
    if unique_groups:
        for shot in shots:
            shot.dup_group = shot.shot_id
    params = RoughCutParams(target_s=30, strategy=strategy)
    actual = rough_cut(shots, assets, params).model_dump_json()
    monkeypatch.setattr(
        "vlogcut.plan.rules.avoid_adjacent_duplicates", lambda picked, _: list(picked)
    )
    assert actual.encode() == rough_cut(shots, assets, params).model_dump_json().encode()


def _candidate(order: int, group: str | None, score: float = 1.0) -> _Candidate:
    shot = _shot("a", order)
    shot.dup_group = group
    return _Candidate(shot=shot, lo=0, hi=shot.duration_s, order=order, score=score)


@pytest.mark.parametrize("later_score,expected", [(1.0, 1), (2.0, 3)])
def test_avoid_duplicates_score_then_chronology(later_score: float, expected: int) -> None:
    cands = [
        _candidate(0, "x"),
        _candidate(1, "y"),
        _candidate(2, "x"),
        _candidate(3, "z", later_score),
    ]
    picked = [cands[0], cands[2]]
    pools = [list(reversed(cands))] * 2
    before = [c.model_dump() for c in cands]
    result = avoid_adjacent_duplicates(picked, pools)
    assert [c.order for c in result] == [0, expected]
    assert result == avoid_adjacent_duplicates(picked, [cands] * 2)
    assert picked == [cands[0], cands[2]]
    assert [c.model_dump() for c in cands] == before
    assert pools == [list(reversed(cands))] * 2


def test_avoid_duplicates_checks_both_neighbours_and_chronology() -> None:
    cands = [
        _candidate(0, None, 10),  # before predecessor
        _candidate(1, "x", 10),  # already selected
        _candidate(2, "x", 10),  # conflicts with predecessor
        _candidate(3, "x"),  # original conflicting pick
        _candidate(4, "y", 10),  # conflicts with successor
        _candidate(5, None),  # valid even with a None successor
        _candidate(6, "y", 10),  # already selected
        _candidate(7, None, 10),  # after successor
    ]
    picked = [cands[i] for i in (1, 3, 6)]
    assert [c.order for c in avoid_adjacent_duplicates(picked, [cands] * 3)] == [1, 5, 6]
    cands[6].shot.dup_group = None
    cands[4].shot.dup_group = "x"
    assert [c.order for c in avoid_adjacent_duplicates(picked, [cands] * 3)] == [1, 5, 6]
    assert avoid_adjacent_duplicates(picked, [[c] for c in picked]) == picked


def test_avoid_duplicates_uses_updated_predecessor() -> None:
    cands = [_candidate(i, group) for i, group in enumerate(["x", None, "x", "x"])]
    picked = [cands[i] for i in (0, 2, 3)]
    assert [c.order for c in avoid_adjacent_duplicates(picked, [cands] * 3)] == [0, 1, 3]


@pytest.mark.parametrize("groups", [[], [None], [None, None], [None, "x", None], ["x", "y"]])
def test_avoid_duplicates_no_conflicts(groups: list[str | None]) -> None:
    picked = [_candidate(i, group) for i, group in enumerate(groups)]
    assert avoid_adjacent_duplicates(picked, [picked] * len(picked)) == picked


@pytest.mark.parametrize("offset", [4.123456, 7.123456])
def test_rough_cut_aligns_rounded_window_to_nearby_candidate(offset: float) -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, 9.0))
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=3.001)
    plain = rough_cut([shot], assets, params).clips[0]
    candidate = shot.start_s + offset
    shot.cut_points = [CutPoint(time_s=candidate, kind="pause")]
    aligned = rough_cut([shot], assets, params).clips[0]
    assert (
        min(abs(shot.start_s + edge - candidate) for edge in (aligned.in_s, aligned.out_s)) < 1e-6
    )
    assert aligned.timeline_duration_s == pytest.approx(plain.timeline_duration_s, abs=1e-12)
    assert (aligned.in_s, aligned.out_s) != (plain.in_s, plain.out_s)


def test_rough_cut_max_shift_and_unbounded_placement() -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, 9.0))
    assets = [_asset("a", 0)]
    params = RoughCutParams(target_s=3)
    plain = rough_cut([shot], assets, params)
    shot.cut_points = [CutPoint(time_s=108.5, kind="pause")]
    assert rough_cut([shot], assets, params).model_dump_json() == plain.model_dump_json()
    for limit in (2.0, None):
        clip = rough_cut([shot], assets, params, max_shift_s=limit).clips[0]
        assert shot.start_s + clip.out_s == 108.5
        assert clip.timeline_duration_s == plain.clips[0].timeline_duration_s


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_rough_cut_placement_preserves_multi_shot_invariants(strategy: str) -> None:
    assets = [_asset("a", 0), _asset("b", 60)]
    shots = [_shot(a.asset_id, i, dur=8, usable_range=(1.0, 7.0)) for a in assets for i in range(6)]
    shots[2].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=24, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=0.3, median=2, p90=5),
        cuts_per_min=30,
        transitions=TransitionMix(hard_cut=0.5, dissolve=0.5),
    )
    plain = rough_cut(shots, assets, params, profile)
    by_id = _by_id(shots)
    for i, clip in enumerate(plain.clips):
        shot = by_id[clip.shot_id]
        if i % 3 != 0 or "kept_for_audio" in shot.quality.warnings:
            shot.cut_points = [
                CutPoint(time_s=shot.start_s + offset, kind="pause")
                for offset in (0.5, clip.in_s + 0.125, 7.5)
            ]
    aligned = rough_cut(shots, assets, params, profile)
    assert any(a.in_s != b.in_s for a, b in zip(plain.clips, aligned.clips, strict=True))
    assert aligned.model_dump(exclude={"clips"}) == plain.model_dump(exclude={"clips"})
    assert aligned.total_duration_s == pytest.approx(plain.total_duration_s, abs=1e-12)
    for a, b in zip(plain.clips, aligned.clips, strict=True):
        assert a.model_dump(exclude={"in_s", "out_s"}) == b.model_dump(exclude={"in_s", "out_s"})
        assert b.timeline_duration_s == pytest.approx(a.timeline_duration_s, abs=1e-12)
        shot = by_id[b.shot_id]
        lo, hi = shot.quality.usable_range
        assert shot.start_s <= shot.start_s + b.in_s < shot.start_s + b.out_s <= shot.end_s
        assert lo <= b.in_s < b.out_s <= hi
        if not shot.cut_points:
            assert b == a


@pytest.mark.parametrize("usable_end,expected_in", [(9.0, 1.25), (4.0, 1.0)])
def test_rough_cut_places_audio_shots_unless_window_is_full(
    usable_end: float, expected_in: float
) -> None:
    shot = _shot("a", 10, dur=10, usable_range=(1.0, usable_end), warnings=["kept_for_audio"])
    shot.cut_points = [CutPoint(time_s=101.25, kind="pause")]
    clip = rough_cut([shot], [_asset("a", 0)], RoughCutParams(target_s=3)).clips[0]
    assert clip.in_s == expected_in
    assert clip.timeline_duration_s == 3
    assert clip.audio == "keep"


@pytest.mark.parametrize("limit", [0.0, 1.0, None])
def test_rough_cut_no_candidates_matches_legacy_json(
    monkeypatch: pytest.MonkeyPatch, limit: float | None
) -> None:
    shots, assets = _day(5)
    # Include an awkward full-window rounding case from the previous behaviour.
    shots[0].quality.usable_range = (0.0004, 4.0004)
    shots[0].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=180)
    actual = rough_cut(shots, assets, params, max_shift_s=limit).model_dump_json()
    monkeypatch.setattr("vlogcut.plan.rules._place_clip", lambda *args, **kwargs: None)
    legacy = rough_cut(shots, assets, params).model_dump_json()
    assert actual.encode() == legacy.encode()


def test_place_on_cut_points_no_candidates_returns_default() -> None:
    assert place_on_cut_points(100, 110, 3, 103.125, []) == 103.125


@pytest.mark.parametrize("candidate,expected", [(107.25, 104.25), (104.25, 104.25)])
def test_place_on_cut_points_snaps_end_or_start(candidate: float, expected: float) -> None:
    start = place_on_cut_points(100, 110, 3, 104, [candidate])
    assert start == expected
    assert candidate in (start, start + 3)


@pytest.mark.parametrize("candidates", [[99, 111], [102, 108], [99, 102, 108, 111]])
def test_place_on_cut_points_ignores_infeasible_placements(candidates: list[float]) -> None:
    # Interior candidates also fail: neither a start nor an end can fit 9 s.
    assert place_on_cut_points(100, 110, 9, 100.5, candidates) == 100.5


@pytest.mark.parametrize("limit,expected", [(None, 105), (0, 103), (1.5, 103), (2, 105)])
def test_place_on_cut_points_max_shift(limit: float | None, expected: float) -> None:
    assert place_on_cut_points(100, 110, 3, 103, [108], max_shift_s=limit) == expected


def test_place_on_cut_points_zero_shift_allows_default_candidate() -> None:
    assert place_on_cut_points(100, 110, 3, 103, [106], max_shift_s=0) == 103


@pytest.mark.parametrize("candidates", [[106, 108, 104], [104, 108, 106]])
def test_place_on_cut_points_tie_prefers_both_ends(candidates: list[float]) -> None:
    # Starts 102 and 104 tie in distance, but only [104, 108] aligns both ends.
    assert place_on_cut_points(100, 110, 4, 103, candidates) == 104


@pytest.mark.parametrize("candidates", [[104, 102], [102, 104]])
def test_place_on_cut_points_tie_prefers_earlier_start(candidates: list[float]) -> None:
    assert place_on_cut_points(100, 110, 4, 103, candidates) == 102


def test_place_on_cut_points_smallest_shift_beats_both_ends() -> None:
    assert place_on_cut_points(100, 110, 4, 103, [103.25, 104, 108]) == 103.25


def test_place_on_cut_points_preserves_length_and_input() -> None:
    candidates = [108, 102, 107.5, 102]
    original = candidates.copy()
    length = 3.5
    start = place_on_cut_points(100, 110, length, 103.5, candidates)
    end = start + length
    assert (start, end) == (104, 107.5)
    assert end - start == length
    assert 100 <= start < end <= 110
    assert candidates == original
    assert place_on_cut_points(100, 110, length, 103.5, tuple(candidates)) == start


@pytest.mark.parametrize("candidate", [100, 110])
def test_place_on_cut_points_accepts_window_boundaries(candidate: float) -> None:
    assert place_on_cut_points(100, 110, 10, 100, [candidate]) == 100


@pytest.mark.parametrize("candidate,expected", [(-5e-10, -5e-10), (10 + 5e-10, 7 + 5e-10)])
def test_place_on_cut_points_window_tolerance(candidate: float, expected: float) -> None:
    start = place_on_cut_points(0, 10, 3, 4, [candidate])
    assert start == expected
    assert -1e-9 <= start and start + 3 <= 10 + 1e-9


def test_place_on_cut_points_short_window_within_tolerance() -> None:
    assert place_on_cut_points(0, 3 - 5e-10, 3, 0, [0]) == 0


@pytest.mark.parametrize(
    "lo,hi,length,limit,message",
    [
        (0, 10, 0, None, "length_s must be positive"),
        (0, 10, -1, None, "length_s must be positive"),
        (0, 2, 3, None, "window is shorter"),
        (0, 3 - 2e-9, 3, None, "window is shorter"),
        (10, 0, 3, None, "window is shorter"),
        (0, 10, 3, -0.1, "max_shift_s must be non-negative"),
    ],
)
def test_place_on_cut_points_invalid_arguments(
    lo: float, hi: float, length: float, limit: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        place_on_cut_points(lo, hi, length, 0, [], max_shift_s=limit)


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


def test_apply_montage_late_runs_split_at_talking_and_preserve_inputs() -> None:
    pace = DurationStats(p10=1, median=4, p90=8)
    base, caps = [4.0] * 12, [8.0] * 12
    talking = [i == 8 for i in range(12)]
    base[8] = caps[8]
    before = (base.copy(), caps.copy(), talking.copy())
    shaped = apply_montage(base, caps, talking, pace, 0.5, 52)
    fitted = fit_lengths(shaped, [pace.p10] * 12, caps, 52)
    assert [i for i, length in enumerate(fitted) if length < 2] == [5, 6, 7, 9, 10, 11]
    assert montage_share([fitted], pace.median) == 0.5
    assert sum(fitted) == pytest.approx(52)
    assert shaped[8] == caps[8]
    assert (base, caps, talking) == before
    assert shaped == apply_montage(tuple(base), tuple(caps), tuple(talking), pace, 0.5, 52)


@pytest.mark.parametrize("share", [None, 0, 0.2, 0.25])
def test_apply_montage_too_few_requested_or_unmeasured_is_unchanged(share: float | None) -> None:
    base = [4.0] * 10  # round(2.5) == 2
    pace = DurationStats(p10=1, median=4, p90=8)
    assert apply_montage(base, [8] * 10, [False] * 10, pace, share, 40) == base


@pytest.mark.parametrize("talking", [[True] * 9, [False, False, True] * 3])
def test_apply_montage_without_eligible_run_is_unchanged(talking: list[bool]) -> None:
    base = [4.0] * 9
    pace = DurationStats(p10=1, median=4, p90=8)
    assert apply_montage(base, [8] * 9, talking, pace, 0.5, 36) == base


def test_apply_montage_empty_and_unrepresentable_short_lengths() -> None:
    pace = DurationStats(p10=1, median=2.001, p90=4)
    assert apply_montage([], [], [], pace, 1, 10) == []
    assert apply_montage([2] * 6, [4] * 6, [False] * 6, pace, 0.5, 12) == [2] * 6


@pytest.mark.parametrize("target,expected_count", [(30, 4), (32, 3), (36, 0)])
def test_apply_montage_reduces_count_when_fitting_would_erase_bursts(
    target: float, expected_count: int
) -> None:
    pace = DurationStats(p10=1, median=4, p90=4)
    base = [3.0] * 10
    shaped = apply_montage(base, [4] * 10, [False] * 10, pace, 0.6, target)
    fitted = fit_lengths(shaped, [1] * 10, [4] * 10, target)
    assert sum(length < 2 for length in fitted) == expected_count
    assert sum(fitted) == pytest.approx(target)
    if expected_count == 0:
        assert shaped == base


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("intro_ratio", [None, 0.5, 2.0])
@pytest.mark.parametrize("cold_open", [False, True])
def test_rough_cut_montage_bursts_improve_share_and_preserve_invariants(
    strategy: str, intro_ratio: float | None, cold_open: bool
) -> None:
    shots, assets = _day()
    shots[20].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=60, strategy=strategy)
    pace = DurationStats(p10=0.75, median=4, p90=6)
    profile = StyleProfile(
        shot_duration_s=pace,
        cuts_per_min=15,
        cold_open_clips=1,
        structure=StructureStats(montage_share=0.4, intro_pace_ratio=intro_ratio),
    )
    plain_profile = profile.model_copy(
        update={"structure": StructureStats(montage_share=0, intro_pace_ratio=intro_ratio)}
    )
    plain = rough_cut(shots, assets, params, plain_profile)
    edl = rough_cut(shots, assets, params, profile, cold_open=cold_open)
    durations = [c.timeline_duration_s for c in edl.clips]
    assert montage_share([durations], pace.median) >= 3 / len(durations)
    before, after = profile_from_edl(plain).structure, profile_from_edl(edl).structure
    assert before is not None and after is not None
    assert abs(after.montage_share - 0.4) < abs(before.montage_share - 0.4)
    assert abs(after.montage_share - 0.4) <= 2 / len(durations)
    assert edl.total_duration_s == pytest.approx(params.target_s, rel=0.1)
    assert all(pace.p10 <= length <= pace.p90 for length in durations)
    ids = [c.shot_id for c in edl.clips]
    original = [c.shot_id for c in plain.clips]
    assert set(ids) == set(original)
    body = ids[1:] if cold_open else ids
    assert body == sorted(body, key=original.index)
    talk = next(c for c in edl.clips if c.shot_id == shots[20].shot_id)
    assert talk.in_s == 0 and talk.timeline_duration_s == pace.p90
    assert (
        edl.model_dump_json()
        == rough_cut(shots, assets, params, profile, cold_open=cold_open).model_dump_json()
    )


@pytest.mark.parametrize(
    "structure",
    [
        None,
        StructureStats(montage_share=0).model_copy(update={"montage_share": None}),
        StructureStats(montage_share=0),
    ],
)
@pytest.mark.parametrize("with_profile", [False, True])
def test_rough_cut_disabled_montage_is_byte_identical_to_legacy(
    monkeypatch: pytest.MonkeyPatch, structure: StructureStats | None, with_profile: bool
) -> None:
    shots, assets = _day()
    params = RoughCutParams(target_s=60)
    profile = (
        StyleProfile(
            shot_duration_s=DurationStats(p10=0.75, median=4, p90=6),
            cuts_per_min=15,
            structure=structure,
        )
        if with_profile
        else None
    )
    actual = rough_cut(shots, assets, params, profile).model_dump_json().encode()
    # Bypass the new helper to reproduce the previous planner, including no profile.
    monkeypatch.setattr("vlogcut.plan.rules.apply_montage", lambda base, *args: list(base))
    legacy = profile.model_copy(update={"structure": None}) if profile else None
    assert actual == rough_cut(shots, assets, params, legacy).model_dump_json().encode()


@pytest.mark.parametrize(
    "p10,share,target,talking",
    [
        (2, 0.4, 60, False),
        (2.5, 0.4, 60, False),
        (1, 0.1, 60, False),
        (1, 0.4, 10, False),
        (1, 0.4, 60, True),
    ],
)
def test_rough_cut_impossible_montage_is_byte_identical(
    p10: float, share: float, target: float, talking: bool
) -> None:
    shots, assets = _day()
    if talking:
        for shot in shots:
            shot.quality.warnings = ["kept_for_audio"]
    profile = StyleProfile(shot_duration_s=DurationStats(p10=p10, median=4, p90=6), cuts_per_min=15)
    params = RoughCutParams(target_s=target)
    plain = rough_cut(shots, assets, params, profile).model_dump_json().encode()
    profile.structure = StructureStats(montage_share=share)
    assert rough_cut(shots, assets, params, profile).model_dump_json().encode() == plain


def test_apply_intro_pace_uses_unscaled_starts_and_preserves_inputs() -> None:
    base, floors, caps = [8.0, 7.0, 5.0, 6.0, 4.0], [1.0] * 5, [10.0] * 5
    before = (base.copy(), floors.copy(), caps.copy())
    # The third clip is still intro; the fourth starts exactly at 20 s.
    assert apply_intro_pace(base, floors, caps, 0.5) == [4.0, 3.5, 2.5, 6.0, 4.0]
    assert (base, floors, caps) == before


@pytest.mark.parametrize("ratio,expected", [(0.1, [2, 3, 5]), (2.0, [12, 11, 5])])
def test_apply_intro_pace_clamps_per_clip(ratio: float, expected: list[float]) -> None:
    assert apply_intro_pace([10, 10, 5], [2, 3, 1], [12, 11, 8], ratio) == expected


@pytest.mark.parametrize("base", [[], [25.0], [10.0, 9.0], [10.0, 10.0]])
def test_apply_intro_pace_without_body_is_unchanged(base: list[float]) -> None:
    assert apply_intro_pace(base, [1.0] * len(base), [30.0] * len(base), 0.5) == base


def test_apply_intro_pace_neutral_ratio_and_window_override() -> None:
    base = [8.0, 7.0, 5.0, 6.0]
    floors, caps = [1.0] * 4, [10.0] * 4
    assert apply_intro_pace(base, floors, caps, 1.0) == base
    assert apply_intro_pace(base, floors, caps, 0.5, intro_s=0) == base
    assert apply_intro_pace(base, floors, caps, 0.5, intro_s=8) == [4.0, 7.0, 5.0, 6.0]


@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_intro_pace_keeps_audio_bases_at_cap_before_fit(
    monkeypatch: pytest.MonkeyPatch, ratio: float
) -> None:
    picked = [_candidate(i, None) for i in range(12)]
    # Audio clips count toward intro starts, but must not be scaled themselves.
    for i in (0, 6, 11):
        picked[i].shot.quality.warnings = ["kept_for_audio"]
        picked[i].hi = 4.0
    calls: list[tuple[list[float], list[float], list[float], float]] = []

    def capture_fit(
        base: list[float], floors: list[float], caps: list[float], target: float
    ) -> list[float]:
        calls.append((base.copy(), floors.copy(), caps.copy(), target))
        return fit_lengths(base, floors, caps, target)

    monkeypatch.setattr("vlogcut.plan.rules.fit_lengths", capture_fit)
    _lengths(picked, DEFAULT_PACING, 40)
    _lengths(picked, DEFAULT_PACING, 40, ratio)
    plain, shaped = calls
    assert shaped[1:] == plain[1:]
    for i in (0, 6, 11):
        assert shaped[0][i] == plain[0][i] == 4.0
    assert shaped[0] != plain[0]


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
def test_rough_cut_intro_pace_moves_measured_ratio_toward_profile(strategy: str) -> None:
    shots, assets = _day()
    shots[20].quality.warnings = ["kept_for_audio"]  # first chronological shot
    params = RoughCutParams(target_s=60, strategy=strategy)
    profile = StyleProfile(
        shot_duration_s=DEFAULT_PACING,
        cuts_per_min=20,
        structure=StructureStats(montage_share=0, intro_pace_ratio=0.5),
    )
    plain = rough_cut(shots, assets, params)
    shaped = rough_cut(shots, assets, params, profile)
    before = profile_from_edl(plain).structure
    after = profile_from_edl(shaped).structure
    assert before is not None and before.intro_pace_ratio is not None
    assert after is not None and after.intro_pace_ratio is not None
    assert abs(after.intro_pace_ratio - 0.5) < abs(before.intro_pace_ratio - 0.5)
    assert shaped.total_duration_s == pytest.approx(params.target_s, rel=0.1)
    assert [c.shot_id for c in shaped.clips] == [c.shot_id for c in plain.clips]
    assert shaped.clips[0].in_s == plain.clips[0].in_s == 0
    assert shaped.clips[0].timeline_duration_s == DEFAULT_PACING.p90
    for a, b in zip(plain.clips, shaped.clips, strict=True):
        assert a.model_dump(exclude={"in_s", "out_s"}) == b.model_dump(exclude={"in_s", "out_s"})


@pytest.mark.parametrize("strategy", ["coverage", "highlights"])
@pytest.mark.parametrize("structure", [None, StructureStats(montage_share=0)])
def test_rough_cut_unmeasured_intro_matches_profileless_json(
    monkeypatch: pytest.MonkeyPatch, strategy: str, structure: StructureStats | None
) -> None:
    shots, assets = _day()
    shots[20].quality.warnings = ["kept_for_audio"]
    params = RoughCutParams(target_s=60, strategy=strategy)
    profile = StyleProfile(shot_duration_s=DEFAULT_PACING, cuts_per_min=20, structure=structure)

    def unexpected_scaling(*args: object, **kwargs: object) -> list[float]:
        pytest.fail("unmeasured intro pacing must bypass scaling")

    monkeypatch.setattr("vlogcut.plan.rules.apply_intro_pace", unexpected_scaling)
    plain = rough_cut(shots, assets, params).model_dump_json().encode()
    assert rough_cut(shots, assets, params, profile).model_dump_json().encode() == plain


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
