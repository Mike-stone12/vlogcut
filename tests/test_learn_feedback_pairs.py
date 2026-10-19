import logging

import pytest

from vlogcut.learn import score, shot_pairs_from_preferences, train_pairwise
from vlogcut.schemas import EDL, Clip, PreferencePair, QualityFlags, Shot


def _shot(sid: str, interest: int = 3) -> Shot:
    return Shot(
        shot_id=sid,
        asset_id="asset",
        start_s=0,
        end_s=4,
        interest=interest,
        quality=QualityFlags(sharpness=100, exposure_clip=0, shake=0),
    )


def _edl(*ids: str) -> EDL:
    # Mapping keys, not this internal label, identify the EDL file stems.
    return EDL(
        variant_id="internal-label",
        target_duration_s=len(ids),
        clips=[Clip(shot_id=sid, in_s=0, out_s=1) for sid in ids],
    )


def _preference(chosen: str = "a", rejected: str = "b") -> PreferencePair:
    return PreferencePair(context="daily vlog", chosen=chosen, rejected=rejected)


def _ids(pairs: list[tuple[Shot, Shot]]) -> list[tuple[str, str]]:
    return [(chosen.shot_id, skipped.shot_id) for chosen, skipped in pairs]


def test_difference_cross_product_excludes_shared_shots_without_mutation() -> None:
    shots = [_shot(sid) for sid in ("c1", "c2", "s1", "s2", "shared")]
    edls = {"a": _edl("c1", "shared", "c2"), "b": _edl("s1", "shared", "s2")}
    preferences = [_preference()]
    before = [item.model_dump() for item in [*shots, *edls.values(), *preferences]]

    result = shot_pairs_from_preferences(preferences, edls, shots)

    assert _ids(result) == [("c1", "s1"), ("c1", "s2"), ("c2", "s1"), ("c2", "s2")]
    assert result[0][0] is shots[0] and result[0][1] is shots[2]
    assert [item.model_dump() for item in [*shots, *edls.values(), *preferences]] == before


@pytest.mark.parametrize("missing", [("a",), ("b",), ("a", "b")])
def test_missing_edls_warn_once_per_id_and_continue(
    missing: tuple[str, ...], caplog: pytest.LogCaptureFixture
) -> None:
    edls = {"a": _edl("c"), "b": _edl("s"), "ok-a": _edl("c"), "ok-b": _edl("s")}
    for variant_id in missing:
        del edls[variant_id]
    preferences = [_preference(), _preference(), _preference("ok-a", "ok-b")]
    with caplog.at_level(logging.WARNING, logger="vlogcut.learn.feedback_pairs"):
        result = shot_pairs_from_preferences(preferences, edls, [_shot("c"), _shot("s")])
    assert _ids(result) == [("c", "s")]
    assert len(caplog.records) == len(missing)
    for record, variant_id in zip(caplog.records, missing, strict=True):
        assert record.levelno == logging.WARNING
        assert repr(variant_id) in record.getMessage()


def test_unknown_shots_ignored_on_both_sides() -> None:
    edls = {"a": _edl("unknown-c", "c"), "b": _edl("s", "unknown-s")}
    assert _ids(shot_pairs_from_preferences([_preference()], edls, [_shot("c"), _shot("s")])) == [
        ("c", "s")
    ]


def test_duplicate_preferences_deduplicated_but_reverse_orientation_retained() -> None:
    edls = {"a": _edl("c"), "b": _edl("s")}
    shots = [_shot("c"), _shot("s")]
    preferences = [_preference(), _preference()]
    assert _ids(shot_pairs_from_preferences(preferences, edls, shots)) == [("c", "s")]
    preferences.append(_preference("b", "a"))
    assert _ids(shot_pairs_from_preferences(preferences, edls, shots)) == [("c", "s"), ("s", "c")]


def test_first_clip_and_preference_order_with_repeated_clips_and_overlapping_pairs() -> None:
    shots = [_shot(sid) for sid in ("c1", "c2", "s1", "s2", "s3")]
    edls = {
        "a": _edl("c2", "c1", "c2"),
        "b": _edl("s2", "s1", "s2"),
        "d": _edl("s3", "s2"),
    }
    preferences = [_preference(), _preference("a", "d")]
    expected = [("c2", "s2"), ("c2", "s1"), ("c1", "s2"), ("c1", "s1"),
                ("c2", "s3"), ("c1", "s3")]  # fmt: skip
    assert _ids(shot_pairs_from_preferences(preferences, edls, shots)) == expected
    assert _ids(shot_pairs_from_preferences(preferences, edls, list(reversed(shots)))) == expected


def test_quality_rejected_shots_retained() -> None:
    shots = [_shot("c"), _shot("s")]
    for shot in shots:
        shot.quality.rejected = True
    result = shot_pairs_from_preferences([_preference()], {"a": _edl("c"), "b": _edl("s")}, shots)
    assert _ids(result) == [("c", "s")]


@pytest.mark.parametrize("chosen,rejected", [("c", "c"), ("c", "unknown"), ("unknown", "s")])
def test_no_exclusive_known_shots_yields_no_signal(chosen: str, rejected: str) -> None:
    edls = {"a": _edl(chosen), "b": _edl(rejected)}
    assert shot_pairs_from_preferences([_preference()], edls, [_shot("c"), _shot("s")]) == []


def test_empty_preferences_or_shots() -> None:
    assert shot_pairs_from_preferences([], {}, []) == []
    assert shot_pairs_from_preferences([_preference()], {"a": _edl("c"), "b": _edl("s")}, []) == []


def test_derived_pairs_train_existing_ranker() -> None:
    chosen, skipped = _shot("interesting", interest=5), _shot("dull", interest=1)
    shots = [chosen, skipped, _shot("shared"), _shot("other-dull", interest=2)]
    edls = {
        "a": _edl("interesting", "shared"),
        "b": _edl("dull", "shared"),
        "d": _edl("other-dull", "shared"),
    }
    pairs = shot_pairs_from_preferences([_preference(), _preference("a", "d")], edls, shots)
    model = train_pairwise(pairs)
    assert score(model, chosen) > score(model, skipped)
    assert score(model, chosen) > score(model, shots[3])
