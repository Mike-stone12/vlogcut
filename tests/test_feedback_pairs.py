from __future__ import annotations

import pytest

from vlogcut.feedback import preference_pairs
from vlogcut.schemas import Feedback, PreferencePair


def test_pairs_order_context_and_inputs_unchanged() -> None:
    feedback = [
        Feedback(variant_id="r2", verdict="reject"),
        Feedback(variant_id="p2", verdict="pick", text="Good pace"),
        Feedback(variant_id="r1", verdict="reject"),
        Feedback(variant_id="p1", verdict="pick"),
    ]
    before = [entry.model_copy(deep=True) for entry in feedback]
    pairs = preference_pairs(feedback, context="A day at the cafe")
    assert pairs == [
        PreferencePair(context="A day at the cafe", chosen="p2", rejected="r2"),
        PreferencePair(context="A day at the cafe", chosen="p2", rejected="r1"),
        PreferencePair(context="A day at the cafe", chosen="p1", rejected="r2"),
        PreferencePair(context="A day at the cafe", chosen="p1", rejected="r1"),
    ]
    assert feedback == before


def test_text_only_entries_are_ignored() -> None:
    feedback = (
        Feedback(variant_id="p1", text="pick"),
        Feedback(variant_id="r", verdict="reject"),
        Feedback(variant_id="p2", verdict="pick"),
        Feedback(variant_id="p1", verdict="pick"),
        Feedback(variant_id="p2", text="reject"),
        Feedback(variant_id="other", text="pick"),
    )
    assert preference_pairs(feedback) == [
        PreferencePair(context="", chosen="p2", rejected="r"),
        PreferencePair(context="", chosen="p1", rejected="r"),
    ]


def test_identical_verdicts_are_deduplicated() -> None:
    feedback = [
        Feedback(variant_id="p2", verdict="pick"),
        Feedback(variant_id="r2", verdict="reject"),
        Feedback(variant_id="p1", verdict="pick"),
        Feedback(variant_id="p2", verdict="pick", text="Still prefer this"),
        Feedback(variant_id="r1", verdict="reject"),
        Feedback(variant_id="r2", verdict="reject"),
    ]
    assert [(pair.chosen, pair.rejected) for pair in preference_pairs(feedback)] == [
        ("p2", "r2"),
        ("p2", "r1"),
        ("p1", "r2"),
        ("p1", "r1"),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_verdicts_raise(reverse: bool) -> None:
    variant_id = "coverage_180s_fast\nvariant"
    feedback = [
        Feedback(variant_id=variant_id, verdict="pick"),
        Feedback(variant_id=variant_id, verdict="reject"),
    ]
    if reverse:
        feedback.reverse()
    before = [entry.model_copy(deep=True) for entry in feedback]
    with pytest.raises(ValueError, match="coverage_180s_fast") as exc:
        preference_pairs(feedback)
    assert len(str(exc.value).splitlines()) == 1
    assert feedback == before


@pytest.mark.parametrize(
    "feedback",
    [
        [],
        [Feedback(variant_id="p", verdict="pick")],
        [Feedback(variant_id="r", verdict="reject")],
        [Feedback(variant_id="text", text="Looks good")],
    ],
)
def test_missing_picks_or_rejects_returns_empty(feedback: list[Feedback]) -> None:
    assert preference_pairs(feedback) == []
