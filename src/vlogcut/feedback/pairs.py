"""Derive preference pairs from explicit verdicts for one run."""

from collections.abc import Sequence
from typing import Literal

from vlogcut.schemas import Feedback, PreferencePair


def preference_pairs(feedback: Sequence[Feedback], context: str = "") -> list[PreferencePair]:
    """Pair every pick with every reject in first-occurrence order, without mutation."""
    verdicts: dict[str, Literal["pick", "reject"]] = {}
    picks: list[str] = []
    rejects: list[str] = []
    for entry in feedback:
        if entry.verdict is None:
            continue
        previous = verdicts.get(entry.variant_id)
        if previous is not None:
            if previous != entry.verdict:
                raise ValueError(f"conflicting verdicts for variant {entry.variant_id!r}")
            continue
        verdicts[entry.variant_id] = entry.verdict
        if entry.verdict == "pick":
            picks.append(entry.variant_id)
        else:
            rejects.append(entry.variant_id)

    return [
        PreferencePair(context=context, chosen=chosen, rejected=rejected)
        for chosen in picks
        for rejected in rejects
    ]
