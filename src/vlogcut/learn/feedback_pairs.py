"""Derive shot-level training preferences from variant selections, without I/O."""

import logging
from collections.abc import Mapping, Sequence

from vlogcut.schemas import EDL, PreferencePair, Shot

logger = logging.getLogger(__name__)


def shot_pairs_from_preferences(
    pairs: Sequence[PreferencePair], edls: Mapping[str, EDL], shots: Sequence[Shot]
) -> list[tuple[Shot, Shot]]:
    """Return chosen-only × rejected-only Shot pairs for ranker training.

    EDL mapping keys are variant IDs (EDL file stems). Missing EDLs skip the
    preference, warning once per missing ID per call; unknown shots are ignored.
    Preserve preference order, then first-clip order for chosen and skipped shots.
    Deduplicate ordered shot-ID pairs globally, but retain both orientations of
    conflicting preferences: the ranker's loss handles that conflict.
    Quality-rejected shots are retained; callers decide whether to filter them.
    Inputs are not mutated, and returned tuples reference the supplied Shots.
    """
    by_id = {shot.shot_id: shot for shot in shots}
    warned: set[str] = set()
    seen: set[tuple[str, str]] = set()
    result: list[tuple[Shot, Shot]] = []
    for pair in pairs:
        chosen_edl = edls.get(pair.chosen)
        rejected_edl = edls.get(pair.rejected)
        for variant_id, edl in ((pair.chosen, chosen_edl), (pair.rejected, rejected_edl)):
            if edl is None and variant_id not in warned:
                logger.warning("Missing EDL for variant %r; skipping preference pair", variant_id)
                warned.add(variant_id)
        if chosen_edl is None or rejected_edl is None:
            continue

        chosen_ids = dict.fromkeys(clip.shot_id for clip in chosen_edl.clips)
        rejected_ids = dict.fromkeys(clip.shot_id for clip in rejected_edl.clips)
        chosen_only = [sid for sid in chosen_ids if sid not in rejected_ids and sid in by_id]
        skipped_only = [sid for sid in rejected_ids if sid not in chosen_ids and sid in by_id]
        for chosen_id in chosen_only:
            for skipped_id in skipped_only:
                key = (chosen_id, skipped_id)
                if key not in seen:
                    result.append((by_id[chosen_id], by_id[skipped_id]))
                    seen.add(key)
    return result
