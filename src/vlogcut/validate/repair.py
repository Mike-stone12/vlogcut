"""Pure repair of unknown/rejected selections and forbidden or oversized dissolves.

Preserve retained clip order, in/out points, speeds, audio and EDL metadata.
Bounds, chronology, duplicates, target duration and pacing are not repaired;
no new selections, beat snapping, model calls or pipeline wiring are involved.
"""

from collections.abc import Sequence

from vlogcut.schemas import EDL, Shot, StyleProfile, Transition


def repair_edl(edl: EDL, shots: Sequence[Shot], profile: StyleProfile | None = None) -> EDL:
    """Return an independent EDL, repairing transitions left to right.

    Raise ValueError if no known, unrejected clips remain. Dissolves use the
    exact timeline limits in validate_edl, with no extra tolerance. The EDL's
    computed total follows the retained clips; its requested target is preserved.
    """
    by_id = {shot.shot_id: shot for shot in shots}
    result = edl.model_copy(deep=True)
    result.clips = [
        clip
        for clip in result.clips
        if clip.shot_id in by_id and not by_id[clip.shot_id].quality.rejected
    ]
    if not result.clips:
        raise ValueError("Cannot repair EDL: no known, unrejected clips remain")

    result.clips[0].transition_in = Transition()
    forbid_dissolves = (
        profile is not None
        and profile.transitions is not None
        and profile.transitions.dissolve == 0
    )
    for i in range(1, len(result.clips)):
        left, right = result.clips[i - 1], result.clips[i]
        if right.transition_in.kind != "dissolve":
            continue
        if forbid_dissolves:
            right.transition_in = Transition()
            continue
        # Reserve the already repaired incoming half-window in the left clip.
        # The right clip's outgoing window will be resolved on the next pass.
        duration = min(
            right.transition_in.duration_s,
            left.timeline_duration_s,
            right.timeline_duration_s,
            2 * (left.timeline_duration_s - left.transition_in.duration_s / 2),
        )
        right.transition_in = (
            Transition(kind="dissolve", duration_s=duration) if duration > 0 else Transition()
        )
    return result
