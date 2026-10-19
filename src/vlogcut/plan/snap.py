"""Pure beat snapping by local edits to consecutive clips; no music analysis."""

from bisect import bisect_left
from collections.abc import Sequence

from vlogcut.schemas import EDL, Clip, Shot


def snap_to_beats(
    edl: EDL,
    shots: Sequence[Shot],
    beats: Sequence[float],
    *,
    tolerance_s: float = 0.08,
    min_clip_s: float = 0.25,
) -> EDL:
    """Snap internal boundaries left to right, preserving all other timeline positions.

    Beats are programme timeline seconds; the caller applies any music cue delay.
    Choose the nearest beat within inclusive tolerance (1e-12 s floating-point
    allowance), preferring the earlier beat on ties. Move the outgoing out-point
    and incoming in-point by the timeline delta times each clip's speed.

    Skip edits exceeding shot-relative bounds, minimum timeline length or adjacent
    dissolve fit, and boundaries touching unknown shots. Inputs are never modified;
    even a no-op returns a new EDL. Music analysis and pipeline wiring are separate.
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    if min_clip_s <= 0:
        raise ValueError("min_clip_s must be positive")
    result = edl.model_copy(deep=True)
    if not len(beats) or len(result.clips) < 2:
        return result

    ordered = sorted(beats)
    by_id = {shot.shot_id: shot for shot in shots}
    clips = result.clips
    boundary = 0.0
    for i in range(1, len(clips)):
        left, right = clips[i - 1], clips[i]
        boundary += left.timeline_duration_s
        left_shot, right_shot = by_id.get(left.shot_id), by_id.get(right.shot_id)
        if left_shot is None or right_shot is None:
            continue
        index = bisect_left(ordered, boundary)
        neighbors = ordered[max(0, index - 1) : index + 1]
        beat = min(neighbors, key=lambda t: (abs(t - boundary), t))
        # Symmetric decimal beats can differ by roundoff after subtraction.
        if abs(neighbors[0] - boundary) <= abs(beat - boundary) + 1e-12:
            beat = neighbors[0]
        delta = beat - boundary
        if abs(delta) > tolerance_s + 1e-12:
            continue

        outgoing = left.model_copy(update={"out_s": left.out_s + delta * left.speed})
        incoming = right.model_copy(update={"in_s": right.in_s + delta * right.speed})
        if not all(
            0 <= clip.in_s < clip.out_s <= shot.duration_s
            and clip.timeline_duration_s >= min_clip_s
            and _dissolves_fit(clips, j, clip.timeline_duration_s)
            for j, clip, shot in ((i - 1, outgoing, left_shot), (i, incoming, right_shot))
        ):
            continue
        clips[i - 1], clips[i] = outgoing, incoming
        # Carry the actual edited duration into the next cumulative position.
        boundary += outgoing.timeline_duration_s - left.timeline_duration_s
    return result


def _dissolves_fit(clips: Sequence[Clip], index: int, duration_s: float) -> bool:
    """Check both adjacent dissolves and their half-windows in the edited clip.

    Each affected clip checks its side of a dissolve; unchanged neighbours retain
    their existing fit. Cuts have zero duration in the shared schema.
    """
    incoming = clips[index].transition_in.duration_s
    outgoing = clips[index + 1].transition_in.duration_s if index + 1 < len(clips) else 0.0
    return (
        incoming <= duration_s
        and outgoing <= duration_s
        and (incoming + outgoing) / 2 <= duration_s
    )
