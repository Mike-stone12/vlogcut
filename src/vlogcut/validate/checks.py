"""Pure hard checks; violations are analysis results, not shared data contracts."""

from collections.abc import Sequence

from pydantic import BaseModel

from vlogcut.schemas import EDL, Shot, StyleProfile


class Violation(BaseModel):
    """One repairable problem; ``clip_index=None`` denotes an EDL-level check."""

    code: str
    clip_index: int | None
    message: str


def validate_edl(
    edl: EDL,
    shots: Sequence[Shot],
    profile: StyleProfile | None = None,
    *,
    target_s: float | None = None,
    duration_tolerance: float = 0.10,
    length_tolerance: float = 0.25,
) -> list[Violation]:
    """Return violations in clip order, followed by EDL-level violations.

    In/out are shot-relative offsets, checked with 1e-6 s boundary tolerance;
    quality.usable_range is only a suggestion and is not enforced. Durations use
    timeline seconds (including speed), unchanged by centred dissolves. Target
    and profile checks run only when supplied; edl.target_duration_s is not used.
    Duration and length tolerances are fractions of the target and profile bounds.

    Shot.dup_group is checked only for actual neighbours with known shot IDs.
    Chronology uses strictly increasing positions in the supplied shots, comparing
    successive known IDs across unknown ones after skipping the first
    profile.cold_open_clips clips (zero without a profile). The opening clips
    still receive all other checks. Audio has no profile dimension to check.
    Unknown IDs receive only unknown_shot; their declared timeline durations still
    contribute to the total and to neighbouring clips' dissolve geometry checks.
    Dissolve boundary errors belong to the incoming clip; window overlap errors
    belong to the clip containing the overlap. Inputs are never modified.
    """
    by_id = {shot.shot_id: (i, shot) for i, shot in enumerate(shots)}
    durations = [clip.timeline_duration_s for clip in edl.clips]
    violations: list[Violation] = []
    previous_position: int | None = None
    cold_open_clips = profile.cold_open_clips if profile is not None else 0

    def add(code: str, index: int | None, message: str) -> None:
        violations.append(Violation(code=code, clip_index=index, message=message))

    for i, clip in enumerate(edl.clips):
        entry = by_id.get(clip.shot_id)
        if entry is None:
            add("unknown_shot", i, f"Shot {clip.shot_id!r} is absent from shots")
            continue
        position, shot = entry
        if not (0 <= clip.in_s < clip.out_s <= shot.duration_s + 1e-6):
            add(
                "out_of_bounds",
                i,
                f"Shot {shot.shot_id!r}: offsets [{clip.in_s}, {clip.out_s}] s "
                f"exceed bounds [0, {shot.duration_s}] s (tolerance 1e-6 s)",
            )
        if shot.quality.rejected:
            add("rejected_shot", i, f"Shot {shot.shot_id!r} has quality.rejected=True")
        duration = durations[i]
        if profile is not None:
            lower = profile.shot_duration_s.p10 * (1 - length_tolerance)
            upper = profile.shot_duration_s.p90 * (1 + length_tolerance)
            if duration < lower:
                add("clip_too_short", i, f"Timeline duration {duration} s < minimum {lower} s")
            if duration > upper:
                add("clip_too_long", i, f"Timeline duration {duration} s > maximum {upper} s")
        previous = by_id.get(edl.clips[i - 1].shot_id) if i else None
        if previous is not None and shot.dup_group is not None:
            if previous[1].dup_group == shot.dup_group:
                add(
                    "duplicate_neighbours",
                    i,
                    f"Shots {previous[1].shot_id!r} and {shot.shot_id!r} "
                    f"share dup_group {shot.dup_group!r}",
                )
        if i >= cold_open_clips:
            if previous_position is not None and position <= previous_position:
                add(
                    "not_chronological",
                    i,
                    f"Shot {shot.shot_id!r} position {position} "
                    f"is not after previous known position {previous_position}",
                )
            previous_position = position

        transition = clip.transition_in
        if transition.kind == "dissolve":
            if profile is not None and profile.transitions is not None:
                if profile.transitions.dissolve == 0:
                    add(
                        "transition_not_allowed",
                        i,
                        f"Entry dissolve {transition.duration_s} s with profile dissolve share 0",
                    )
            if i and not (transition.duration_s <= min(durations[i - 1], duration)):
                add(
                    "dissolve_too_long",
                    i,
                    f"Entry dissolve {transition.duration_s} s exceeds neighbouring "
                    f"timeline durations {durations[i - 1]} s and {duration} s",
                )
        incoming = transition.duration_s / 2
        outgoing = edl.clips[i + 1].transition_in.duration_s / 2 if i + 1 < len(edl.clips) else 0
        if incoming + outgoing > duration:
            add(
                "dissolve_too_long",
                i,
                f"Dissolve halves {incoming} s + {outgoing} s "
                f"exceed clip timeline duration {duration} s",
            )

    if target_s is not None:
        lower, upper = target_s * (1 - duration_tolerance), target_s * (1 + duration_tolerance)
        total = edl.total_duration_s
        if not lower <= total <= upper:
            add(
                "total_duration",
                None,
                f"Timeline total {total} s outside [{lower}, {upper}] s for target {target_s} s",
            )
    return violations
