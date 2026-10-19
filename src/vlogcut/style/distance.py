"""Distance between style profiles, and the profile a planned cut would measure as.

Pacing dimensions (shot-duration p10 / median / p90, cuts per minute, intro/body) are positive ratio
quantities: a 1 s shot next to a 2 s one is as different as 4 s next to 8 s. Each is
compared as |log2(a / b)|, so the unit is octaves and 1.0 means twice as fast or twice as
slow. Ratio dimensions in [0, 1] (`beat_aligned_ratio`, `speech_ratio`, `dissolve`,
`montage_share`, `kind_alternation`, look brightness and saturation) use the absolute difference.
Look warmth in [-1, 1] uses half the absolute difference to normalize to [0, 1].
A dimension that either side has not
measured (`None`) is skipped and listed
in `StyleDistance.unmeasured`. Dimensions are equally weighted until the eval set says
otherwise. Pure functions, no numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from vlogcut.analyze.kind import classify_kind
from vlogcut.schemas import EDL, Clip, Shot, StyleProfile, TransitionMix
from vlogcut.style.profile import kind_alternation, profile_from_durations

LOG_DIMS = ("p10", "median", "p90", "cuts_per_min", "intro_pace")  # |log2 ratio|
RATIO_DIMS = (  # |difference|
    "beat_aligned_ratio", "speech_ratio", "dissolve", "montage_share", "kind_alternation",
)  # fmt: skip
LOOK_DIMS = ("brightness", "saturation", "warmth")
_LABELS = {
    "p10": "p10 shot s",
    "median": "median shot s",
    "p90": "p90 shot s",
    "cuts_per_min": "cuts/min",
    "beat_aligned_ratio": "beat-aligned",
    "speech_ratio": "speech ratio",
    "dissolve": "dissolve",
    "montage_share": "montage_share",
    "intro_pace": "intro_pace",
    "kind_alternation": "kind_alternation",
    "brightness": "brightness",
    "saturation": "saturation",
    "warmth": "warmth",
}


class StyleDistance(BaseModel):
    """How far apart two profiles are; an analysis result, not a pipeline contract."""

    total: float = Field(ge=0)  # mean of per_dim; octaves while only pacing is measured
    per_dim: dict[str, float]
    unmeasured: list[str] = Field(default_factory=list)  # None on at least one side


def dimensions(profile: StyleProfile) -> dict[str, float | None]:
    """The comparable scalars of a profile, in table order."""
    s = profile.shot_duration_s
    return {
        "p10": s.p10,
        "median": s.median,
        "p90": s.p90,
        "cuts_per_min": profile.cuts_per_min,
        "beat_aligned_ratio": profile.beat_aligned_ratio,
        "speech_ratio": profile.speech_ratio,
        "dissolve": profile.transitions.dissolve if profile.transitions is not None else None,
        "montage_share": (
            profile.structure.montage_share if profile.structure is not None else None
        ),
        "intro_pace": (
            profile.structure.intro_pace_ratio if profile.structure is not None else None
        ),
        "kind_alternation": (
            profile.structure.kind_alternation if profile.structure is not None else None
        ),
        "brightness": profile.look.brightness if profile.look is not None else None,
        "saturation": profile.look.saturation if profile.look is not None else None,
        "warmth": profile.look.warmth if profile.look is not None else None,
    }


def profile_distance(a: StyleProfile, b: StyleProfile) -> StyleDistance:
    va, vb = dimensions(a), dimensions(b)
    per_dim: dict[str, float] = {}
    unmeasured: list[str] = []
    for dim in (*LOG_DIMS, *RATIO_DIMS, *LOOK_DIMS):
        x, y = va[dim], vb[dim]
        if x is None or y is None:
            unmeasured.append(dim)
        elif dim in LOG_DIMS:
            per_dim[dim] = abs(math.log2(x / y))
        elif dim == "warmth":
            per_dim[dim] = abs(x - y) / 2
        else:
            per_dim[dim] = abs(x - y)
    total = sum(per_dim.values()) / len(per_dim)  # the pacing dims are always measured
    return StyleDistance(total=total, per_dim=per_dim, unmeasured=unmeasured)


def speech_ratio(
    clips: Sequence[Clip], kinds: Mapping[str, Literal["talking", "broll"] | None]
) -> float | None:
    """Talking timeline duration / known-kind duration; unknown IDs are excluded."""
    talking_s, known_s = 0.0, 0.0
    for clip in clips:
        kind = kinds.get(clip.shot_id)
        if kind is not None:
            duration_s = clip.timeline_duration_s
            known_s += duration_s
            if kind == "talking":
                talking_s += duration_s
    return min(1.0, max(0.0, talking_s / known_s)) if known_s else None


def profile_from_edl(
    edl: EDL, source: str = "edl", *, shots: Sequence[Shot] | None = None
) -> StyleProfile:
    """Measure pacing, structure and transitions; shots enable alternation and speech ratio."""
    profile = profile_from_durations([[c.timeline_duration_s for c in edl.clips]], [source])
    dissolve = sum(c.transition_in.kind == "dissolve" for c in edl.clips) / (len(edl.clips) - 1)
    profile.transitions = TransitionMix(hard_cut=1 - dissolve, dissolve=dissolve)
    if shots:
        kinds = {shot.shot_id: classify_kind(shot) for shot in shots}
        profile.speech_ratio = speech_ratio(edl.clips, kinds)
        # Successful pacing measurement always includes the required montage share.
        assert profile.structure is not None
        profile.structure.kind_alternation = kind_alternation(
            [kinds.get(clip.shot_id) for clip in edl.clips]
        )
    return profile


# ---------------------------------------------------------------------------
# text tables for `vlogcut style show / diff`
# ---------------------------------------------------------------------------


def format_profile(name: str, profile: StyleProfile) -> str:
    rows = [f"{name}: {profile.notes}" if profile.notes else name]
    for dim, value in dimensions(profile).items():
        if value is not None:
            rows.append(f"{_LABELS[dim]:<14}{value:>9.2f}")
    if profile.music_bpm is not None:
        rows.append(f"{'music BPM':<14}{profile.music_bpm:>9.1f}")
    mix = profile.transitions
    rows.append(
        "transitions   "
        + (
            f"hard cut {mix.hard_cut:.1%}, dissolve {mix.dissolve:.1%}"
            if mix is not None
            else "not measured"
        )
    )
    if profile.source_refs:
        rows.append("sources       " + ", ".join(profile.source_refs))
    return "\n".join(rows)


def format_diff(
    a_name: str, a: StyleProfile, b_name: str, b: StyleProfile, d: StyleDistance
) -> str:
    va, vb = dimensions(a), dimensions(b)
    w = max(len(a_name), len(b_name), 9)
    rows = [f"{'':<14}{a_name:>{w}}  {b_name:>{w}}  {'delta':>8}"]
    for dim, delta in d.per_dim.items():
        rows.append(f"{_LABELS[dim]:<14}{va[dim]:>{w}.2f}  {vb[dim]:>{w}.2f}  {delta:>8.2f}")
    pacing_only = all(dim in LOG_DIMS for dim in d.per_dim)
    unit = "octaves" if pacing_only else "octaves (pacing) and ratio points, mixed"
    rows.append(f"distance      {d.total:.2f} {unit}; mean over {len(d.per_dim)} dims")
    if d.unmeasured:
        rows.append("not measured on both sides: " + ", ".join(d.unmeasured))
    return "\n".join(rows)
