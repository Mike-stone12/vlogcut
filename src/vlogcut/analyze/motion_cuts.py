"""Pure cut candidates from sampled camera motion; no video decoding."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from statistics import median

from vlogcut.schemas import CutPoint


def camera_motion_cut_points(
    motion: Sequence[tuple[float, float]], sample_fps: float, start_s: float, end_s: float
) -> list[CutPoint]:
    """Convert per-step (dy, dx) translations into absolute-time candidates.

    Assign each step's magnitude to its destination frame, at (i + 1) / fps.
    Extend the first magnitude back to frame zero so it cannot create a minimum.
    This reuses the quality analyzer's phase-correlation signal without decoding.
    """
    levels = [math.hypot(dy, dx) for dy, dx in motion]
    return motion_cut_points(levels[:1] + levels, sample_fps, start_s, end_s)


def motion_cut_points(
    motion: Sequence[float],
    sample_fps: float,
    start_s: float,
    end_s: float,
    *,
    min_gap_s: float = 0.5,
    edge_margin_s: float = 0.1,
    max_relative_level: float = 0.5,
) -> list[CutPoint]:
    """Find quiet motion minima at absolute source-asset times.

    Sample i lies at start_s + i / sample_fps. A minimum (including a flat
    plateau) must have strictly higher neighbours on both sides; use its first
    sample. The default threshold is half the supplied shot signal's median,
    selecting dips well below typical motion. Bounds plus the 100 ms margin
    are strict. Greedily keep lowest minima first, earlier on ties, suppressing
    others less than min_gap_s away (exactly 0.5 s apart is allowed by default).
    Empty, constant and fewer-than-three-sample signals have no internal minima.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    if not all(math.isfinite(v) for v in (start_s, end_s)) or not 0 <= start_s <= end_s:
        raise ValueError("bounds must be finite, nonnegative and ordered")
    for name, value in (
        ("min_gap_s", min_gap_s),
        ("edge_margin_s", edge_margin_s),
        ("max_relative_level", max_relative_level),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    levels = [float(value) for value in motion]
    if any(not math.isfinite(value) or value < 0 for value in levels):
        raise ValueError("motion values must be finite and nonnegative")
    if len(levels) < 3:
        return []

    threshold = max_relative_level * median(levels)
    candidates: list[tuple[float, float]] = []
    i = 1
    while i < len(levels) - 1:
        first = i
        while i + 1 < len(levels) and levels[i + 1] == levels[first]:
            i += 1
        time_s = start_s + first / sample_fps
        if (
            i + 1 < len(levels)
            and levels[first] < levels[first - 1]
            and levels[first] < levels[i + 1]
            and levels[first] <= threshold
            and start_s + edge_margin_s < time_s < end_s - edge_margin_s
        ):
            candidates.append((levels[first], time_s))
        i += 1

    kept: set[float] = set()
    for _, time_s in sorted(candidates):
        if all(abs(time_s - other) >= min_gap_s for other in kept):
            kept.add(time_s)
    return [CutPoint(time_s=t, kind="motion") for t in sorted(kept)]


def merge_cut_points(
    a: Sequence[CutPoint], b: Sequence[CutPoint], *, tolerance_s: float = 0.1
) -> list[CutPoint]:
    """Merge candidates for the same shot, preferring pauses within 100 ms.

    Remove every motion candidate within tolerance_s (inclusive) of any pause.
    Keep distinct times of the same kind, deduplicate exact times, and sort.
    Inputs may be unsorted; neither input is mutated. Callers supply candidates
    inside the same shot, since this helper has no shot bounds to check.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    points = [*a, *b]
    if any(not math.isfinite(p.time_s) or p.time_s < 0 for p in points):
        raise ValueError("cut point times must be finite and nonnegative")
    pauses = sorted({p.time_s for p in points if p.kind == "pause"})
    merged = {t: CutPoint(time_s=t, kind="pause") for t in pauses}
    for point in points:
        if point.kind == "motion":
            index = bisect_left(pauses, point.time_s)
            neighbours = pauses[max(0, index - 1) : index + 1]
            if all(abs(point.time_s - t) > tolerance_s for t in neighbours):
                merged[point.time_s] = CutPoint(time_s=point.time_s, kind="motion")
    return [merged[t] for t in sorted(merged)]
