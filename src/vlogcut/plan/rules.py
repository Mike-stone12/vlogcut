"""Rule-based chronological rough cut: no model, just shots + pacing -> EDL.

Week-1 baseline and, later, the fallback when the planner's EDL cannot be repaired.

1. drop rejected shots; only use each shot's `usable_range`
2. order chronologically (asset creation time, then file name, then shot start)
3. pick shots: "coverage" = every recording (a moment of the day) gets a clip, the rest
   by how much usable footage it has, best shot per chronological bin; "highlights" =
   the k best overall, then re-sorted by time
4. sample the profile's length distribution, using its mean to choose k; assign long
   samples to shots with room, then fit to the target with one shared offset clamped
   to p10..p90 and each shot's usable part; grow / shrink k until within tolerance
5. place each clip in the middle of the usable part; shots kept for their audio
   (someone talking) start at the beginning of the usable part and run as long as the
   pacing allows, so speech is less likely to be cut mid-sentence

`score` is pluggable so highlight features (faces, audio peaks, ...) can replace the
quality-only default later.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from vlogcut.schemas import EDL, Clip, DurationStats, MediaAsset, Shot, StyleProfile

log = logging.getLogger(__name__)

DEFAULT_PACING = DurationStats(p10=1.5, median=3.0, p90=5.0)
_SEARCH_STEPS = 60


class RoughCutParams(BaseModel):
    target_s: float = Field(default=180.0, gt=0)
    strategy: Literal["coverage", "highlights"] = "coverage"
    tolerance: float = Field(default=0.1, gt=0, lt=1)  # validate/ accepts ±10 %


class RoughCutError(RuntimeError):
    pass


class _Candidate(BaseModel):
    shot: Shot
    lo: float  # usable part, offsets from shot start
    hi: float
    order: int
    score: float

    @property
    def avail(self) -> float:
        return self.hi - self.lo

    @property
    def talking(self) -> bool:
        return "kept_for_audio" in self.shot.quality.warnings


def score_shot(shot: Shot) -> float:
    """Quality-only default score in [0, 1]: sharp, steady, warning-free shots win."""
    q = shot.quality
    score = 0.5 + 0.2 * min(q.sharpness / 200.0, 1.0) - 0.2 * min(q.shake / 10.0, 1.0)
    soft = [w for w in q.warnings if w != "kept_for_audio"]
    score -= 0.1 * len(soft)
    if "kept_for_audio" in q.warnings:
        score += 0.3  # someone talking: usually the spine of a vlog
    return max(0.0, min(1.0, score))


def order_key(asset: MediaAsset | None) -> tuple:
    """Chronological key for an asset: creation time if known, else file name."""
    if asset is None:
        return (2, "", "")
    if asset.created_at is not None:
        return (0, asset.created_at.timestamp(), asset.path)
    return (1, asset.path, "")


def expected_length(pace: DurationStats) -> float:
    """Mean of the piecewise-linear quantile function with flat tails."""
    return 0.3 * pace.p10 + 0.4 * pace.median + 0.3 * pace.p90


def sample_lengths(pace: DurationStats, k: int) -> list[float]:
    """Sample Q(j / (k - 1)) in ascending order; a single clip gets the median."""
    if k == 1:
        return [pace.median]
    lengths = []
    for j in range(k):
        u = j / (k - 1)
        if u <= 0.1:
            length = pace.p10
        elif u <= 0.5:
            length = pace.p10 + (pace.median - pace.p10) * (u - 0.1) / 0.4
        elif u < 0.9:
            length = pace.median + (pace.p90 - pace.median) * (u - 0.5) / 0.4
        else:
            length = pace.p90
        lengths.append(length)
    return lengths


def assign_lengths(
    targets: Sequence[float],
    caps: Sequence[float],
    scores: Sequence[float],
    talking: Sequence[bool],
) -> list[float]:
    """Pair long targets with roomy, high-score shots; ties prefer earlier shots.

    Talking shots take their cap, replacing the nearest remaining target. Inputs
    have equal lengths and shots are chronological; output follows that order.
    """
    remaining = sorted(targets, reverse=True)
    base = list(caps)
    for i, is_talking in enumerate(talking):
        if is_talking:
            nearest = min(range(len(remaining)), key=lambda j: abs(remaining[j] - caps[i]))
            remaining.pop(nearest)
    order = sorted(
        (i for i in range(len(caps)) if not talking[i]),
        key=lambda i: (-caps[i], -scores[i], i),
    )
    for i, length in zip(order, remaining, strict=True):
        base[i] = length
    return base


def fit_lengths(base: Sequence[float], lo: Sequence[float], hi: Sequence[float], target: float):
    """Lengths l_i = clamp(base_i + t, lo_i, hi_i) with one offset t chosen so the total
    is as close to `target` as the bounds allow (bisection; total is monotone in t)."""

    def total(t: float) -> float:
        return sum(min(h, max(lw, b + t)) for b, lw, h in zip(base, lo, hi, strict=True))

    a, b = -max(base) - 1.0, max(hi) + 1.0
    for _ in range(_SEARCH_STEPS):
        mid = (a + b) / 2
        a, b = (mid, b) if total(mid) < target else (a, mid)
    t = (a + b) / 2
    return [min(h, max(lw, bb + t)) for bb, lw, h in zip(base, lo, hi, strict=True)]


def select(cands: list[_Candidate], k: int, strategy: str) -> list[_Candidate]:
    """k candidates, returned in chronological order."""
    k = max(1, min(k, len(cands)))
    if strategy == "highlights":
        picked = sorted(cands, key=lambda c: (-c.score, c.order))[:k]
    else:
        picked = _coverage(cands, k)
    return sorted(picked, key=lambda c: c.order)


def _coverage(cands: list[_Candidate], k: int) -> list[_Candidate]:
    """Every recording (asset) is a moment of the day. With enough slots each moment gets
    one clip and the rest go by usable footage (largest remainder), so a long one-take
    cannot crowd out the last scene; with fewer slots than moments, pick from k evenly
    spaced chronological bins of moments."""
    moments: list[list[_Candidate]] = []
    for c in cands:  # cands are chronological, so each asset is one contiguous run
        if moments and moments[-1][0].shot.asset_id == c.shot.asset_id:
            moments[-1].append(c)
        else:
            moments.append([c])
    if k < len(moments):
        m = len(moments)
        bins = [sum(moments[i * m // k : (i + 1) * m // k], []) for i in range(k)]
        return [max(b, key=lambda c: (c.score, -c.order)) for b in bins]

    weights = [sum(c.avail for c in mo) for mo in moments]
    extra = k - len(moments)
    quotas = [1 + extra * w / sum(weights) for w in weights]
    alloc = [min(len(mo), int(q)) for mo, q in zip(moments, quotas, strict=True)]
    by_remainder = sorted(range(len(moments)), key=lambda i: alloc[i] - quotas[i])
    while sum(alloc) < k and any(a < len(mo) for a, mo in zip(alloc, moments, strict=True)):
        for i in by_remainder:
            if sum(alloc) < k and alloc[i] < len(moments[i]):
                alloc[i] += 1
    picked = []
    for mo, a in zip(moments, alloc, strict=True):
        n = len(mo)
        bins = [mo[j * n // a : (j + 1) * n // a] for j in range(a)]
        picked += [max(b, key=lambda c: (c.score, -c.order)) for b in bins]
    return picked


def rough_cut(
    shots: Sequence[Shot],
    assets: Sequence[MediaAsset],
    params: RoughCutParams | None = None,
    profile: StyleProfile | None = None,
    score: Callable[[Shot], float] = score_shot,
) -> EDL:
    params = params or RoughCutParams()
    pace = profile.shot_duration_s if profile else DEFAULT_PACING
    by_id = {a.asset_id: a for a in assets}

    ordered = sorted(shots, key=lambda s: (order_key(by_id.get(s.asset_id)), s.asset_id, s.start_s))
    cands = []
    for i, shot in enumerate(ordered):
        if shot.quality.rejected:
            continue
        lo, hi = shot.quality.usable_range or (0.0, shot.duration_s)
        if hi - lo >= pace.p10:
            cands.append(_Candidate(shot=shot, lo=lo, hi=hi, order=i, score=score(shot)))
    if not cands:
        raise RoughCutError(
            f"no usable shots: {len(shots)} shots, none unrejected with >= {pace.p10:.1f} s"
        )

    target, tol = params.target_s, params.tolerance
    k = max(1, round(target / expected_length(pace)))
    seen: set[int] = set()
    while True:
        picked = select(cands, k, params.strategy)
        lengths = _lengths(picked, pace, target)
        total = sum(lengths)
        seen.add(len(picked))
        if total < target * (1 - tol) and len(picked) < len(cands) and len(picked) + 1 not in seen:
            k = len(picked) + 1
        elif total > target * (1 + tol) and len(picked) > 1 and len(picked) - 1 not in seen:
            k = len(picked) - 1
        else:
            break

    clips = [
        _clip(c, length, i, len(picked), params.strategy)
        for i, (c, length) in enumerate(zip(picked, lengths, strict=True))
    ]
    notes = (
        f"rules/{params.strategy}: {len(clips)} of {len(cands)} usable shots "
        f"({len(shots) - len(cands)} rejected or too short), {total:.1f} s for a "
        f"{target:.0f} s target"
    )
    if total < target * (1 - tol):
        notes += f"; footage too short, only {sum(c.avail for c in cands):.1f} s usable"
    log.info(notes)
    return EDL(
        variant_id=f"rules-{params.strategy}", target_duration_s=target, clips=clips, notes=notes
    )


def _lengths(picked: list[_Candidate], pace: DurationStats, target: float) -> list[float]:
    caps = [min(pace.p90, c.avail) for c in picked]
    floors = [min(pace.p10, cap) for cap in caps]
    base = assign_lengths(
        sample_lengths(pace, len(picked)),
        caps,
        [c.score for c in picked],
        [c.talking for c in picked],
    )
    return fit_lengths(base, floors, caps, target)


def _clip(c: _Candidate, length: float, i: int, n: int, strategy: str) -> Clip:
    start = c.lo if c.talking else c.lo + (c.avail - length) / 2
    why = "talking, kept whole" if c.talking else f"score {c.score:.2f}"
    where = f"bin {i + 1}/{n}" if strategy == "coverage" else f"top {n}"
    return Clip(
        shot_id=c.shot.shot_id,
        in_s=round(start, 3),
        out_s=min(round(start + length, 3), c.hi),
        reason=f"rules: {where}, {why}",
    )
