"""Rule-based chronological rough cut: no model, just shots + pacing -> EDL.

Week-1 baseline and, later, the fallback when the planner's EDL cannot be repaired.

1. drop rejected shots; only use each shot's `usable_range`
2. order chronologically (asset creation time, then file name, then shot start)
3. pick shots: "coverage" = every recording (a moment of the day) gets a clip, the rest
   by how much usable footage it has, best shot per chronological bin; "highlights" =
   the k best overall, then re-sorted by time; replace adjacent duplicate-group
   picks when a chronological, conflict-free alternative exists; after count fitting,
   swap eligible shots toward measured kind alternation without reordering clips
4. sample the profile's length distribution, using its mean to choose k; assign long
   samples to shots with room, shape intro pacing and montage bursts, then fit with
   one shared offset clamped to p10..p90 and each shot's usable part; adjust k to fit
5. place each clip in the middle of the usable part; shots kept for their audio
   (someone talking) start at the beginning of the usable part and run as long as the
   pacing allows, so speech is less likely to be cut mid-sentence
6. shift the rounded, fixed-length windows onto nearby cut-point candidates
7. optionally move one later hook to the front when the profile allows a cold open

`score` is pluggable so highlight features (faces, audio peaks, ...) can replace the
quality-only default later.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from itertools import groupby
from typing import Literal

from pydantic import BaseModel, Field

from vlogcut.analyze.kind import classify_kind
from vlogcut.schemas import EDL, Clip, DurationStats, MediaAsset, Shot, StyleProfile, Transition
from vlogcut.style.profile import kind_alternation

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
        return classify_kind(self.shot) == "talking"


def score_shot(shot: Shot) -> float:
    """Quality-only default score in [0, 1]: sharp, steady, warning-free shots win."""
    q = shot.quality
    score = 0.5 + 0.2 * min(q.sharpness / 200.0, 1.0) - 0.2 * min(q.shake / 10.0, 1.0)
    soft = [w for w in q.warnings if w != "kept_for_audio"]
    score -= 0.1 * len(soft)
    if classify_kind(shot) == "talking":
        score += 0.3  # someone talking: usually the spine of a vlog
    return max(0.0, min(1.0, score))


def _ranker_percentiles(
    shot_ids: Sequence[str], ranker_scores: Mapping[str, float]
) -> dict[str, float]:
    """Rank finite scores among eligible IDs, with neutral defaults for missing IDs.

    Divide zero-based average ranks by n - 1, giving [0, 1] percentiles.
    A single scored shot gets 0.5; missing scores do not enter the ranking.
    """
    result = dict.fromkeys(shot_ids, 0.5)
    scored = sorted(
        (shot_id for shot_id in shot_ids if shot_id in ranker_scores),
        key=ranker_scores.__getitem__,
    )
    if len(scored) > 1:
        start = 0
        for _, group in groupby(scored, key=ranker_scores.__getitem__):
            tied = list(group)
            percentile = (start + (len(tied) - 1) / 2) / (len(scored) - 1)
            result.update(dict.fromkeys(tied, percentile))
            start += len(tied)
    return result


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


def apply_intro_pace(
    base: Sequence[float],
    floors: Sequence[float],
    caps: Sequence[float],
    ratio: float,
    intro_s: float = 20.0,
) -> list[float]:
    """Multiply intro bases by ratio, clamping each to its own floor/cap.

    Whole clips starting strictly before intro_s on the unscaled base timeline
    form the intro. Bases sample one pacing distribution, so multiplication is
    a simple relative bias (1.0 is neutral within bounds), not an exact median
    match after clamping, fitting and boundary shifts. Body clips stay unchanged;
    without both intro and body clips, return an unchanged copy. Inputs have
    equal lengths, positive bases/ratio, and ordered floor/cap pairs.
    """
    start = 0.0
    n_intro = 0
    for length in base:
        if start < intro_s:
            n_intro += 1
        start += length
    if n_intro == 0 or n_intro == len(base):
        return list(base)
    return [
        min(cap, max(floor, length * ratio)) if i < n_intro else length
        for i, (length, floor, cap) in enumerate(zip(base, floors, caps, strict=True))
    ]


def apply_montage(
    base: Sequence[float],
    caps: Sequence[float],
    talking: Sequence[bool],
    pace: DurationStats,
    share: float | None,
    target: float,
) -> list[float]:
    """Shape bases into late montage runs, leaving bounds and talking bases intact.

    Inputs follow timeline order and have equal lengths; caps are at least p10.
    Aim for round(share * k) clips in runs of >=3, filling eligible runs from the
    end backwards (their suffixes first), favouring the body over the intro.
    Talking clips split runs. Use p10 rounded up plus 1 ms for short bases, and
    spread any target deficit over non-burst, non-talking headroom proportionally.
    The unchanged shared-offset fit must leave bursts 1 ms below median/2, so
    in/out rounding cannot erase them. Try fewer burst clips if capacity prevents
    this; if even three cannot fit, return the original bases. Missing/zero share,
    too few requested clips, or no representable short duration are exact no-ops.
    """
    wanted = round(share * len(base)) if share else 0
    short = math.ceil(pace.p10 * 1000) / 1000 + 0.001
    limit = pace.median / 2 - 0.001
    if wanted < 3 or short >= limit:
        return list(base)
    runs: list[list[int]] = []
    run: list[int] = []
    for i, (cap, is_talking) in enumerate(zip(caps, talking, strict=True)):
        if not is_talking and cap >= short:
            run.append(i)
        else:
            if len(run) >= 3:
                runs.append(run)
            run = []
    if len(run) >= 3:
        runs.append(run)
    floors = [pace.p10] * len(base)
    for count in range(min(wanted, sum(map(len, runs))), 2, -1):
        burst: set[int] = set()
        for run in reversed(runs):
            take = min(len(run), count - len(burst))
            if take >= 3:
                burst.update(run[-take:])
        shaped = [
            short if i in burst else min(cap, max(pace.p10, length))
            for i, (length, cap) in enumerate(zip(base, caps, strict=True))
        ]
        room = [
            cap - length if i not in burst and not talking[i] else 0.0
            for i, (length, cap) in enumerate(zip(shaped, caps, strict=True))
        ]
        fraction = min(1.0, max(0.0, target - sum(shaped)) / sum(room)) if sum(room) else 0.0
        shaped = [length + fraction * extra for length, extra in zip(shaped, room, strict=True)]
        fitted = fit_lengths(shaped, floors, caps, target)
        if all(fitted[i] <= limit for i in burst):
            return shaped
    return list(base)


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


def place_on_cut_points(
    lo_s: float,
    hi_s: float,
    length_s: float,
    default_in_s: float,
    candidates: Sequence[float],
    max_shift_s: float | None = None,
) -> float:
    """Place a fixed-length clip using absolute source times from Shot.cut_points.

    The caller supplies the absolute usable window and a feasible default in-point.
    Only position changes, never clip length. Feasibility allows 1e-9 s tolerance.
    Choose the smallest shift within max_shift_s, if given; ties prefer both ends
    on candidates, then the earlier in-point. With no qualifying placement, return
    the default unchanged.
    """
    tolerance = 1e-9
    if length_s <= 0:
        raise ValueError("length_s must be positive")
    if hi_s - lo_s < length_s - tolerance:
        raise ValueError("window is shorter than length_s")
    if max_shift_s is not None and max_shift_s < 0:
        raise ValueError("max_shift_s must be non-negative")

    times = set(candidates)
    placements = {
        start
        for c in times
        for start in (c, c - length_s)
        if lo_s - tolerance <= start and start + length_s <= hi_s + tolerance
        if max_shift_s is None or abs(start - default_in_s) <= max_shift_s
    }
    return min(
        placements,
        key=lambda start: (
            abs(start - default_in_s),
            not (start in times and start + length_s in times),
            start,
        ),
        default=default_in_s,
    )


def assign_transitions(
    lengths: Sequence[float], dissolve_share: float | None, duration_s: float = 0.5
) -> list[Transition]:
    """Evenly place dissolves over boundaries, capped by final timeline lengths.

    Lengths are positive, the measured share is in [0, 1], and duration_s is positive.
    The first clip always enters with a cut.
    """
    transitions = [Transition() for _ in lengths]
    if dissolve_share is None or len(lengths) < 2:
        return transitions
    boundaries = len(lengths) - 1
    n = round(dissolve_share * boundaries)
    for j in range(n):
        i = math.floor((j + 0.5) * boundaries / n) + 1
        transitions[i] = Transition(
            kind="dissolve", duration_s=min(duration_s, 0.5 * min(lengths[i - 1], lengths[i]))
        )
    return transitions


def apply_cold_open(
    clips: Sequence[Clip],
    shots_by_id: Mapping[str, Shot],
    *,
    score: Callable[[Shot], float] = score_shot,
) -> list[Clip]:
    """Move one hook to the front without mutating clips or shots.

    Prefer indices >= len(clips) // 2, then the remaining indices after zero.
    Within each pool, higher shot scores win, with ties going to later clips.
    Skip moves that create duplicate neighbours at either changed boundary.
    All clip IDs must exist in shots_by_id. Fewer than three clips stay unchanged.
    """
    if len(clips) < 3:
        return list(clips)
    groups = [shots_by_id[c.shot_id].dup_group for c in clips]
    candidates = sorted(
        range(1, len(clips)),
        key=lambda i: (i >= len(clips) // 2, score(shots_by_id[clips[i].shot_id]), i),
        reverse=True,
    )
    for hook in candidates:
        if groups[hook] is not None and groups[hook] == groups[0]:
            continue
        if (
            hook + 1 < len(clips)
            and groups[hook - 1] is not None
            and groups[hook - 1] == groups[hook + 1]
        ):
            continue
        order = [hook, *range(hook), *range(hook + 1, len(clips))]
        return [
            clips[i].model_copy(update={"transition_in": Transition()})
            if i in (hook, 0, hook + 1)
            else clips[i]
            for i in order
        ]
    log.debug("Skipping cold open: no hook avoids duplicate neighbours")
    return list(clips)


def avoid_adjacent_duplicates(
    picked: Sequence[_Candidate],
    alternatives: Sequence[Sequence[_Candidate]],
) -> list[_Candidate]:
    """Replace conflicting later picks left to right, without mutating inputs.

    Picks are chronological with unique shot IDs and order values. Each position
    has an alternative pool (its coverage bin, or all candidates for highlights).
    Replacements must be unselected, between both neighbours, and conflict with
    neither. Prefer higher scores, then earlier order; keep the pick if none fits.
    None groups never conflict. The number of picks is preserved.
    """
    result = list(picked)
    selected = {c.shot.shot_id for c in result}
    for i in range(1, len(result)):
        previous, current = result[i - 1], result[i]
        group = previous.shot.dup_group
        if group is None or group != current.shot.dup_group:
            continue
        following = result[i + 1] if i + 1 < len(result) else None
        replacement = min(
            (
                c
                for c in alternatives[i]
                if c.shot.shot_id not in selected
                and previous.order < c.order
                and (following is None or c.order < following.order)
                and (
                    c.shot.dup_group is None
                    or (
                        c.shot.dup_group != group
                        and (following is None or c.shot.dup_group != following.shot.dup_group)
                    )
                )
            ),
            key=lambda c: (-c.score, c.order),
            default=None,
        )
        if replacement is not None:
            selected.remove(current.shot.shot_id)
            selected.add(replacement.shot.shot_id)
            result[i] = replacement
    return result


def apply_kind_alternation(
    picked: Sequence[_Candidate],
    eligible: Sequence[_Candidate],
    target: float,
    *,
    max_length_s: float = math.inf,
    min_total_s: float = 0.0,
) -> list[_Candidate]:
    """Greedily swap eligible shots without reordering or mutating the inputs.

    Use classified kinds and the style measurement's unknown-pair semantics.
    Each of at most len(picked) swaps strictly improves the absolute target gap.
    Prefer the replacement's blended score, then source start and shot ID; for
    the same replacement, prefer the smallest gap, then earlier positions.
    Preserve capped capacity for the caller's fit (or existing footage shortfall).
    """
    result = list(picked)
    kinds = [classify_kind(c.shot) for c in result]
    ranked = sorted(eligible, key=lambda c: (-c.score, c.shot.start_s, c.shot.shot_id))
    for _ in range(len(picked)):
        current = kind_alternation(kinds)
        if current is None:
            break
        selected = {c.shot.shot_id for c in result}
        capacity = sum(min(max_length_s, c.avail) for c in result)
        replacement: tuple[int, _Candidate] | None = None
        for candidate in ranked:
            best_gap = abs(current - target)
            kind = classify_kind(candidate.shot)
            if candidate.shot.shot_id in selected or kind is None:
                continue
            for i, original in enumerate(result):
                if kind == kinds[i]:
                    continue
                previous = result[i - 1] if i else None
                following = result[i + 1] if i + 1 < len(result) else None
                if (previous is not None and candidate.order <= previous.order) or (
                    following is not None and candidate.order >= following.order
                ):
                    continue
                group = candidate.shot.dup_group
                if group is not None and any(
                    neighbour is not None and neighbour.shot.dup_group == group
                    for neighbour in (previous, following)
                ):
                    continue
                new_capacity = (
                    capacity
                    - min(max_length_s, original.avail)
                    + min(max_length_s, candidate.avail)
                )
                if new_capacity < min(min_total_s, capacity):
                    continue
                trial = kinds.copy()
                trial[i] = kind
                measured = kind_alternation(trial)
                if measured is not None and abs(measured - target) < best_gap:
                    replacement = i, candidate
                    best_gap = abs(measured - target)
            if replacement is not None:
                break
        if replacement is None:
            break
        i, candidate = replacement
        result[i] = candidate
        kinds[i] = classify_kind(candidate.shot)
    return result


def select(cands: list[_Candidate], k: int, strategy: str) -> list[_Candidate]:
    """k candidates, returned in chronological order."""
    k = max(1, min(k, len(cands)))
    if strategy == "highlights":
        picked = sorted(sorted(cands, key=lambda c: (-c.score, c.order))[:k], key=lambda c: c.order)
        alternatives = [cands] * k
    else:
        alternatives = _coverage_bins(cands, k)
        picked = [max(b, key=lambda c: (c.score, -c.order)) for b in alternatives]
    return avoid_adjacent_duplicates(picked, alternatives)


def _coverage_bins(cands: list[_Candidate], k: int) -> list[list[_Candidate]]:
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
        return [sum(moments[i * m // k : (i + 1) * m // k], []) for i in range(k)]

    weights = [sum(c.avail for c in mo) for mo in moments]
    extra = k - len(moments)
    quotas = [1 + extra * w / sum(weights) for w in weights]
    alloc = [min(len(mo), int(q)) for mo, q in zip(moments, quotas, strict=True)]
    by_remainder = sorted(range(len(moments)), key=lambda i: alloc[i] - quotas[i])
    while sum(alloc) < k and any(a < len(mo) for a, mo in zip(alloc, moments, strict=True)):
        for i in by_remainder:
            if sum(alloc) < k and alloc[i] < len(moments[i]):
                alloc[i] += 1
    bins = []
    for mo, a in zip(moments, alloc, strict=True):
        n = len(mo)
        bins.extend(mo[j * n // a : (j + 1) * n // a] for j in range(a))
    return bins


def rough_cut(
    shots: Sequence[Shot],
    assets: Sequence[MediaAsset],
    params: RoughCutParams | None = None,
    profile: StyleProfile | None = None,
    score: Callable[[Shot], float] = score_shot,
    *,
    max_shift_s: float | None = 1.0,
    cold_open: bool = False,
    ranker_scores: Mapping[str, float] | None = None,
) -> EDL:
    """Build a rough cut, shifting onto candidates by at most max_shift_s (None: unbounded).

    cold_open opts into moving one later hook when profile.cold_open_clips permits it.
    Optional precomputed ranker_scores use higher = better. Known IDs must have
    finite values (even rejected/too-short shots); unknown IDs are ignored.
    Eligible scored candidates get within-table percentiles with average ranks
    for ties; single and missing scores get 0.5. Blend 0.5 * score(shot) +
    0.5 * percentile everywhere scores are used. score_shot already clamps to
    [0, 1], so no further normalization is needed; custom scorers should use
    that range when blending. None or empty scores preserve the original EDL.
    """
    params = params or RoughCutParams()
    pace = profile.shot_duration_s if profile else DEFAULT_PACING
    intro_ratio = (
        profile.structure.intro_pace_ratio if profile is not None and profile.structure else None
    )
    montage = profile.structure.montage_share if profile is not None and profile.structure else None
    alternation = (
        profile.structure.kind_alternation if profile is not None and profile.structure else None
    )
    by_id = {a.asset_id: a for a in assets}

    if ranker_scores:
        ranker_scores = {
            s.shot_id: ranker_scores[s.shot_id] for s in shots if s.shot_id in ranker_scores
        }
        for shot_id, value in ranker_scores.items():
            if not math.isfinite(value):
                raise ValueError(f"Ranker score for shot {shot_id!r} must be finite")

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

    if ranker_scores:
        percentiles = _ranker_percentiles([c.shot.shot_id for c in cands], ranker_scores)
        cands = [
            c.model_copy(update={"score": 0.5 * c.score + 0.5 * percentiles[c.shot.shot_id]})
            for c in cands
        ]

    target, tol = params.target_s, params.tolerance
    k = max(1, round(target / expected_length(pace)))
    seen: set[int] = set()
    while True:
        picked = select(cands, k, params.strategy)
        if alternation is None:
            lengths = _lengths(picked, pace, target, intro_ratio, montage)
            total = sum(lengths)
        else:
            # Count fitting needs only bounds; assign lengths after the final swaps.
            total = min(
                max(target, len(picked) * pace.p10), sum(min(pace.p90, c.avail) for c in picked)
            )
        seen.add(len(picked))
        if total < target * (1 - tol) and len(picked) < len(cands) and len(picked) + 1 not in seen:
            k = len(picked) + 1
        elif total > target * (1 + tol) and len(picked) > 1 and len(picked) - 1 not in seen:
            k = len(picked) - 1
        else:
            break

    if alternation is not None:
        picked = apply_kind_alternation(
            picked, cands, alternation, max_length_s=pace.p90, min_total_s=target * (1 - tol)
        )
        lengths = _lengths(picked, pace, target, intro_ratio, montage)
        total = sum(lengths)

    clips = [
        _clip(c, length, i, len(picked), params.strategy)
        for i, (c, length) in enumerate(zip(picked, lengths, strict=True))
    ]
    mix = profile.transitions if profile is not None else None
    transitions = assign_transitions(
        [c.timeline_duration_s for c in clips], mix.dissolve if mix is not None else None
    )
    for clip, transition in zip(clips, transitions, strict=True):
        clip.transition_in = transition
    for clip, candidate in zip(clips, picked, strict=True):
        _place_clip(clip, candidate, max_shift_s=max_shift_s)
    if cold_open and profile is not None and profile.cold_open_clips >= 1:
        scores = {c.shot.shot_id: c.score for c in picked}
        clips = apply_cold_open(
            clips,
            {c.shot.shot_id: c.shot for c in picked},
            score=lambda shot: scores[shot.shot_id],
        )
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


def _lengths(
    picked: list[_Candidate],
    pace: DurationStats,
    target: float,
    intro_ratio: float | None = None,
    montage: float | None = None,
) -> list[float]:
    caps = [min(pace.p90, c.avail) for c in picked]
    floors = [min(pace.p10, cap) for cap in caps]
    base = assign_lengths(
        sample_lengths(pace, len(picked)),
        caps,
        [c.score for c in picked],
        [c.talking for c in picked],
    )
    if intro_ratio is not None:
        base = apply_intro_pace(base, floors, caps, intro_ratio)
        # Talking clips keep their assigned cap before the existing shared-offset fit.
        for i, c in enumerate(picked):
            if c.talking:
                base[i] = caps[i]
    if montage:
        base = apply_montage(base, caps, [c.talking for c in picked], pace, montage, target)
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


def _place_clip(clip: Clip, c: _Candidate, *, max_shift_s: float | None = 1.0) -> None:
    # Preserve the existing rounded length; candidates use absolute source times,
    # whereas Clip stores offsets from the shot start. Do not round again afterward.
    length = clip.out_s - clip.in_s
    if length > c.avail + 1e-9:
        # Legacy rounding can enlarge a full-window clip; no fixed-length move fits.
        return
    origin = c.shot.start_s
    default = origin + clip.in_s
    start = place_on_cut_points(
        origin + c.lo,
        origin + c.hi,
        length,
        default,
        [point.time_s for point in c.shot.cut_points],
        max_shift_s=max_shift_s,
    )
    if start != default:
        # Contain the helper's 1e-9 feasibility tolerance at the usable boundaries.
        clip.in_s = min(max(start - origin, c.lo), c.hi - length)
        clip.out_s = clip.in_s + length
