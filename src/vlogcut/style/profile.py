"""Style profile from reference videos: pacing, transitions and beat alignment.

PoC slice: the shot-duration distribution (p10 / median / p90) and cuts per minute,
pooled over all references, plus transition mix and beat alignment. Structure and speech
come later.

References are finished edits, so this does not reuse `analyze.detect_shots`: that step
splits long shots into equal parts (which would destroy the duration distribution) and
extracts keyframes the profile does not need. The raw cut times come from the same
detectors (`shots.detect_cuts` + `transitions`) and are cached per reference at
`<workspace>/style/<asset_id>.cuts.json`; everything after that is pure and cheap.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from vlogcut.analyze import shots, transitions
from vlogcut.ingest import make_proxy, probe_asset
from vlogcut.schemas import DurationStats, StyleProfile, TransitionMix

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


class ReferenceParams(BaseModel):
    """Detection knobs; part of the cache key."""

    threshold: float = Field(default=3.0, gt=0)  # AdaptiveDetector adaptive_threshold
    # finished edits have real 0.3 s montage shots; the footage default (0.5 s) would merge
    # them away and inflate p10 for fast-cutting creators
    min_shot_s: float = Field(default=0.25, gt=0)
    dissolves: bool = True  # also find cross-dissolves / fades (analyze.transitions)
    beats: bool = True  # measure alignment to the reference's own audio


class ReferenceCuts(BaseModel):
    """Raw cut times of one reference; cached, since detection is the expensive step."""

    asset_id: str
    source: str  # the path as given, for `StyleProfile.source_refs`
    duration_s: float = Field(gt=0)
    params: ReferenceParams
    cuts: list[float]  # hard cuts + dissolve midpoints, seconds
    dissolve_cuts: list[float]  # required: old caches must be recomputed
    beats: list[float]  # sorted reference-timeline seconds; required for cache invalidation

    @model_validator(mode="after")
    def _dissolves_are_cuts(self) -> ReferenceCuts:
        if not set(self.dissolve_cuts).issubset(self.cuts):
            raise ValueError("dissolve_cuts must be a subset of cuts")
        return self


class StyleError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# pure aggregation
# ---------------------------------------------------------------------------


def combine_cuts(
    hard_cuts: Sequence[float],
    dissolves: Sequence[tuple[float, float]],
    min_gap_s: float,
) -> tuple[list[float], list[float]]:
    """Merge detections and identify unique surviving dissolve boundaries."""
    cuts = transitions.merge_cuts(hard_cuts, dissolves, min_gap_s)
    # merge_cuts emits midpoints unchanged, so exact membership tracks provenance
    # without labelling nearby hard cuts as dissolves. Take values from its output.
    midpoints = {(start + end) / 2 for start, end in dissolves}
    dissolve_cuts = [cut for cut in dict.fromkeys(cuts) if cut in midpoints]
    return cuts, dissolve_cuts


def shot_durations(cuts: Sequence[float], duration_s: float, min_shot_s: float) -> list[float]:
    """Shot lengths between cuts; slivers shorter than `min_shot_s` join a neighbour.
    Long shots are kept whole (no `split_long`): their length *is* the style."""
    spans = shots.merge_short(shots.cuts_to_spans(list(cuts), duration_s), min_shot_s)
    return [end - start for start, end in spans]


def duration_stats(durations: Sequence[float]) -> DurationStats:
    if len(durations) < 2:
        raise StyleError("need at least two shots (one cut) to describe pacing")
    deciles = statistics.quantiles(durations, n=10, method="inclusive")
    return DurationStats(p10=deciles[0], median=statistics.median(durations), p90=deciles[-1])


def summarize(
    n_refs: int, total_s: float, n_cuts: int, stats: DurationStats, cuts_per_min: float
) -> str:
    refs = f"{n_refs} reference" + ("s" if n_refs != 1 else "")
    return (
        f"{refs}, {total_s / 60:.1f} min, {n_cuts} cuts; median shot {stats.median:.1f} s "
        f"(p10 {stats.p10:.1f}, p90 {stats.p90:.1f}); {cuts_per_min:.1f} cuts/min"
    )


def transition_mix(n_dissolves: int | None, n_cuts: int) -> TransitionMix | None:
    """Pooled shares of surviving cuts; missing detection or no cuts is unmeasured."""
    if n_dissolves is None or n_cuts == 0:
        return None
    dissolve = n_dissolves / n_cuts
    return TransitionMix(hard_cut=1 - dissolve, dissolve=dissolve)


def pooled_beat_ratio(
    refs: Sequence[ReferenceCuts], per_ref_cuts: Sequence[Sequence[float]]
) -> float | None:
    """Pool aligned / surviving cuts only over references with measured beats."""
    from vlogcut.analyze.beats import beat_aligned_ratio

    if any(not ref.params.beats for ref in refs):
        return None
    aligned, total = 0, 0
    for ref, cuts in zip(refs, per_ref_cuts, strict=True):
        ratio = beat_aligned_ratio(cuts, ref.beats)
        if ratio is not None:
            aligned += round(ratio * len(cuts))
            total += len(cuts)
    return aligned / total if total else None


def profile_from_durations(
    per_ref: Sequence[Sequence[float]],
    source_refs: Sequence[str],
    *,
    n_dissolves: int | None = None,
    beat_ratio: float | None = None,
) -> StyleProfile:
    """Pool finished edits into one profile: a single duration distribution over all
    shots, and cuts per minute from the totals, so a short reference does not weigh as
    much as a long one. Each inner sequence is one edit's consecutive shot lengths (it
    sums to the edit's duration and has one cut less than it has shots).
    `n_dissolves` counts surviving dissolve boundaries; None means not measured."""
    if not per_ref:
        raise StyleError("no references given")
    durations = [d for lengths in per_ref for d in lengths]
    n_cuts = len(durations) - len(per_ref)
    if n_cuts < 1:
        raise StyleError("no cuts: need at least two shots to describe pacing")
    total_s = sum(durations)
    stats = duration_stats(durations)
    cuts_per_min = n_cuts / (total_s / 60.0)
    mix = transition_mix(n_dissolves, n_cuts)
    notes = summarize(len(per_ref), total_s, n_cuts, stats, cuts_per_min)
    if mix is not None:
        notes += f"; {mix.dissolve:.1%} dissolves"
    if beat_ratio is not None:
        notes += f"; {beat_ratio:.1%} beat-aligned cuts"
    return StyleProfile(
        source_refs=list(source_refs),
        shot_duration_s=stats,
        cuts_per_min=cuts_per_min,
        transitions=mix,
        beat_aligned_ratio=beat_ratio,
        notes=notes,
    )


def aggregate(refs: Sequence[ReferenceCuts], params: ReferenceParams) -> StyleProfile:
    """Detected cuts of several references -> one pooled profile."""
    spans = [
        shots.merge_short(shots.cuts_to_spans(r.cuts, r.duration_s), params.min_shot_s)
        for r in refs
    ]
    n_dissolves = None
    if params.dissolves and all(r.params.dissolves for r in refs):
        # cuts_to_spans rounds to microseconds; only surviving interior boundaries
        # count toward both cuts/min and the transition mix.
        n_dissolves = sum(
            len({end for _, end in ss[:-1]} & {round(c, 6) for c in r.dissolve_cuts})
            for r, ss in zip(refs, spans, strict=True)
        )
    return profile_from_durations(
        [[end - start for start, end in ss] for ss in spans],
        [r.source for r in refs],
        n_dissolves=n_dissolves,
        beat_ratio=(
            pooled_beat_ratio(refs, [[end for _, end in ss[:-1]] for ss in spans])
            if params.beats
            else None
        ),
    )


# ---------------------------------------------------------------------------
# detection + cache
# ---------------------------------------------------------------------------


def analyze_reference(
    path: str | Path,
    workspace: str | Path,
    params: ReferenceParams | None = None,
    progress: Progress = log.info,
) -> ReferenceCuts:
    """Cut times of one reference (cached by asset fingerprint + params)."""
    params = params or ReferenceParams()
    workspace = Path(workspace)
    asset = make_proxy(probe_asset(path), workspace)
    cache_path = workspace / "style" / f"{asset.asset_id}.cuts.json"
    cached = _load_cache(cache_path, asset.asset_id, params)
    if cached is not None:
        log.debug("reusing reference cuts %s", cache_path)
        return cached.model_copy(update={"source": str(path)})

    progress(f"detecting cuts in {Path(path).name}")
    cuts = shots.detect_cuts(
        asset.proxy_path, threshold=params.threshold, min_scene_len_s=params.min_shot_s
    )
    dissolve_cuts = []
    if params.dissolves:
        found = transitions.find_dissolves(
            transitions.load_frames(asset.proxy_path), transitions.SAMPLE_FPS
        )
        cuts, dissolve_cuts = combine_cuts(cuts, found, params.min_shot_s)
    beats = []
    if params.beats and asset.has_audio:
        from vlogcut.analyze.beats import BeatParams, detect_beats
        from vlogcut.analyze.pauses import load_mono_audio

        sample_rate = BeatParams().sample_rate
        samples = load_mono_audio(asset.proxy_path, sample_rate)
        if samples is not None:
            beats = sorted(detect_beats(samples, sample_rate).beats)
    record = ReferenceCuts(
        asset_id=asset.asset_id, source=str(path), duration_s=asset.duration_s, params=params,
        cuts=cuts, dissolve_cuts=dissolve_cuts, beats=beats,
    )  # fmt: skip
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".part")
    tmp.write_text(record.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(cache_path)
    return record


def build_profile(
    paths: Sequence[str | Path],
    workspace: str | Path,
    params: ReferenceParams | None = None,
    progress: Progress = log.info,
) -> StyleProfile:
    """Reference videos -> `StyleProfile` (pacing, transition mix and beat alignment)."""
    params = params or ReferenceParams()
    refs = []
    for i, path in enumerate(paths, 1):
        progress(f"[{i}/{len(paths)}] reference {Path(path).name}")
        refs.append(analyze_reference(path, workspace, params, progress))
    return aggregate(refs, params)


def load_profile(path: str | Path) -> StyleProfile:
    """Read a profile JSON written by `vlogcut style` (or edited by hand).

    Invalid files raise `StyleError` with the first problem on one line, so callers can
    fail before any analysis starts.
    """
    path = Path(path)
    try:
        return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "json"
        raise StyleError(f"invalid style profile {path}: {where}: {first['msg']}") from exc


def _load_cache(path: Path, asset_id: str, params: ReferenceParams) -> ReferenceCuts | None:
    if not path.is_file():
        return None
    try:
        record = ReferenceCuts.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        return None
    if record.asset_id != asset_id or record.params != params:
        return None
    return record
