"""Style profile from references: pacing, structure, transitions, beats and look.

PoC slice: the shot-duration distribution (p10 / median / p90) and cuts per minute,
pooled over references, plus montage-burst share, intro/body pacing, transitions, beats and look.
Other structure dimensions and speech come later.

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
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from vlogcut.analyze import look, shots, transitions
from vlogcut.ingest import make_proxy, probe_asset
from vlogcut.paths import from_workspace, to_workspace_relative
from vlogcut.schemas import DurationStats, LookStats, StructureStats, StyleProfile, TransitionMix

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
INTRO_S = 20.0
LOOK_MAX_FRAMES = 32


class ReferenceParams(BaseModel):
    """Detection knobs; part of the cache key."""

    threshold: float = Field(default=3.0, gt=0)  # AdaptiveDetector adaptive_threshold
    # finished edits have real 0.3 s montage shots; the footage default (0.5 s) would merge
    # them away and inflate p10 for fast-cutting creators
    min_shot_s: float = Field(default=0.25, gt=0)
    dissolves: bool = True  # also find cross-dissolves / fades (analyze.transitions)
    beats: bool = True  # measure alignment and tempo of the reference's own audio
    look: bool = True  # sample RGB appearance from the proxy


class ReferenceCuts(BaseModel):
    """Raw cut times of one reference; cached, since detection is the expensive step."""

    asset_id: str
    # Path as given (for StyleProfile.source_refs); workspace-relative in cache JSON
    # when it lies inside the workspace.
    source: str
    duration_s: float = Field(gt=0)
    params: ReferenceParams
    cuts: list[float]  # hard cuts + dissolve midpoints, seconds
    dissolve_cuts: list[float]  # required: old caches must be recomputed
    beats: list[float]  # sorted reference-timeline seconds; required for cache invalidation
    look: LookStats | None = None

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


def montage_share(
    per_ref: Sequence[Sequence[float]], median_s: float, rel: float = 0.5, min_run: int = 3
) -> float | None:
    """Pool shot counts in short-shot bursts, preserving each edit's timeline order.

    Short means strictly below `rel * median_s`, using the pooled profile median.
    Maximal runs of at least `min_run` qualify; runs never cross edit boundaries.
    """
    if median_s <= 0 or rel <= 0 or min_run < 2:
        raise ValueError("median_s and rel must be positive; min_run must be at least 2")
    threshold = rel * median_s
    total, in_bursts = 0, 0
    for durations in per_ref:
        total += len(durations)
        run = 0
        for duration in durations:
            if duration < threshold:
                run += 1
            else:
                if run >= min_run:
                    in_bursts += run
                run = 0
        if run >= min_run:
            in_bursts += run
    return in_bursts / total if total else None


def kind_alternation(kinds: Sequence[Literal["talking", "broll"] | None]) -> float | None:
    """Share of known adjacent pairs that differ; unknown kinds break adjacency."""
    eligible, differing = 0, 0
    for left, right in zip(kinds, kinds[1:], strict=False):
        if left is not None and right is not None:
            eligible += 1
            differing += left != right
    return differing / eligible if eligible else None


def intro_pace_ratio(
    sequences: Sequence[Sequence[float]], intro_s: float = INTRO_S
) -> float | None:
    """Ratio of pooled intro/body medians, assigning whole shots by their start time.

    Each timeline starts at zero; each pool needs at least two shots to be measured.
    """
    if intro_s <= 0:
        raise ValueError("intro_s must be positive")
    intro: list[float] = []
    body: list[float] = []
    for durations in sequences:
        start = 0.0
        for duration in durations:
            if start < intro_s:
                intro.append(duration)
            else:
                body.append(duration)
            start += duration
    if len(intro) < 2 or len(body) < 2:
        return None
    return statistics.median(intro) / statistics.median(body)


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


def pooled_bpm(beat_lists: Sequence[Sequence[float]]) -> float | None:
    """Tempo from pooled positive intervals, never crossing reference boundaries."""
    intervals: list[float] = []
    for beats in beat_lists:
        ordered = sorted(beats)
        intervals.extend(
            right - left for left, right in zip(ordered, ordered[1:], strict=False) if right > left
        )
    return 60 / statistics.median(intervals) if intervals else None


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
    music_bpm: float | None = None,
    look_stats: Sequence[LookStats | None] = (),
) -> StyleProfile:
    """Pool finished edits into one profile: a single duration distribution over all
    shots, and cuts per minute from the totals, so a short reference does not weigh as
    much as a long one. Each inner sequence is one edit's consecutive shot lengths (it
    sums to the edit's duration and has one cut less than it has shots).
    `n_dissolves` counts surviving dissolve boundaries; None means not measured.
    `look_stats` pools measured RGB means by frame count; no samples means None."""
    if not per_ref:
        raise StyleError("no references given")
    durations = [d for lengths in per_ref for d in lengths]
    n_cuts = len(durations) - len(per_ref)
    if n_cuts < 1:
        raise StyleError("no cuts: need at least two shots to describe pacing")
    total_s = sum(durations)
    stats = duration_stats(durations)
    share = montage_share(per_ref, stats.median)
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
        structure=(
            StructureStats(montage_share=share, intro_pace_ratio=intro_pace_ratio(per_ref))
            if share is not None
            else None
        ),
        beat_aligned_ratio=beat_ratio,
        music_bpm=music_bpm,
        look=look.pool_look(look_stats),
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
        music_bpm=(
            pooled_bpm([r.beats for r in refs])
            if params.beats and all(r.params.beats for r in refs)
            else None
        ),
        look_stats=(
            [r.look for r in refs] if params.look and all(r.params.look for r in refs) else ()
        ),
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
    workspace = Path(workspace).resolve()
    asset = make_proxy(probe_asset(path), workspace)
    cache_path = workspace / "style" / f"{asset.asset_id}.cuts.json"
    cached = _load_cache(cache_path, asset.asset_id, params, workspace)
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
    measured_look = None
    if params.look:
        # About 1 fps for short references; spread at most 32 samples over longer ones.
        sample_fps = max(1 / asset.duration_s, min(1.0, LOOK_MAX_FRAMES / asset.duration_s))
        frames = look.load_rgb_frames(
            asset.proxy_path, look.LookParams(sample_fps=sample_fps), max_frames=LOOK_MAX_FRAMES
        )
        measured_look = look.look_stats(frames)
    record = ReferenceCuts(
        asset_id=asset.asset_id, source=str(path), duration_s=asset.duration_s, params=params,
        cuts=cuts, dissolve_cuts=dissolve_cuts, beats=beats, look=measured_look,
    )  # fmt: skip
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".part")
    stored = record.model_copy(update={"source": to_workspace_relative(record.source, workspace)})
    tmp.write_text(stored.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(cache_path)
    return record


def build_profile(
    paths: Sequence[str | Path],
    workspace: str | Path,
    params: ReferenceParams | None = None,
    progress: Progress = log.info,
) -> StyleProfile:
    """Reference videos -> `StyleProfile`, including cached RGB look measurements."""
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


def _load_cache(
    path: Path, asset_id: str, params: ReferenceParams, workspace: Path
) -> ReferenceCuts | None:
    if not path.is_file():
        return None
    try:
        record = ReferenceCuts.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        return None
    if record.asset_id != asset_id or record.params != params:
        return None
    if params.beats and "beats" not in record.params.model_fields_set:
        return None
    if params.look and (
        "look" not in record.model_fields_set or "look" not in record.params.model_fields_set
    ):
        return None
    record.source = str(from_workspace(record.source, workspace))
    return record
