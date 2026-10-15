"""Style profile from reference videos: finished edits -> pacing statistics.

PoC slice: the shot-duration distribution (p10 / median / p90) and cuts per minute,
pooled over all references. Structure, transition, beat and speech dimensions come later.

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

from pydantic import BaseModel, Field, ValidationError

from vlogcut.analyze import shots, transitions
from vlogcut.ingest import make_proxy, probe_asset
from vlogcut.schemas import DurationStats, StyleProfile

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


class ReferenceParams(BaseModel):
    """Detection knobs; part of the cache key."""

    threshold: float = Field(default=3.0, gt=0)  # AdaptiveDetector adaptive_threshold
    # finished edits have real 0.3 s montage shots; the footage default (0.5 s) would merge
    # them away and inflate p10 for fast-cutting creators
    min_shot_s: float = Field(default=0.25, gt=0)
    dissolves: bool = True  # also find cross-dissolves / fades (analyze.transitions)


class ReferenceCuts(BaseModel):
    """Raw cut times of one reference; cached, since detection is the expensive step."""

    asset_id: str
    source: str  # the path as given, for `StyleProfile.source_refs`
    duration_s: float = Field(gt=0)
    params: ReferenceParams
    cuts: list[float]  # hard cuts + dissolve midpoints, seconds


class StyleError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# pure aggregation
# ---------------------------------------------------------------------------


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


def aggregate(refs: Sequence[ReferenceCuts], params: ReferenceParams) -> StyleProfile:
    """Pool the references: one duration distribution; cuts per minute from the totals,
    so a short reference does not weigh as much as a long one."""
    if not refs:
        raise StyleError("no references given")
    durations = [d for r in refs for d in shot_durations(r.cuts, r.duration_s, params.min_shot_s)]
    n_cuts = len(durations) - len(refs)  # after sliver merging, each reference has cuts + 1 shots
    if n_cuts < 1:
        raise StyleError("no cuts found in the references")
    total_s = sum(r.duration_s for r in refs)
    stats = duration_stats(durations)
    cuts_per_min = n_cuts / (total_s / 60.0)
    return StyleProfile(
        source_refs=[r.source for r in refs],
        shot_duration_s=stats,
        cuts_per_min=cuts_per_min,
        notes=summarize(len(refs), total_s, n_cuts, stats, cuts_per_min),
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
    if params.dissolves:
        found = transitions.find_dissolves(
            transitions.load_frames(asset.proxy_path), transitions.SAMPLE_FPS
        )
        cuts = transitions.merge_cuts(cuts, found, params.min_shot_s)
    record = ReferenceCuts(
        asset_id=asset.asset_id, source=str(path), duration_s=asset.duration_s, params=params,
        cuts=cuts,
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
    """Reference videos -> `StyleProfile` (pacing only, for now)."""
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
