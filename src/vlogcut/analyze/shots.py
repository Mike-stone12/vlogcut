"""Shot detection: split each asset's proxy into contiguous shot spans + keyframes.

Cut detection uses PySceneDetect's `AdaptiveDetector`, which compares each frame's
change against its neighbours and so tolerates handheld shake and pans better than
a fixed threshold. Gradual transitions (dissolves, fades), which it misses, come from
`analyze.transitions`. The span post-processing (`cuts_to_spans`, `merge_short`,
`split_long`, `keyframe_times`) is pure. `detect_shots` adds the detector, keyframe
extraction and a per-asset cache at `<workspace>/shots/<asset_id>.json`.

Spans are not `Shot`s yet: `Shot` needs `QualityFlags`, which the quality step adds.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from vlogcut.ingest import ffmpeg
from vlogcut.paths import from_workspace, to_workspace_relative
from vlogcut.schemas import MediaAsset

log = logging.getLogger(__name__)

Span = tuple[float, float]
KEYFRAME_FRACTIONS = (0.25, 0.5, 0.75)
_EPS = 1e-6


class ShotParams(BaseModel):
    """Knobs that change the result; part of the cache key."""

    threshold: float = Field(default=3.0, gt=0)  # AdaptiveDetector adaptive_threshold
    min_shot_s: float = Field(default=0.5, gt=0)
    max_shot_s: float = Field(default=10.0, gt=0)
    dissolves: bool = True  # also detect cross-dissolves / fades (analyze.transitions)


class ShotSpan(BaseModel):
    """One shot before quality analysis: times are absolute within the asset."""

    shot_id: str
    asset_id: str
    start_s: float = Field(ge=0)
    end_s: float
    keyframes: list[str] = Field(default_factory=list)


class ShotCache(BaseModel):
    asset_id: str
    params: ShotParams
    spans: list[ShotSpan]


class ShotDetectionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# pure span logic
# ---------------------------------------------------------------------------


def cuts_to_spans(cuts: list[float], duration_s: float) -> list[Span]:
    """Cut times -> contiguous spans covering [0, duration_s] with no gaps or overlaps."""
    inner = sorted({round(c, 6) for c in cuts if _EPS < c < duration_s - _EPS})
    bounds = [0.0, *inner, duration_s]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def merge_short(spans: list[Span], min_s: float) -> list[Span]:
    """Merge spans shorter than `min_s` into their shorter neighbour (fewest frames lost)."""
    out = list(spans)
    while len(out) > 1:
        i = min(range(len(out)), key=lambda k: out[k][1] - out[k][0])
        if out[i][1] - out[i][0] >= min_s:
            break
        if i == 0:
            j = 1
        elif i == len(out) - 1:
            j = i - 1
        else:
            prev_len = out[i - 1][1] - out[i - 1][0]
            next_len = out[i + 1][1] - out[i + 1][0]
            j = i - 1 if prev_len <= next_len else i + 1
        lo, hi = min(i, j), max(i, j)
        out[lo : hi + 1] = [(out[lo][0], out[hi][1])]
    return out


def split_long(spans: list[Span], max_s: float) -> list[Span]:
    """Split spans longer than `max_s` into equal parts, each <= max_s."""
    out: list[Span] = []
    for start, end in spans:
        n = max(1, math.ceil((end - start) / max_s - _EPS))
        step = (end - start) / n
        out += [(start + k * step, end if k == n - 1 else start + (k + 1) * step) for k in range(n)]
    return out


def keyframe_times(span: Span, fractions: tuple[float, ...] = KEYFRAME_FRACTIONS) -> list[float]:
    start, end = span
    return [start + f * (end - start) for f in fractions]


def postprocess(cuts: list[float], duration_s: float, params: ShotParams) -> list[Span]:
    spans = merge_short(cuts_to_spans(cuts, duration_s), params.min_shot_s)
    return split_long(spans, params.max_shot_s)


# ---------------------------------------------------------------------------
# detector, keyframes, cache
# ---------------------------------------------------------------------------


def detect_cuts(video_path: str | Path, *, threshold: float, min_scene_len_s: float) -> list[float]:
    """Cut times in seconds (start of every scene after the first)."""
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError as exc:  # heavy optional dependency
        raise ShotDetectionError('shot detection needs: pip install -e ".[analyze]"') from exc

    from scenedetect import open_video

    fps = open_video(str(video_path)).frame_rate
    detector = AdaptiveDetector(
        adaptive_threshold=threshold, min_scene_len=max(1, round(min_scene_len_s * fps))
    )
    scenes = detect(str(video_path), detector)
    return [start.seconds for start, _ in scenes[1:]]


def extract_keyframes(video_path: str | Path, times: list[float], stem: Path) -> list[str]:
    """Write one jpg per time as `<stem>_k<i>.jpg`; return the paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        out = stem.with_name(f"{stem.name}_k{i}.jpg")
        args = ["-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video_path)]
        ffmpeg.run("ffmpeg", [*args, "-frames:v", "1", "-q:v", "3", str(out)])
        paths.append(str(out))
    return paths


def detect_shots(
    asset: MediaAsset, workspace: str | Path, params: ShotParams | None = None
) -> list[ShotSpan]:
    """Detect (or load cached) shot spans for an asset that already has a proxy."""
    params = params or ShotParams()
    if asset.proxy_path is None:
        raise ShotDetectionError(f"{asset.path}: no proxy; run ingest.make_proxy first")
    workspace = Path(workspace).resolve()
    cache_path = workspace / "shots" / f"{asset.asset_id}.json"
    cached = _load_cache(cache_path, asset, params, workspace)
    if cached is not None:
        log.debug("reusing shots %s", cache_path)
        return cached

    log.info("detecting shots in %s", asset.proxy_path)
    cuts = detect_cuts(
        asset.proxy_path, threshold=params.threshold, min_scene_len_s=params.min_shot_s
    )
    if params.dissolves:
        from vlogcut.analyze import transitions  # numpy; imported only when used

        found = transitions.find_dissolves(
            transitions.load_frames(asset.proxy_path), transitions.SAMPLE_FPS
        )
        cuts = transitions.merge_cuts(cuts, found, params.min_shot_s)
    spans = []
    for i, (start, end) in enumerate(postprocess(cuts, asset.duration_s, params)):
        shot_id = f"{asset.asset_id}_{i:03d}"
        stem = workspace / "keyframes" / asset.asset_id / shot_id
        keyframes = extract_keyframes(asset.proxy_path, keyframe_times((start, end)), stem)
        spans.append(
            ShotSpan(
                shot_id=shot_id,
                asset_id=asset.asset_id,
                start_s=start,
                end_s=end,
                keyframes=keyframes,
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".part")
    stored = [
        s.model_copy(
            update={"keyframes": [to_workspace_relative(k, workspace) for k in s.keyframes]}
        )
        for s in spans
    ]
    tmp.write_text(
        ShotCache(asset_id=asset.asset_id, params=params, spans=stored).model_dump_json(indent=1),
        encoding="utf-8",
    )
    tmp.replace(cache_path)
    return spans


def _load_cache(
    path: Path, asset: MediaAsset, params: ShotParams, workspace: Path
) -> list[ShotSpan] | None:
    if not path.is_file():
        return None
    try:
        cache = ShotCache.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        return None
    if cache.asset_id != asset.asset_id or cache.params != params:
        return None
    for span in cache.spans:
        span.keyframes = [str(from_workspace(k, workspace)) for k in span.keyframes]
    if not all(Path(k).is_file() for s in cache.spans for k in s.keyframes):
        return None
    return cache.spans
