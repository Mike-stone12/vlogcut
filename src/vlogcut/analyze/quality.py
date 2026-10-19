"""Quality flags per shot: sharpness, exposure clipping, camera shake, usable range.

This step measures; it does not judge taste. Blur and shake are often deliberate in a
vlog (talking while walking, concerts, a handheld look), so:

- only clearly unusable shots are `rejected` (covered lens / black frames, blown out,
  or bad from start to end with nobody talking);
- bad stretches at the start or end (raising / lowering the phone, focus hunting) are
  trimmed via `usable_range` instead of dropping the shot;
- milder issues become `warnings` for the style profile and planner to weigh.

Per-frame metrics, motion estimation and `assess` are pure numpy. `assess_shots`
decodes each shot from the proxy (every frame at 270p luma + 8 kHz mono audio) and caches
the result at `<workspace>/shots/<asset_id>.quality.json`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from vlogcut.analyze.pauses import _load_mono_pcm
from vlogcut.analyze.shots import ShotSpan
from vlogcut.ingest import ffmpeg
from vlogcut.ingest.proxy import proxy_dims, proxy_rate
from vlogcut.paths import from_workspace, to_workspace_relative
from vlogcut.schemas import MediaAsset, QualityFlags, Shot

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

SHORT_SIDE = 270
AUDIO_RATE = 8000


class QualityParams(BaseModel):
    """Thresholds; part of the cache key. Units: 0..255 luma, px at 270p, seconds."""

    # Defaults calibrated on generated footage (see the decisions log): a slight blur
    # (gblur sigma 3 at 1080p) stays above blur_warn, sigma 8 falls below blur_bad.
    blur_bad: float = Field(default=10.0, gt=0)  # per-frame Laplacian variance below = bad
    blur_warn: float = Field(default=30.0, gt=0)  # shot median below = "blurry" warning
    # relative to the shot's own sharp part (p90): catches raise/lower motion blur and
    # focus hunting whatever the content; a shot that is all sky keeps a low baseline
    blur_rel: float = Field(default=0.2, gt=0, lt=1)
    dark_bad: float = Field(default=0.95, gt=0, le=1)  # near-black frame: lens covered
    bright_bad: float = Field(default=0.9, gt=0, le=1)  # blown-out frame
    clip_warn: float = Field(default=0.3, gt=0, le=1)  # night scenes may be dark, not bad
    jitter_bad: float = Field(default=25.0, gt=0)  # RMS jitter over vote_s above = bad
    shake_warn: float = Field(default=3.0, gt=0)  # shot RMS jitter above = "shaky" warning
    smooth_s: float = Field(default=1.0, gt=0)  # window that defines "deliberate" motion
    vote_s: float = Field(default=0.5, gt=0)  # bad samples need a local majority to count
    min_usable_s: float = Field(default=0.5, gt=0)
    audio_active_db: float = -35.0  # 100 ms RMS above this counts as sound
    audio_keep: float = Field(default=0.3, ge=0, le=1)  # active fraction that saves a shot


class ShotMetrics(BaseModel):
    """Raw per-sample measurements for one shot (sampled at `fps`)."""

    fps: float
    sharpness: list[float]
    dark: list[float]  # fraction of pixels <= 5
    bright: list[float]  # fraction of pixels >= 250
    motion: list[tuple[float, float]]  # (dy, dx) per step between samples, px
    audio_active: float = 0.0


# ---------------------------------------------------------------------------
# per-frame metrics (pure)
# ---------------------------------------------------------------------------


def laplacian_var(frames: np.ndarray) -> np.ndarray:
    """Variance of the 4-neighbour Laplacian per frame; frames are (n, h, w) on 0..255."""
    import numpy as np

    f = frames.astype(np.float32)
    lap = f[:, :-2, 1:-1] + f[:, 2:, 1:-1] + f[:, 1:-1, :-2] + f[:, 1:-1, 2:] - 4 * f[:, 1:-1, 1:-1]
    return lap.reshape(len(f), -1).var(axis=1)


def clipped(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dark, bright) fractions per frame: luma <= 5 and >= 250."""
    flat = frames.reshape(len(frames), -1)
    return (flat <= 5).mean(axis=1), (flat >= 250).mean(axis=1)


def global_motion(frames: np.ndarray, grid: int = 3, min_peak: float = 0.03) -> np.ndarray:
    """Per-step (dy, dx) camera translation by phase correlation, shape (n - 1, 2).

    The frame is split into `grid` x `grid` tiles and the per-tile shifts are reduced
    with a median: the camera moves every tile, a subject crossing the frame only a few.
    Tiles with a weak correlation peak (textureless, cut) are ignored; a step with no
    usable tile reports 0.
    """
    import numpy as np

    n, h, w = frames.shape
    if n < 2:
        return np.zeros((0, 2))
    th, tw = h // grid, w // grid
    shifts = np.stack(
        [
            _phase_shifts(frames[:, r * th : (r + 1) * th, c * tw : (c + 1) * tw], min_peak)
            for r in range(grid)
            for c in range(grid)
        ]
    )  # (tiles, n - 1, 2), NaN where unreliable
    valid = ~np.isnan(shifts[..., 0])
    out = np.zeros((n - 1, 2))
    for i in range(n - 1):
        if valid[:, i].any():
            out[i] = np.median(shifts[valid[:, i], i], axis=0)
    return out


def _phase_shifts(frames: np.ndarray, min_peak: float) -> np.ndarray:
    """(dy, dx) between consecutive frames of one tile stack; NaN for weak peaks."""
    import numpy as np

    n, h, w = frames.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spec = np.fft.rfft2((frames - frames.mean(axis=(1, 2), keepdims=True)) * win)
    cross = spec[1:] * np.conj(spec[:-1])
    corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=(h, w))
    out = np.full((n - 1, 2), np.nan)
    for i, c in enumerate(corr):
        y, x = np.unravel_index(int(np.argmax(c)), c.shape)
        if c[y, x] < min_peak:
            continue
        dy = y + _subpixel(c[(y - 1) % h, x], c[y, x], c[(y + 1) % h, x])
        dx = x + _subpixel(c[y, (x - 1) % w], c[y, x], c[y, (x + 1) % w])
        out[i] = ((dy + h / 2) % h - h / 2, (dx + w / 2) % w - w / 2)
    return out


def jitter(motion: np.ndarray, fps: float, smooth_s: float) -> np.ndarray:
    """|camera path - smoothed path| per sample (n = len(motion) + 1).

    The smoothed path is a local straight-line fit over `smooth_s` (an order-1
    Savitzky-Golay filter): a moving average inside the shot, the line through the
    first / last window at the ends. A constant-speed pan scores exactly zero
    everywhere, and shake at the very start or end is not flattened away.
    """
    import numpy as np

    path = np.vstack([np.zeros((1, 2)), np.cumsum(motion, axis=0)])
    n = len(path)
    k = min(max(3, round(smooth_s * fps)) | 1, n if n % 2 else n - 1)
    if k < 3:
        return np.zeros(n)
    half = k // 2
    smooth = np.empty_like(path)
    kernel = np.ones(k) / k
    for j in range(2):
        smooth[half : n - half, j] = np.convolve(path[:, j], kernel, mode="valid")
    t = np.arange(k)
    for idx, seg in ((np.arange(half), slice(0, k)), (np.arange(n - half, n), slice(n - k, n))):
        coef = np.polyfit(t, path[seg], 1)  # (2, 2): slope and intercept per axis
        smooth[idx] = np.outer(idx - (seg.start or 0), coef[0]) + coef[1]
    return np.linalg.norm(path - smooth, axis=1)


def audio_active(samples: np.ndarray, rate: int, threshold_db: float) -> float:
    """Fraction of 100 ms windows whose RMS (dBFS) is above `threshold_db`."""
    import numpy as np

    win = rate // 10
    n = len(samples) // win
    if n == 0:
        return 0.0
    x = samples[: n * win].reshape(n, win).astype(np.float64) / 32768.0
    db = 20 * np.log10(np.sqrt((x**2).mean(axis=1)) + 1e-9)
    return float((db > threshold_db).mean())


# ---------------------------------------------------------------------------
# decision (pure)
# ---------------------------------------------------------------------------


def assess(m: ShotMetrics, params: QualityParams) -> QualityFlags:
    import numpy as np

    sharp, dark, bright = np.array(m.sharpness), np.array(m.dark), np.array(m.bright)
    motion = np.array(m.motion, dtype=float).reshape(-1, 2)
    jit = jitter(motion, m.fps, params.smooth_s)
    k = max(1, round(params.vote_s * m.fps)) | 1
    floor = max(params.blur_bad, params.blur_rel * float(np.percentile(sharp, 90)))
    blurry = _majority(sharp < floor, k)
    too_dark = _majority(dark > params.dark_bad, k)
    too_bright = _majority(bright > params.bright_bad, k)
    # shake oscillates through zero, so judge its local RMS, not single samples
    shaky = _local_rms(jit, k) > params.jitter_bad
    bad = blurry | too_dark | too_bright | shaky

    n, duration = len(bad), len(bad) / m.fps
    # usable = first to last good run that lasts at least the vote window, so a stray
    # good frame inside a bad stretch does not anchor the range
    runs = [r for r in _runs(~bad) if r[1] - r[0] >= min(k, n)]
    usable = None
    if runs:
        lo, hi = runs[0][0] / m.fps, min(runs[-1][1] / m.fps, duration)
        if hi - lo >= params.min_usable_s:
            usable = (lo, hi)
    sl = slice(round(usable[0] * m.fps), round(usable[1] * m.fps)) if usable else slice(0, n)

    # Rejected only if bad from start to end, and then only for things no style wants:
    # a covered lens / black frames, blown out, or violent shake with nobody talking.
    # Blur alone never rejects: skies and walls look "blurry" too, and soft focus can
    # be deliberate; the captioner and planner can judge it.
    reasons: list[str] = []
    saved_by_audio = False
    if usable is None:
        if too_dark.mean() > 0.5:
            reasons.append("too_dark")
        elif too_bright.mean() > 0.5:
            reasons.append("overexposed")
        elif shaky.mean() > 0.5:
            if m.audio_active >= params.audio_keep:
                saved_by_audio = True  # walk-and-talk, concerts
            else:
                reasons.append("shaky")

    sharpness = float(np.median(sharp[sl]))
    clip = float((dark[sl] + bright[sl]).mean())
    # re-measure on the usable part alone, so trimmed shake does not leak into it
    kept = jitter(motion[sl.start : max(sl.start, sl.stop - 1)], m.fps, params.smooth_s)
    shake = float(np.sqrt((kept**2).mean()))
    warnings = []
    if sharpness < params.blur_warn:
        warnings.append("blurry")
    if shake > params.shake_warn:
        warnings.append("shaky")
    if clip > params.clip_warn:
        warnings.append("dark" if dark[sl].mean() >= bright[sl].mean() else "bright")
    if saved_by_audio:
        warnings.append("kept_for_audio")
    if usable is not None and usable[1] - usable[0] > duration - 0.5 / m.fps:
        usable = None  # nothing trimmed

    return QualityFlags(
        sharpness=sharpness,
        exposure_clip=min(1.0, clip),
        shake=shake,
        rejected=bool(reasons),
        reject_reasons=reasons,
        warnings=[] if reasons else warnings,  # a rejected shot only needs its reasons
        usable_range=usable,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where `mask` is True."""
    import numpy as np

    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _local_rms(x: np.ndarray, k: int) -> np.ndarray:
    import numpy as np

    if k <= 1 or len(x) < k:
        return np.sqrt(np.full(len(x), (x**2).mean())) if len(x) else x
    sq = np.convolve(np.pad(x**2, k // 2, mode="edge"), np.ones(k) / k, "valid")
    return np.sqrt(sq)


def _majority(mask: np.ndarray, k: int) -> np.ndarray:
    """Centered k-sample majority vote, so isolated spikes do not trim or reject."""
    import numpy as np

    if k <= 1 or len(mask) < k:
        return mask
    votes = np.convolve(np.pad(mask.astype(float), k // 2, mode="edge"), np.ones(k), "valid")
    return votes > k / 2


def _subpixel(left: float, mid: float, right: float) -> float:
    denom = left - 2 * mid + right
    return 0.0 if abs(denom) < 1e-12 else float(0.5 * (left - right) / denom)


# ---------------------------------------------------------------------------
# decoding, cache
# ---------------------------------------------------------------------------


class QualityCache(BaseModel):
    asset_id: str
    params: QualityParams
    spans: list[tuple[str, float, float]]
    shots: list[Shot]


def measure_shot(asset: MediaAsset, span: ShotSpan, params: QualityParams) -> ShotMetrics:
    import numpy as np

    assert asset.proxy_path is not None
    w, h = proxy_dims(asset.width, asset.height, SHORT_SIDE)
    # every proxy frame: hand shake is fast, and at lower rates the shift between two
    # samples outgrows what phase correlation can measure
    fps = proxy_rate(asset.fps)
    seek = ["-ss", f"{span.start_s:.3f}", "-t", f"{span.end_s - span.start_s:.3f}"]
    raw = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", *seek, "-i", asset.proxy_path, "-an"]
        + ["-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
    )
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
    if len(frames) == 0:
        frames = np.zeros((1, h, w), np.uint8)
    dark, bright = clipped(frames)
    motion = global_motion(frames.astype(np.float32))
    active = 0.0
    if asset.has_audio:
        pcm = _load_mono_pcm(asset.proxy_path, AUDIO_RATE, seek)
        active = audio_active(pcm, AUDIO_RATE, params.audio_active_db)
    return ShotMetrics(
        fps=float(fps),
        sharpness=laplacian_var(frames).tolist(),
        dark=dark.tolist(),
        bright=bright.tolist(),
        motion=[(float(dy), float(dx)) for dy, dx in motion],
        audio_active=active,
    )


def assess_shots(
    asset: MediaAsset,
    spans: list[ShotSpan],
    workspace: str | Path,
    params: QualityParams | None = None,
) -> list[Shot]:
    """Measure every span and return full `Shot` records (cached per asset)."""
    params = params or QualityParams()
    key = [(s.shot_id, s.start_s, s.end_s) for s in spans]
    cache_path = Path(workspace) / "shots" / f"{asset.asset_id}.quality.json"
    if cache_path.is_file():
        try:
            cache = QualityCache.model_validate_json(cache_path.read_text(encoding="utf-8"))
            if cache.params == params and cache.spans == key:
                for shot in cache.shots:
                    shot.keyframes = [str(from_workspace(k, workspace)) for k in shot.keyframes]
                return cache.shots
        except (ValidationError, ValueError):
            pass

    shots = []
    for span in spans:
        flags = assess(measure_shot(asset, span, params), params)
        if flags.usable_range is not None:  # sampling can overshoot the shot end slightly
            lo, hi = flags.usable_range
            flags.usable_range = (lo, min(hi, span.end_s - span.start_s))
        shots.append(
            Shot(
                shot_id=span.shot_id,
                asset_id=span.asset_id,
                start_s=span.start_s,
                end_s=span.end_s,
                keyframes=span.keyframes,
                quality=flags,
            )
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".part")
    stored = [
        s.model_copy(
            update={"keyframes": [to_workspace_relative(k, workspace) for k in s.keyframes]}
        )
        for s in shots
    ]
    cache = QualityCache(asset_id=asset.asset_id, params=params, spans=key, shots=stored)
    tmp.write_text(cache.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(cache_path)
    return shots
