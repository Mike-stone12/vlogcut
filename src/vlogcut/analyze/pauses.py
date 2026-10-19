"""Energy-based audio gaps, cached per asset and mapped to shot cut candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vlogcut.ingest import ffmpeg
from vlogcut.schemas import CutPoint, MediaAsset

if TYPE_CHECKING:
    import numpy as np


class PauseParams(BaseModel):
    """Serializable analysis settings, part of the per-asset cache key."""

    model_config = ConfigDict(allow_inf_nan=False)

    sample_rate: int = Field(default=16000, gt=0)
    frame_s: float = Field(default=0.02, gt=0)
    absolute_floor_db: float = Field(default=-50.0, ge=-100, le=0)
    relative_drop_db: float = Field(default=25.0, gt=0)
    loud_percentile: float = Field(default=95.0, gt=0, le=100)
    min_pause_s: float = Field(default=0.25, gt=0)
    smooth_s: float = Field(default=0.06, ge=0)


@dataclass(frozen=True)
class Pause:
    """Quiet interval [start_s, end_s) in seconds from the asset's start."""

    start_s: float
    end_s: float

    @property
    def mid_s(self) -> float:
        return (self.start_s + self.end_s) / 2


class PauseCache(BaseModel):
    """Per-asset detection result; not a cross-module data contract."""

    asset_id: str
    params: PauseParams
    pauses: list[Pause]

    @model_validator(mode="after")
    def _check_intervals(self) -> PauseCache:
        for pause in self.pauses:
            if not (
                math.isfinite(pause.start_s)
                and math.isfinite(pause.end_s)
                and 0 <= pause.start_s < pause.end_s
            ):
                raise ValueError("pause intervals must be finite, nonnegative and ordered")
        return self


def rms_envelope(samples: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> np.ndarray:
    """RMS dBFS of mono floats (full scale 1), floored at -100 dB.

    Frames use round(sample_rate * frame_s) samples. Include a final partial
    frame, averaging only its real samples; empty input yields an empty array.
    """
    import numpy as np

    if sample_rate <= 0 or not math.isfinite(frame_s) or frame_s <= 0:
        raise ValueError("sample_rate and frame_s must be positive and finite")
    win = round(sample_rate * frame_s)
    if win < 1:
        raise ValueError("frame_s must span at least one sample")
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("samples must be a finite mono array")
    if not len(x):
        return np.empty(0)
    starts = np.arange(0, len(x), win)
    power = np.add.reduceat(x**2, starts) / np.minimum(win, len(x) - starts)
    return 10 * np.log10(np.maximum(power, 1e-10))


def find_pauses(envelope_db: np.ndarray, frame_s: float, params: PauseParams) -> list[Pause]:
    """Find quiet runs bounded by sound, with frame-quantized asset timestamps.

    Quiet means strictly below max(absolute_floor_db, loud percentile minus
    relative_drop_db). A centered odd-window majority removes isolated frames.
    Leading/trailing quiet and entirely quiet recordings provide no internal gap.
    """
    import numpy as np

    if not math.isfinite(frame_s) or frame_s <= 0:
        raise ValueError("frame_s must be positive and finite")
    levels = np.asarray(envelope_db, dtype=np.float64)
    if levels.ndim != 1 or not np.isfinite(levels).all():
        raise ValueError("envelope_db must be a finite one-dimensional array")
    if not len(levels):
        return []
    threshold = max(
        params.absolute_floor_db,
        float(np.percentile(levels, params.loud_percentile)) - params.relative_drop_db,
    )
    quiet = levels < threshold
    k = max(1, round(params.smooth_s / frame_s)) | 1
    if 1 < k <= len(quiet):
        votes = np.convolve(np.pad(quiet.astype(float), k // 2, mode="edge"), np.ones(k), "valid")
        quiet = votes > k / 2
    edges = np.diff(np.concatenate([[0], quiet.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        Pause(float(start * frame_s), float(end * frame_s))
        for start, end in zip(starts, ends, strict=True)
        if start > 0 and end < len(quiet) and (end - start) * frame_s >= params.min_pause_s
    ]


def cut_points_in(
    pauses: Sequence[Pause], start_s: float, end_s: float, margin_s: float = 0.1
) -> list[float]:
    """Sorted asset-time midpoints strictly inside the bounds plus a 100 ms margin."""
    return _midpoints_in([p.mid_s for p in pauses], start_s, end_s, margin_s)


def cut_points_for_span(
    midpoints: Sequence[float], start_s: float, end_s: float, margin_s: float = 0.1
) -> list[CutPoint]:
    """Unique pause candidates in asset time, strictly inside the span plus margin."""
    return [
        CutPoint(time_s=t, kind="pause")
        for t in sorted(set(_midpoints_in(midpoints, start_s, end_s, margin_s)))
    ]


def _midpoints_in(
    midpoints: Sequence[float], start_s: float, end_s: float, margin_s: float
) -> list[float]:
    """Share strict margin filtering between pause records and midpoint candidates."""
    if not all(math.isfinite(v) for v in (start_s, end_s, margin_s)):
        raise ValueError("bounds and margin must be finite")
    if end_s < start_s or margin_s < 0:
        raise ValueError("bounds must be ordered and margin nonnegative")
    return sorted(t for t in midpoints if start_s + margin_s < t < end_s - margin_s)


def _load_mono_pcm(path: str | Path, sample_rate: int, seek: Sequence[str] = ()) -> np.ndarray:
    """Shared signed 16-bit decoder; callers establish that audio exists."""
    import numpy as np

    pcm = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", *seek, "-i", str(path), "-vn", "-ac", "1"]
        + ["-ar", str(sample_rate), "-f", "s16le", "-"],
    )
    return np.frombuffer(pcm, dtype="<i2")


def load_mono_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray | None:
    """Decode normalized mono floats, or None if no audio stream exists.

    Probe and decode failures propagate; they must not be mistaken for silence.
    Uses quality's PCM decoder so its existing 8 kHz activity measure is unchanged.
    """
    import numpy as np

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    streams = ffmpeg.run(
        "ffprobe",
        ["-v", "error", "-select_streams", "a", "-show_entries", "stream=index"]
        + ["-of", "csv=p=0", str(path)],
    )
    if not streams.strip():
        return None
    return _load_mono_pcm(path, sample_rate).astype(np.float32) / 32768.0


def detect_pauses(path: str | Path, params: PauseParams | None = None) -> list[Pause]:
    """Decode one proxy and find its internal quiet gaps; no audio yields []."""
    params = params or PauseParams()
    samples = load_mono_audio(path, params.sample_rate)
    if samples is None:
        return []
    envelope = rms_envelope(samples, params.sample_rate, params.frame_s)
    # Match the actual frame grid when frame_s * sample_rate is not an integer.
    frame_s = round(params.frame_s * params.sample_rate) / params.sample_rate
    return find_pauses(envelope, frame_s, params)


def analyze_pauses(
    asset: MediaAsset, workspace: str | Path, params: PauseParams | None = None
) -> list[Pause]:
    """Detect or load proxy pauses, keyed by asset fingerprint and detection params."""
    params = params or PauseParams()
    if asset.proxy_path is None:
        raise ValueError(f"{asset.path}: no proxy; run ingest.make_proxy first")
    cache_path = Path(workspace) / "pauses" / f"{asset.asset_id}.json"
    if cache_path.is_file():
        try:
            cache = PauseCache.model_validate_json(cache_path.read_text(encoding="utf-8"))
            if cache.asset_id == asset.asset_id and cache.params == params:
                return cache.pauses
        except (ValidationError, ValueError):
            pass

    pauses = detect_pauses(asset.proxy_path, params) if asset.has_audio else []
    cache = PauseCache(asset_id=asset.asset_id, params=params, pauses=pauses)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".part")
    tmp.write_text(cache.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(cache_path)
    return pauses
