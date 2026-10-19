"""Pure beat and energy analysis of mono samples; no decoding or caching."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np


class BeatParams(BaseModel):
    """Frame sizes refer to sample_rate; scale them for other input rates."""

    model_config = ConfigDict(allow_inf_nan=False)

    sample_rate: int = Field(default=22050, gt=0)
    frame_size: int = Field(default=1024, gt=0)
    hop_size: int = Field(default=512, gt=0)
    min_bpm: float = Field(default=60, gt=0)
    max_bpm: float = Field(default=200, gt=0)
    align_tolerance_s: float = Field(default=0.08, ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> BeatParams:
        if self.min_bpm >= self.max_bpm:
            raise ValueError("min_bpm must be less than max_bpm")
        if self.hop_size > self.frame_size:
            raise ValueError("hop_size must not exceed frame_size")
        return self


@dataclass(frozen=True)
class BeatResult:
    """Local analysis result: None BPM and empty beats mean no measured pulse."""

    bpm: float | None
    beats: list[float]


def _array(values: np.ndarray, rate: float) -> np.ndarray:
    import numpy as np

    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sample/frame rate must be positive and finite")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("audio/envelope must be a finite one-dimensional array")
    return array


def energy_curve(samples: np.ndarray, sample_rate: int, bins: int = 10) -> list[float] | None:
    """RMS over equal time spans, normalized by the largest bin RMS.

    Boundaries are rounded down to sample indices; every sample is included.
    Empty, too-short and silent signals are unmeasured.
    """
    import numpy as np

    if bins < 1:
        raise ValueError("bins must be positive")
    values = _array(samples, sample_rate)
    if len(values) < bins:
        return None
    peak = float(np.max(np.abs(values)))
    if peak == 0:
        return None
    # Scaling first avoids overflow/underflow without changing normalized RMS.
    values = values / peak
    edges = [i * len(values) // bins for i in range(bins + 1)]
    rms = np.array(
        [
            np.sqrt(np.mean(values[start:end] ** 2))
            for start, end in zip(edges, edges[1:], strict=False)
        ]
    )
    return (rms / rms.max()).tolist()


def onset_envelope(
    samples: np.ndarray, sample_rate: int, params: BeatParams
) -> tuple[np.ndarray, float]:
    """Positive log-energy differences, with a centered 200 ms mean removed.

    Energy frames are centered at time zero and each subsequent hop. Frame/hop
    durations follow params.sample_rate without resampling the input audio.
    """
    import numpy as np

    x = _array(samples, sample_rate)
    frame = max(1, round(params.frame_size * sample_rate / params.sample_rate))
    hop = max(1, round(params.hop_size * sample_rate / params.sample_rate))
    frame_rate = sample_rate / hop
    if not len(x):
        return np.empty(0), frame_rate
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros((len(x) - 1) // hop + 1), frame_rate
    padded = np.pad(x / peak, (frame // 2, frame - 1 - frame // 2), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded**2)))
    starts = np.arange(0, len(x), hop)
    power = np.maximum(0, (cumulative[starts + frame] - cumulative[starts]) / frame)
    energy = np.log1p(100 * power)
    flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
    window = max(1, round(0.2 * frame_rate)) | 1
    local_mean = np.convolve(
        np.pad(flux, window // 2, mode="edge"), np.ones(window) / window, mode="valid"
    )
    return np.maximum(0, flux - local_mean), frame_rate


def estimate_tempo(envelope: np.ndarray, frame_rate: float, params: BeatParams) -> float | None:
    """Autocorrelation peak with a shorter-period tie-break and parabolic refinement.

    Require two slowest periods of evidence and a normalized correlation of 0.3.
    Centering rejects stationary noise; smoothing reduces frame-grid octave bias.
    Refine over up to four periods to reduce accumulated frame-quantization drift.
    """
    import numpy as np

    values = _array(envelope, frame_rate)
    low = max(2, math.floor(60 * frame_rate / params.max_bpm))
    high = math.ceil(60 * frame_rate / params.min_bpm)
    if high < low or len(values) < 2 * (high + 1) or not np.any(values):
        return None
    centered = values / np.max(np.abs(values))
    centered = centered - centered.mean()
    energy = float(centered @ centered)
    if energy < 1e-12:
        return None
    limit = min(4 * high + 2, len(values) // 2)
    correlation = (
        np.array([centered[: len(values) - lag] @ centered[lag:] for lag in range(limit + 1)])
        / energy
    )
    smooth = np.convolve(correlation, [0.25, 0.5, 0.25], mode="same")
    candidates = [
        lag
        for lag in range(low, high + 1)
        if smooth[lag] >= smooth[lag - 1] and smooth[lag] >= smooth[lag + 1] and smooth[lag] >= 0.3
    ]
    if not candidates:
        return None
    # Frame quantization can make a multiple of the true period correlate better.
    # Prefer the shortest peak within 25% of the best, avoiding half-time clicks.
    best = max(smooth[k] for k in candidates)
    lag = next(k for k in candidates if smooth[k] >= 0.75 * best)
    numerator, denominator = 0.0, 0.0
    for multiple in range(1, 5):
        start = max(1, math.floor((multiple - 0.1) * lag))
        end = math.ceil((multiple + 0.1) * lag) + 1
        if end > limit:
            break
        peak = start + int(np.argmax(smooth[start:end]))
        if smooth[peak] < 0.3:
            continue
        left, middle, right = smooth[peak - 1 : peak + 2]
        curvature = left - 2 * middle + right
        offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
        numerator += multiple * (peak + offset)
        denominator += multiple**2
    period = numerator / denominator if denominator else float(lag)
    return float(np.clip(60 * frame_rate / period, params.min_bpm, params.max_bpm))


def track_beats(envelope: np.ndarray, frame_rate: float, bpm: float) -> list[float]:
    """Fit a fixed-period grid, then snap within 10% of a period to onset peaks.

    Unsupported grid points (below 10% of the strongest onset) are omitted.
    Returned seconds refer to the same time origin as the input samples.
    """
    import numpy as np

    values = _array(envelope, frame_rate)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be positive and finite")
    if not len(values) or np.max(values) <= 0:
        return []
    period = 60 * frame_rate / bpm
    if period < 2:
        raise ValueError("bpm must span at least two envelope frames")
    phases = np.linspace(0, period, math.ceil(period * 4), endpoint=False)
    frames = np.arange(len(values))
    phase = max(
        phases,
        key=lambda p: float(
            np.interp(np.arange(p, len(values), period), frames, values, right=0).sum()
        ),
    )
    beats: set[float] = set()
    threshold = 0.1 * float(np.max(values))
    for point in np.arange(phase, len(values), period):
        start = max(0, math.ceil(point - 0.1 * period))
        end = min(len(values), math.floor(point + 0.1 * period) + 1)
        if start < end:
            peak = start + int(np.argmax(values[start:end]))
            if values[peak] >= threshold:
                beats.add(peak / frame_rate)
    return sorted(beats)


def detect_beats(
    samples: np.ndarray, sample_rate: int, params: BeatParams | None = None
) -> BeatResult:
    """Measure mono audio; silence, short or unstructured input is unmeasured."""
    params = params or BeatParams()
    envelope, frame_rate = onset_envelope(samples, sample_rate, params)
    bpm = estimate_tempo(envelope, frame_rate, params)
    return BeatResult(bpm, track_beats(envelope, frame_rate, bpm) if bpm is not None else [])


def beat_aligned_ratio(
    cuts: Sequence[float], beats: Sequence[float], tolerance_s: float = 0.08
) -> float | None:
    """Share of cuts within inclusive tolerance; empty inputs mean not measured.

    Inputs need not be sorted. A 1e-12 s allowance handles floating-point boundary
    subtraction (e.g. 1.08 - 1.0); it does not turn an 81 ms gap into an 80 ms one.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    if not all(math.isfinite(t) for times in (cuts, beats) for t in times):
        raise ValueError("cut and beat times must be finite")
    if not len(cuts) or not len(beats):
        return None
    ordered = sorted(beats)
    aligned = 0
    for cut in cuts:
        index = bisect_left(ordered, cut)
        neighbors = ordered[max(0, index - 1) : index + 1]
        aligned += any(abs(cut - beat) <= tolerance_s + 1e-12 for beat in neighbors)
    return aligned / len(cuts)
