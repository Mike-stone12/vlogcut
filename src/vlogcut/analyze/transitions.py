"""Gradual transitions (cross-dissolves, fades) that frame-difference cut detectors miss.

A dissolve from A to B is a linear blend: the frame at fraction f of the way through is
close to (1 - f)·A + f·B. For every candidate window we take the frames at both ends
and test three things:

1. the ends differ enough (`min_change`) — rules out static shots and sensor noise;
2. the ends are *structurally* different: low correlation of the z-scored frames
   (`max_corr`) — rules out exposure / white-balance swings, which keep the structure;
3. the frames at 1/4, 1/2, 3/4 fit the blend (`max_residual`, relative to the change) —
   rules out pans, zooms and subject motion, and hard cuts (a cut's middle frame is A
   or B, residual ≈ 0.5).

Overlapping hits are reduced to one (start, end) span per transition; callers cut at
its midpoint. Fades to/from black are blends with a flat frame and are found the same way.

`find_dissolves` is pure numpy over a (frames, h, w) array; `load_frames` decodes a
proxy into that array with ffmpeg.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from vlogcut.ingest import ffmpeg

if TYPE_CHECKING:
    import numpy as np

SAMPLE_FPS = 15
FRAME_SIZE = 64  # frames are squashed to 64x64 luma; aspect ratio does not matter here
WINDOWS_S = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
MIN_CHANGE = 0.05  # mean |A - B| on 0..1 luma
MAX_CORR = 0.5
MAX_RESIDUAL = 0.2
_FIT_POINTS = (0.25, 0.5, 0.75)
_FIT_TOLERANCE = 0.02  # "almost as good" when picking the longest window in a group


def load_frames(
    video_path: str | Path, *, fps: int = SAMPLE_FPS, size: int = FRAME_SIZE
) -> np.ndarray:
    """Decode to float32 luma frames in [0, 1], shape (n, size, size)."""
    import numpy as np

    raw = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", "-i", str(video_path), "-an"]
        + ["-vf", f"fps={fps},scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
    )
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size).astype(np.float32) / 255.0


def find_dissolves(
    frames: np.ndarray,
    fps: float,
    *,
    windows_s: Sequence[float] = WINDOWS_S,
    min_change: float = MIN_CHANGE,
    max_corr: float = MAX_CORR,
    max_residual: float = MAX_RESIDUAL,
) -> list[tuple[float, float]]:
    """(start_s, end_s) of gradual transitions in `frames` sampled at `fps`."""
    import numpy as np

    n = len(frames)
    flat = frames.reshape(n, -1).astype(np.float32)
    std = flat.std(axis=1, keepdims=True)
    # flat (e.g. black) frames get z = 0, i.e. no correlation with anything
    z = np.where(std > 1e-3, (flat - flat.mean(axis=1, keepdims=True)) / np.maximum(std, 1e-3), 0)

    hits: list[tuple[float, int, int]] = []  # (relative residual, start, end)
    for length in sorted({max(4, round(w * fps)) for w in windows_s}):
        if length >= n:
            continue
        a, b = flat[: n - length], flat[length:]
        change = np.abs(a - b).mean(axis=1)
        corr = (z[: n - length] * z[length:]).mean(axis=1)
        residual = np.zeros_like(change)
        for point in _FIT_POINTS:
            k = round(point * length)
            f = k / length
            residual += np.abs(flat[k : n - length + k] - ((1 - f) * a + f * b)).mean(axis=1)
        ratio = residual / len(_FIT_POINTS) / np.maximum(change, 1e-6)
        ok = (change > min_change) & (corr < max_corr) & (ratio < max_residual)
        hits += [(float(ratio[i]), int(i), int(i) + length) for i in np.flatnonzero(ok)]

    # Group overlapping hits around the best-fitting ones. Any sub-window of a blend is
    # also a blend, so the best fit is often a slice of the transition; within a group
    # take the longest window that fits almost as well, which spans the whole transition.
    groups: list[list[tuple[float, int, int]]] = []
    for hit in sorted(hits):
        _, start, end = hit
        for group in groups:
            _, gs, ge = group[0]
            if start < ge and end > gs:
                group.append(hit)
                break
        else:
            groups.append([hit])
    spans = []
    for group in groups:
        best = group[0][0]
        _, start, end = max(
            (h for h in group if h[0] <= best + _FIT_TOLERANCE),
            key=lambda h: (h[2] - h[1], -h[0]),
        )
        spans.append((start / fps, end / fps))
    return sorted(spans)


def merge_cuts(
    hard: Sequence[float], transitions: Sequence[tuple[float, float]], min_gap_s: float
) -> list[float]:
    """Cut times from hard cuts plus transition midpoints. A hard cut inside (or within
    `min_gap_s` of) a transition is the detector firing on the blend, so it is dropped."""
    mids = [(s + e) / 2 for s, e in transitions]
    extra = [
        h for h in hard if all(not (s - min_gap_s <= h <= e + min_gap_s) for s, e in transitions)
    ]
    return sorted([*mids, *extra])
