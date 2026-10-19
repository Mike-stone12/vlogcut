"""Pooled RGB look measurements and a small-frame decoder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from vlogcut.ingest import ffmpeg
from vlogcut.schemas import LookStats as LookStats

if TYPE_CHECKING:
    import numpy as np


class LookParams(BaseModel):
    """Sampling rate and square frame size for proxy decoding."""

    model_config = ConfigDict(allow_inf_nan=False)

    sample_fps: float = Field(default=1.0, gt=0)
    size: int = Field(default=64, ge=8)


def look_stats(frames: np.ndarray) -> LookStats | None:
    """Mean Rec.709 luma, HSV saturation and R-B over all RGB pixels equally.

    Input shape is (N, H, W, 3), with positive H and W. uint8 is scaled to
    [0, 1]; floating-point input is clipped to that range without mutation.
    Zero frames means not measured.
    """
    import numpy as np

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3 or min(array.shape[1:3]) < 1:
        raise ValueError("frames must have shape (N, H, W, 3) with positive H and W")
    if len(array) == 0:
        return None
    if array.dtype == np.uint8:
        rgb = array.astype(np.float64) / 255.0
    elif np.issubdtype(array.dtype, np.floating):
        rgb = np.clip(array.astype(np.float64), 0, 1)
    else:
        raise ValueError("frames must contain uint8 or floating-point RGB values")

    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = np.divide(
        maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0
    )
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return LookStats(
        brightness=float(luma.mean()),
        saturation=float(saturation.mean()),
        warmth=float((rgb[..., 0] - rgb[..., 2]).mean()),
        frames=len(array),
    )


def pool_look(stats: Sequence[LookStats | None]) -> LookStats | None:
    """Frame-weighted means across measured results, skipping None entries."""
    measured = [stat for stat in stats if stat is not None]
    frames = sum(stat.frames for stat in measured)
    if not frames:
        return None
    return LookStats(
        brightness=sum(stat.brightness * stat.frames for stat in measured) / frames,
        saturation=sum(stat.saturation * stat.frames for stat in measured) / frames,
        warmth=sum(stat.warmth * stat.frames for stat in measured) / frames,
        frames=frames,
    )


def load_rgb_frames(
    path: str | Path, params: LookParams | None = None, *, max_frames: int | None = None
) -> np.ndarray:
    """Decode a proxy to uint8 RGB frames, shape (N, size, size, 3).

    Defaults to LookParams() (1 fps, 64x64). Optional cap bounds decoded output.
    Shared ffmpeg errors propagate.
    """
    import numpy as np

    params = params or LookParams()
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    size = params.size
    raw = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", "-i", str(path), "-an"]
        + [
            "-vf",
            f"fps={params.sample_fps},scale={size}:{size},format=rgb24",
        ]
        + (["-frames:v", str(max_frames)] if max_frames is not None else [])
        + [
            "-f",
            "rawvideo",
            "-",
        ],
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3)
