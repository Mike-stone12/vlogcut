"""Low-res proxies for analysis and preview rendering.

Proxies are 540 px on the short side (never upscaled), constant frame rate at
min(source fps, 30), and keep the source timeline so shot times map 1:1.
`proxy_dims`, `proxy_rate` and `proxy_args` are pure; `make_proxy` runs ffmpeg
and caches the result under `<workspace>/proxies/<asset_id>.mp4`.
"""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path

from vlogcut.ingest import ffmpeg
from vlogcut.schemas import MediaAsset

log = logging.getLogger(__name__)

PROXY_SHORT_SIDE = 540
PROXY_MAX_FPS = 30
_CRF = 28
_AUDIO_BITRATE = "96k"
# A cached proxy is reused only if its duration is this close to the source
_DURATION_TOLERANCE_S = 0.25
_PROBE_TIMEOUT_S = 60


class ProxyError(RuntimeError):
    """ffmpeg failed to produce a usable proxy."""


def proxy_dims(width: int, height: int, short_side: int = PROXY_SHORT_SIDE) -> tuple[int, int]:
    """Display dims -> proxy dims: scale the short side down to `short_side`, both even."""
    scale = min(1.0, short_side / min(width, height))
    return _even(width * scale), _even(height * scale)


def proxy_rate(fps: float, max_fps: int = PROXY_MAX_FPS) -> Fraction:
    """Output frame rate as an exact fraction: 59.94 -> 30, 29.97 -> 30000/1001."""
    rate = Fraction(fps).limit_denominator(1001)
    return min(rate, Fraction(max_fps))


def proxy_args(asset: MediaAsset, out_path: str | Path) -> list[str]:
    """ffmpeg arguments that turn `asset.path` into a proxy at `out_path`.

    ffmpeg auto-rotates on decode, so the scale target uses the display dims
    stored on `MediaAsset` and the proxy carries no rotation metadata.
    """
    w, h = proxy_dims(asset.width, asset.height)
    rate = proxy_rate(asset.fps)
    gop = max(1, round(float(rate)))  # keyframe every ~1 s keeps seeking cheap
    args = ["-v", "error", "-y", "-i", str(asset.path)]
    args += ["-map", "0:V:0"]  # capital V: skip attached cover pictures
    args += ["-vf", f"fps={rate},scale={w}:{h},setsar=1"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(_CRF)]
    args += ["-pix_fmt", "yuv420p", "-g", str(gop)]
    if asset.has_audio:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", _AUDIO_BITRATE, "-ac", "2"]
    else:
        args += ["-an"]
    args += ["-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", str(out_path)]
    return args


def proxy_path_for(asset: MediaAsset, workspace: str | Path) -> Path:
    return Path(workspace) / "proxies" / f"{asset.asset_id}.mp4"


def make_proxy(
    asset: MediaAsset, workspace: str | Path, *, timeout: float | None = None
) -> MediaAsset:
    """Build (or reuse) the proxy for `asset`; return a copy with `proxy_path` set."""
    out = proxy_path_for(asset, workspace)
    if _is_valid_proxy(out, asset):
        log.debug("reusing proxy %s", out)
        return asset.model_copy(update={"proxy_path": str(out)})

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.part.mp4")  # never leave a half-written proxy
    log.info("building proxy for %s -> %s", asset.path, out)
    try:
        ffmpeg.run("ffmpeg", proxy_args(asset, tmp), timeout=timeout)
    except ffmpeg.FFmpegError as exc:
        tmp.unlink(missing_ok=True)
        raise ProxyError(f"{asset.path}: {exc}") from exc
    tmp.replace(out)
    return asset.model_copy(update={"proxy_path": str(out)})


def _is_valid_proxy(path: Path, asset: MediaAsset) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        out = ffmpeg.run(
            "ffprobe",
            ["-v", "error", "-print_format", "json", "-show_format", str(path)],
            timeout=_PROBE_TIMEOUT_S,
        )
        duration = float(json.loads(out)["format"]["duration"])
    except (ffmpeg.FFmpegError, KeyError, ValueError):
        return False
    return abs(duration - asset.duration_s) <= _DURATION_TOLERANCE_S


def _even(x: float) -> int:
    return max(2, int(x) // 2 * 2)  # round down: never larger than the source
