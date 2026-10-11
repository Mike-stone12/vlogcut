"""ffprobe metadata -> `MediaAsset`.

`parse_probe` is pure (ffprobe JSON in, MediaAsset out) so almost all of the
logic is testable without ffmpeg. `probe_asset` adds the subprocess and the
content fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from vlogcut.ingest import ffmpeg
from vlogcut.schemas import MediaAsset

_FINGERPRINT_CHUNK = 1 << 20  # 1 MiB from each end of the file
_PROBE_TIMEOUT_S = 60
# ffprobe "format_name" values that mean a still image, not footage
_IMAGE_FORMATS = ("image2", "_pipe", "webp")
# Camera clocks that were never set report dates like 1904-01-01 or 1970-01-01
_MIN_PLAUSIBLE_YEAR = 2000
_ISO6709 = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


class ProbeError(RuntimeError):
    """The file could not be probed or is not usable footage."""


def probe_asset(path: str | Path) -> MediaAsset:
    path = Path(path)
    try:
        out = ffmpeg.run(
            "ffprobe",
            ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            timeout=_PROBE_TIMEOUT_S,
        )
    except ffmpeg.FFmpegError as exc:
        raise ProbeError(f"{path}: {exc}") from exc
    return parse_probe(json.loads(out), path=path, asset_id=asset_id_for(path))


def asset_id_for(path: str | Path) -> str:
    """Content fingerprint: stable across renames/moves without reading whole files."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(_FINGERPRINT_CHUNK))
        if size > _FINGERPRINT_CHUNK:
            f.seek(max(size - _FINGERPRINT_CHUNK, _FINGERPRINT_CHUNK))
            h.update(f.read(_FINGERPRINT_CHUNK))
    return h.hexdigest()[:12]


def parse_probe(probe: dict[str, Any], *, path: str | Path, asset_id: str) -> MediaAsset:
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    if any(tag in fmt.get("format_name", "") for tag in _IMAGE_FORMATS):
        raise ProbeError(f"{path}: still image ({fmt.get('format_name')}), not footage")
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise ProbeError(f"{path}: no video stream")

    duration = _to_float(video.get("duration")) or _to_float(fmt.get("duration"))
    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    if not duration or not fps:
        raise ProbeError(f"{path}: missing duration or frame rate")

    width, height = int(video["width"]), int(video["height"])
    if rotation_of(video) % 180 == 90:
        width, height = height, width  # store display (upright) dimensions

    tags = _merged_tags(fmt, video)
    return MediaAsset(
        asset_id=asset_id,
        path=str(path),
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        created_at=_created_at(tags),
        gps=_gps(tags),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


# ---------------------------------------------------------------------------
# field helpers (pure)
# ---------------------------------------------------------------------------


def parse_rate(value: str | None) -> float | None:
    """'30000/1001' -> 29.97; '0/0', '' or None -> None."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def rotation_of(video: dict[str, Any]) -> int:
    """Display rotation in degrees as ffprobe reports it, normalised to [0, 360)."""
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            return round(float(side["rotation"])) % 360
    rotate = video.get("tags", {}).get("rotate")  # older ffmpeg / some Android files
    return round(float(rotate)) % 360 if rotate else 0


def parse_creation_time(value: str | None) -> datetime | None:
    """Parse a container date tag into an aware datetime; implausible dates -> None."""
    if not value:
        return None
    value = value.split(";")[0].strip()  # mp4 metadata tags may repeat: "a;b"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)  # QuickTime creation_time is defined as UTC
    return dt if dt.year >= _MIN_PLAUSIBLE_YEAR else None


def parse_iso6709(value: str | None) -> tuple[float, float] | None:
    """'+31.2304+121.4737+004.000/' -> (31.2304, 121.4737)."""
    if not value:
        return None
    m = _ISO6709.match(value)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    return lat, lon


def _merged_tags(fmt: dict[str, Any], video: dict[str, Any]) -> dict[str, str]:
    # format-level tags win; lower-cased keys because vendors disagree on case
    tags = {k.lower(): v for k, v in video.get("tags", {}).items()}
    tags.update({k.lower(): v for k, v in fmt.get("tags", {}).items()})
    return tags


def _created_at(tags: dict[str, str]) -> datetime | None:
    # Apple's key carries the local UTC offset, so prefer it over creation_time
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        dt = parse_creation_time(tags.get(key))
        if dt is not None:
            return dt
    return None


def _gps(tags: dict[str, str]) -> tuple[float, float] | None:
    for key in ("com.apple.quicktime.location.iso6709", "location", "location-eng"):
        gps = parse_iso6709(tags.get(key))
        if gps is not None:
            return gps
    return None


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None
