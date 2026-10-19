"""Shared fixtures. Media is generated on the fly with ffmpeg; nothing is committed."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from vlogcut.ingest import ffmpeg
from vlogcut.learn.ranker import FEATURE_NAMES, RankerModel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def ranker_model() -> RankerModel:
    return RankerModel(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * len(FEATURE_NAMES),
        scale=[1.0] * len(FEATURE_NAMES),
        weights=[float(name == "interest") for name in FEATURE_NAMES],
    )


def _have_ffmpeg() -> bool:
    try:
        ffmpeg.find_binary("ffmpeg")
        ffmpeg.find_binary("ffprobe")
    except ffmpeg.FFmpegNotFoundError:
        return False
    return True


def _have_scenedetect() -> bool:
    return importlib.util.find_spec("scenedetect") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = {
        "ffmpeg": not _have_ffmpeg() and "ffmpeg/ffprobe not available",
        "scenedetect": not _have_scenedetect() and 'needs pip install -e ".[analyze]"',
    }
    for item in items:
        for marker, reason in missing.items():
            if reason and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def ffprobe_json() -> Callable[[str], dict[str, Any]]:
    def load(name: str) -> dict[str, Any]:
        return json.loads((FIXTURES / "ffprobe" / f"{name}.json").read_text(encoding="utf-8"))

    return load


@pytest.fixture
def make_clip(tmp_path: Path) -> Callable[..., Path]:
    """Generate a tiny test clip (default: 1 s, 320x240, 30 fps, with a sine tone)."""

    def make(
        name: str = "clip.mp4",
        *,
        duration: float = 1.0,
        size: str = "320x240",
        fps: int = 30,
        audio: bool = True,
        rotation: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        out = tmp_path / name
        args = ["-v", "error", "-y", "-f", "lavfi"]
        args += ["-i", f"testsrc2=size={size}:rate={fps}:duration={duration}"]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
        args += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
        for key, value in (metadata or {}).items():
            args += ["-metadata", f"{key}={value}"]
        args += ["-movflags", "+use_metadata_tags"]
        if rotation is None:
            ffmpeg.run("ffmpeg", [*args, str(out)])
            return out
        # display rotation can only be attached when remuxing an existing file
        raw = tmp_path / f"_unrotated_{name}"
        ffmpeg.run("ffmpeg", [*args, str(raw)])
        ffmpeg.run(
            "ffmpeg",
            ["-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(raw)]
            + ["-c", "copy", "-map_metadata", "0", "-movflags", "+use_metadata_tags", str(out)],
        )
        raw.unlink()
        return out

    return make


# deterministic lavfi sources that look nothing alike, so every boundary is a hard cut
LAVFI_SOURCES = ("testsrc2", "mandelbrot", "smptehdbars", "rgbtestsrc", "yuvtestsrc")


@pytest.fixture
def make_edited_clip(tmp_path: Path) -> Callable[..., Path]:
    """Generate a finished-looking clip: shots of the given lengths hard-cut together.

    Consecutive shots come from different sources, so the cut times are exactly the
    cumulative sums of `shot_lengths`. No audio.
    """

    def make(
        name: str = "edited.mp4",
        *,
        shot_lengths: Sequence[float],
        size: str = "320x240",
        fps: int = 30,
    ) -> Path:
        out = tmp_path / name
        args = ["-v", "error", "-y"]
        for i, length in enumerate(shot_lengths):
            src = LAVFI_SOURCES[i % len(LAVFI_SOURCES)]
            args += ["-f", "lavfi", "-i", f"{src}=size={size}:rate={fps},trim=duration={length}"]
        inputs = "".join(f"[{i}]" for i in range(len(shot_lengths)))
        args += ["-filter_complex", f"{inputs}concat=n={len(shot_lengths)}:v=1:a=0"]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)]
        ffmpeg.run("ffmpeg", args)
        return out

    return make
