"""Locate and run the ffmpeg / ffprobe binaries.

Binaries are found via `VLOGCUT_FFMPEG` / `VLOGCUT_FFPROBE` (full paths) or
`PATH`. This module is the only place that spawns them.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Literal

log = logging.getLogger(__name__)

Binary = Literal["ffmpeg", "ffprobe"]


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    def __init__(self, args: list[str], returncode: int, stderr: str):
        tail = stderr.strip().splitlines()[-3:]
        super().__init__(f"{args[0]} exited with {returncode}: {' | '.join(tail)}")
        self.args_list = args
        self.returncode = returncode
        self.stderr = stderr


def find_binary(name: Binary) -> str:
    override = os.environ.get(f"VLOGCUT_{name.upper()}")
    if override:
        if not os.path.isfile(override):
            raise FFmpegNotFoundError(f"VLOGCUT_{name.upper()}={override} does not exist")
        return override
    found = shutil.which(name)
    if not found:
        raise FFmpegNotFoundError(
            f"{name} not found on PATH; install ffmpeg or set VLOGCUT_{name.upper()}"
        )
    return found


def run(name: Binary, args: list[str], *, timeout: float | None = None) -> str:
    """Run a binary and return stdout; raise FFmpegError on a non-zero exit."""
    cmd = [find_binary(name), *args]
    log.debug("running %s", cmd)
    # utf-8 explicitly: ffprobe emits UTF-8 JSON and paths may be non-ASCII on Windows
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout
    )
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr)
    return proc.stdout


def run_bytes(name: Binary, args: list[str], *, timeout: float | None = None) -> bytes:
    """Like `run`, but return raw stdout bytes (e.g. `-f rawvideo -` frame dumps)."""
    cmd = [find_binary(name), *args]
    log.debug("running %s", cmd)
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", errors="replace"))
    return proc.stdout
