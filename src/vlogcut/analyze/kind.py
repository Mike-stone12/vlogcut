"""Shot kind from the planner's existing audio-retention heuristic, not speech VAD."""

from typing import Literal

from vlogcut.schemas import Shot


def classify_kind(shot: Shot) -> Literal["talking", "broll"] | None:
    """Prefer a measured kind, otherwise reuse the quality audio-retention signal.

    Quality analysis clears warnings on rejection, so those shots are unmeasured.
    For eligible shots, absence of kept_for_audio means broll under the existing
    planner heuristic; it does not establish that the recording contains no speech.
    """
    if shot.kind is not None:
        return shot.kind
    if shot.quality.rejected:
        return None
    return "talking" if "kept_for_audio" in shot.quality.warnings else "broll"
