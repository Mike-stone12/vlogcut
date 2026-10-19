"""Feedback training orchestration and readers for existing pipeline outputs."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from vlogcut.feedback import preference_pairs
from vlogcut.learn.feedback_pairs import shot_pairs_from_preferences
from vlogcut.learn.ranker import RankerModel, train_pairwise
from vlogcut.schemas import EDL, Feedback, Shot


class RankerTrainingError(ValueError):
    """Feedback or workspace outputs cannot supply ranker training data."""


def train_from_feedback(
    feedback: Sequence[Feedback],
    edls: Mapping[str, EDL],
    shots: Sequence[Shot],
    *,
    context: str = "",
) -> tuple[RankerModel, int]:
    """Train using existing pair derivation and math, returning the shot-pair count.

    The underlying helpers leave quality filtering to callers. Here, exclude
    quality-rejected shots on both sides; a rejected *variant* is still a valid
    source of negative examples. Shared/unknown shots and duplicate pairs follow
    shot_pairs_from_preferences policy. Context labels preferences, not features.
    """
    preferences = preference_pairs(feedback, context=context)
    eligible = [shot for shot in shots if not shot.quality.rejected]
    pairs = shot_pairs_from_preferences(preferences, edls, eligible)
    if not pairs:
        raise RankerTrainingError("no training shot pairs derived from feedback")
    return train_pairwise(pairs), len(pairs)


def load_training_workspace(workspace: Path) -> tuple[dict[str, EDL], list[Shot]]:
    """Read edl/<stem>.json and the pipeline's complete shots.json snapshot.

    Ignore violation reports, directories and JSON that does not validate as an
    EDL. Keyframe paths need no resolution: training never opens media files.
    """
    if not workspace.is_dir():
        raise RankerTrainingError(f"workspace not found: {workspace}")
    edls: dict[str, EDL] = {}
    for path in sorted((workspace / "edl").glob("*.json")):
        if not path.is_file() or path.name.endswith(".violations.json"):
            continue
        try:
            edls[path.stem] = EDL.model_validate_json(path.read_bytes())
        except ValidationError:
            continue
    if not edls:
        raise RankerTrainingError(f"no EDLs found in {workspace / 'edl'}")
    path = workspace / "shots.json"
    try:
        shots = TypeAdapter(list[Shot]).validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RankerTrainingError(f"cannot load analyzed shots from {path}: {exc}") from exc
    if not shots:
        raise RankerTrainingError(f"no analyzed shots found in {path}")
    return edls, shots
