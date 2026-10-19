"""Pure-Python linear pairwise logistic ranker; no media or model-service access.

Rejected shots are never filtered here: callers decide which shots to use.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from vlogcut.schemas import Shot

# Version-1 feature order, shared by extraction, training and persisted models.
FEATURE_NAMES: tuple[str, ...] = (
    "duration_s",
    "usable_duration_s",
    "usable_share",
    "usable_range_missing",
    "sharpness",
    "exposure_clip",
    "shake",
    "warning_count",
    "cut_point_count",
    "interest",
    "interest_missing",
)


def shot_features(shot: Shot) -> dict[str, float]:
    """Return finite floats in FEATURE_NAMES order, using only existing fields.

    Durations are seconds; usable_share is usable duration / shot duration.
    Quality metrics retain their raw units; warnings and cut points are counts.
    Missing usable_range means the whole shot, per QualityFlags. Missing interest
    uses the neutral midpoint (3) of its 1..5 range. Both have 0/1 missing flags.
    Text, identifiers, absolute timestamps and embeddings are not features.
    Non-finite measurements raise ValueError rather than poisoning training.
    """
    q = shot.quality
    duration = shot.duration_s
    usable = duration if q.usable_range is None else q.usable_range[1] - q.usable_range[0]
    values = (
        duration,
        usable,
        usable / duration,
        float(q.usable_range is None),
        q.sharpness,
        q.exposure_clip,
        q.shake,
        float(len(q.warnings)),
        float(len(shot.cut_points)),
        float(3 if shot.interest is None else shot.interest),
        float(shot.interest is None),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("shot features must be finite")
    return dict(zip(FEATURE_NAMES, map(float, values), strict=True))


class RankerModel(BaseModel):
    """Versioned analysis artefact; all vectors follow feature_names order."""

    model_config = ConfigDict(allow_inf_nan=False)

    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    weights: list[float]
    version: int = 1

    @model_validator(mode="after")
    def _check_vectors(self) -> RankerModel:
        if self.version != 1 or tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("unsupported ranker version or feature names")
        if any(len(v) != len(FEATURE_NAMES) for v in (self.mean, self.scale, self.weights)):
            raise ValueError("ranker vectors must match feature_names length")
        if any(s <= 0 for s in self.scale):
            raise ValueError("ranker scales must be positive")
        return self


def train_pairwise(
    pairs: Sequence[tuple[Shot, Shot]],
    *,
    epochs: int = 200,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> RankerModel:
    """Fit chosen-minus-skipped preferences with deterministic full-batch descent.

    Standardise over all shot occurrences (including repeats), using population
    standard deviation and scale 1 for constant features. Minimise mean pairwise
    logistic loss + l2 / 2 * sum(weights**2), starting with zero weights. There is
    no intercept: it cancels in pair differences. Zero epochs returns zero weights.
    """
    if not pairs:
        raise ValueError("at least one shot pair is required")
    if epochs < 0 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("epochs must be non-negative, lr positive and l2 non-negative")
    rows = [list(shot_features(shot).values()) for pair in pairs for shot in pair]
    columns = list(zip(*rows, strict=True))
    mean = [statistics.mean(column) for column in columns]
    scale = [statistics.pstdev(column) or 1.0 for column in columns]
    standardised = [[(x - m) / s for x, m, s in zip(row, mean, scale, strict=True)] for row in rows]
    differences = [
        [a - b for a, b in zip(chosen, skipped, strict=True)]
        for chosen, skipped in zip(standardised[::2], standardised[1::2], strict=True)
    ]
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for difference in differences:
            margin = sum(w * d for w, d in zip(weights, difference, strict=True))
            # sigmoid(-margin), avoiding exp overflow for either sign.
            e = math.exp(-abs(margin))
            probability = e / (1.0 + e) if margin >= 0 else 1.0 / (1.0 + e)
            for j, d in enumerate(difference):
                gradient[j] -= probability * d
        weights = [
            w - lr * (g / len(pairs) + l2 * w) for w, g in zip(weights, gradient, strict=True)
        ]
    return RankerModel(feature_names=list(FEATURE_NAMES), mean=mean, scale=scale, weights=weights)


def score(model: RankerModel, shot: Shot) -> float:
    """Return the linear score using the model's stored standardisation."""
    features = shot_features(shot)
    return sum(
        w * ((features[name] - m) / s)
        for name, m, s, w in zip(
            model.feature_names, model.mean, model.scale, model.weights, strict=True
        )
    )


def rank(model: RankerModel, shots: Sequence[Shot]) -> list[Shot]:
    """Return all shots by descending score; equal scores retain input order."""
    return sorted(shots, key=lambda shot: score(model, shot), reverse=True)
