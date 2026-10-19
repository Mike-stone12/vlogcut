from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from vlogcut.learn import FEATURE_NAMES, RankerModel, rank, score, shot_features, train_pairwise
from vlogcut.schemas import CutPoint, QualityFlags, Shot


def _shot(sid: str, sharpness: float = 100.0, interest: int | None = None) -> Shot:
    return Shot(
        shot_id=sid,
        asset_id="asset",
        start_s=10.0,
        end_s=14.0,
        interest=interest,
        quality=QualityFlags(sharpness=sharpness, exposure_clip=0.1, shake=0.5),
    )


def test_feature_contract_and_missing_values() -> None:
    expected = (
        "duration_s", "usable_duration_s", "usable_share", "usable_range_missing",
        "sharpness", "exposure_clip", "shake", "warning_count", "cut_point_count",
        "interest", "interest_missing",
    )  # fmt: skip
    features = shot_features(_shot("missing"))
    assert FEATURE_NAMES == expected
    assert tuple(features) == expected
    assert all(type(value) is float and math.isfinite(value) for value in features.values())
    assert list(features.values()) == [4.0, 4.0, 1.0, 1.0, 100.0, 0.1, 0.5, 0.0, 0.0, 3.0, 1.0]

    shot = _shot("measured", interest=5)
    shot.quality.usable_range = (1.0, 3.0)
    shot.quality.warnings = ["blur", "kept_for_audio"]
    shot.cut_points = [CutPoint(time_s=12.0, kind="pause")]
    measured = shot_features(shot)
    assert tuple(measured) == expected
    assert list(measured.values()) == [4.0, 2.0, 0.5, 0.0, 100.0, 0.1, 0.5, 2.0, 1.0, 5.0, 0.0]


@pytest.mark.parametrize("feature", ["sharpness", "interest"])
def test_learns_preference_and_ranks_held_out(feature: str) -> None:
    if feature == "sharpness":
        pairs = [(_shot("chosen", v + 100), _shot("skipped", v)) for v in (10, 30, 50)]
        good, bad = _shot("sharp", 180), _shot("blurry", 20)
    else:
        pairs = [(_shot("chosen", interest=v + 2), _shot("skipped", interest=v)) for v in (1, 2)]
        good, bad = _shot("interesting", interest=5), _shot("dull", interest=1)
    model = train_pairwise(pairs)
    assert model.weights[FEATURE_NAMES.index(feature)] > 0
    assert score(model, good) > score(model, bad)
    assert rank(model, [bad, good]) == [good, bad]
    assert train_pairwise(pairs).weights == model.weights
    reloaded = RankerModel.model_validate_json(model.model_dump_json())
    assert reloaded == model
    assert score(reloaded, good) == score(model, good)
    assert score(reloaded, bad) == score(model, bad)


def test_population_standardisation_and_full_batch_update() -> None:
    # Sharpness values 6, 2, 4, 4 have mean 4 and population scale sqrt(2).
    pairs = [(_shot("a", 6), _shot("b", 2)), (_shot("c", 4), _shot("d", 4))]
    model = train_pairwise(pairs, epochs=1, lr=0.2, l2=0)
    j = FEATURE_NAMES.index("sharpness")
    assert model.mean[j] == 4.0
    assert model.scale[j] == pytest.approx(math.sqrt(2))
    assert model.weights[j] == pytest.approx(0.2 / math.sqrt(2))
    assert all(w == 0 for i, w in enumerate(model.weights) if i != j)
    assert score(model, _shot("held-out", 8)) == pytest.approx(0.4)
    regularised = train_pairwise(pairs, epochs=2, lr=0.2, l2=1)
    unregularised = train_pairwise(pairs, epochs=2, lr=0.2, l2=0)
    assert regularised.weights[j] == pytest.approx(
        unregularised.weights[j] - 0.2 * model.weights[j]
    )


def test_constant_features_and_stable_ties_keep_rejected_shots() -> None:
    first, second = _shot("z"), _shot("a")
    first.quality.rejected = True
    # Repeated non-binary-exact values (exposure 0.1) must stay zero-variance.
    model = train_pairwise([(first, second)] * 3)
    assert model.scale == [1.0] * len(FEATURE_NAMES)
    assert model.weights == [0.0] * len(FEATURE_NAMES)
    shots = [first, second]
    result = rank(model, shots)
    assert result is not shots
    assert result[0] is first and result[1] is second
    assert score(model, first) == score(model, second) == 0.0
    assert rank(model, []) == []


def test_empty_training_rejected() -> None:
    with pytest.raises(ValueError, match="at least one shot pair"):
        train_pairwise([])


def test_zero_epochs_and_large_margin() -> None:
    pairs = [(_shot("chosen", 200), _shot("skipped", 20))]
    assert train_pairwise(pairs, epochs=0).weights == [0.0] * len(FEATURE_NAMES)
    model = train_pairwise(pairs, epochs=3, lr=1000, l2=0)
    assert all(math.isfinite(w) for w in model.weights)
    # Conflicting preferences also exercise a large negative margin.
    model = train_pairwise(pairs * 2 + [(pairs[0][1], pairs[0][0])], epochs=3, lr=10000)
    assert all(math.isfinite(w) for w in model.weights)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurement_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        shot_features(_shot("invalid", value))


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", -1), ("lr", 0), ("lr", float("inf")), ("l2", -1), ("l2", float("nan"))],
)
def test_invalid_training_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        train_pairwise([(_shot("a"), _shot("b"))], **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean", []), ("weights", [float("nan")]), ("scale", [0.0] * len(FEATURE_NAMES)),
     ("feature_names", list(reversed(FEATURE_NAMES))), ("version", 2)],
)  # fmt: skip
def test_invalid_model_rejected(field: str, value: object) -> None:
    data = train_pairwise([(_shot("a"), _shot("b"))]).model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        RankerModel.model_validate(data)
