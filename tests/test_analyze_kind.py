from typing import Literal

import pytest

from vlogcut.analyze.kind import classify_kind
from vlogcut.schemas import QualityFlags, Shot


@pytest.mark.parametrize(
    "warnings,rejected,expected",
    [
        (["shaky", "kept_for_audio"], False, "talking"),
        ([], False, "broll"),
        (["shaky"], False, "broll"),
        ([], True, None),
    ],
)
def test_classify_kind(warnings: list[str], rejected: bool, expected: str | None) -> None:
    shot = Shot(
        shot_id="s", asset_id="a", start_s=0, end_s=4,
        quality=QualityFlags(
            sharpness=50, exposure_clip=0, shake=0, warnings=warnings, rejected=rejected
        ),
    )  # fmt: skip
    before = shot.model_dump_json()
    assert classify_kind(shot) == expected
    assert shot.model_dump_json() == before


@pytest.mark.parametrize("kind", ["talking", "broll"])
def test_classify_kind_prefers_explicit_kind(kind: Literal["talking", "broll"]) -> None:
    shot = Shot(
        shot_id="s", asset_id="a", start_s=0, end_s=4, kind=kind,
        quality=QualityFlags(
            sharpness=50, exposure_clip=0, shake=0,
            warnings=[] if kind == "talking" else ["kept_for_audio"],
        ),
    )  # fmt: skip
    assert classify_kind(shot) == kind
