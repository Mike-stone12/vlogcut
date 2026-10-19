import json

import pytest

from vlogcut.plan import format_shot_table
from vlogcut.schemas import CutPoint, QualityFlags, Shot

HEADER = (
    "shot_id\tasset_id\tstart_s\tduration_s\tusable_range_rel_s\twarnings\tdup_group\t"
    "kept_for_audio\tcut_points_rel_s\trejected\treject_reasons"
)


def _shot(shot_id: str = "s1", *, rejected: bool = False) -> Shot:
    return Shot(
        shot_id=shot_id,
        asset_id="asset-full-id",
        start_s=10.0,
        end_s=16.0,
        quality=QualityFlags(
            sharpness=100,
            exposure_clip=0,
            shake=0,
            rejected=rejected,
            reject_reasons=["near_black", "violent_shake"] if rejected else [],
        ),
    )


def test_relative_times_and_fixed_precision() -> None:
    shot = _shot()
    shot.quality.usable_range = (1.125, 5.876)
    shot.cut_points = [
        CutPoint(time_s=10.5, kind="pause"),
        CutPoint(time_s=12.5, kind="pause"),
    ]

    assert format_shot_table([shot]) == (
        HEADER + '\n"s1"\t"asset-full-id"\t10.00\t6.00\t1.12..5.88\t-\t-\tfalse\t'
        "0.50:pause,2.50:pause\tfalse\t-\nOmitted rejected shots: 0"
    )


def test_warnings_duplicate_group_and_audio_flag() -> None:
    shot = _shot()
    shot.dup_group = "same-take"
    shot.quality.warnings = ["shaky", "kept_for_audio", "blur"]

    row = format_shot_table([shot]).splitlines()[1].split("\t")

    assert row[4] == "0.00..6.00"
    assert row[5:9] == ['["shaky","kept_for_audio","blur"]', '"same-take"', "true", "-"]


def test_rejected_shots_omitted_or_shown_with_reasons() -> None:
    shots = [_shot("bad1", rejected=True), _shot(), _shot("bad2", rejected=True)]

    default = format_shot_table(shots).splitlines()
    assert len(default) == 3
    assert default[1].startswith('"s1"\t')
    assert default[-1] == "Omitted rejected shots: 2"

    included = format_shot_table(shots, include_rejected=True).splitlines()
    assert len(included) == 5
    assert [row.split("\t")[0] for row in included[1:-1]] == ['"bad1"', '"s1"', '"bad2"']
    for row in (included[1], included[3]):
        assert row.endswith('\ttrue\t["near_black","violent_shake"]')
    assert included[-1] == "Omitted rejected shots: 0"


@pytest.mark.parametrize("include_rejected", [False, True])
def test_empty_input(include_rejected: bool) -> None:
    assert format_shot_table([], include_rejected=include_rejected) == (
        HEADER + "\nOmitted rejected shots: 0"
    )


def test_all_rejected() -> None:
    assert format_shot_table([_shot(rejected=True)]) == HEADER + "\nOmitted rejected shots: 1"


def test_determinism_order_and_no_mutation() -> None:
    shots = (_shot("z"), _shot("a"))
    shots[1].start_s = 0.0
    shots[1].asset_id = "asset-full-id-other"
    before = [shot.model_dump() for shot in shots]

    first = format_shot_table(shots)
    assert first == format_shot_table(shots)
    rows = [line.split("\t") for line in first.splitlines()[1:-1]]
    assert [row[0] for row in rows] == ['"z"', '"a"']
    assert [row[1] for row in rows] == ['"asset-full-id"', '"asset-full-id-other"']
    assert [shot.model_dump() for shot in shots] == before


def test_text_cannot_add_rows_or_columns() -> None:
    shot = _shot('id\nwith\ttabs"')
    shot.asset_id = "asset\r\nname"
    shot.dup_group = "group\tname"
    shot.quality.warnings = ["warning\ncontinued"]
    shot.quality.reject_reasons = ["reason\tcontinued"]

    lines = format_shot_table([shot]).splitlines()
    assert len(lines) == 3
    row = lines[1].split("\t")
    assert len(row) == len(HEADER.split("\t"))
    assert json.loads(row[0]) == shot.shot_id
    assert json.loads(row[1]) == shot.asset_id
    assert json.loads(row[5]) == shot.quality.warnings
    assert json.loads(row[6]) == shot.dup_group
    assert json.loads(row[10]) == shot.quality.reject_reasons
