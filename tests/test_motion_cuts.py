from __future__ import annotations

import pytest

from vlogcut.analyze.motion_cuts import merge_cut_points, motion_cut_points
from vlogcut.schemas import CutPoint, QualityFlags, Shot


def test_v_shaped_dip_uses_absolute_sample_time() -> None:
    assert motion_cut_points([4, 2, 0, 2, 4], 2, 10, 12.5) == [CutPoint(time_s=11, kind="motion")]


@pytest.mark.parametrize("margin,expected", [(0, [1, 3]), (1, []), (3, [])])
def test_edge_margin_is_strict_at_both_ends(margin: float, expected: list[float]) -> None:
    points = motion_cut_points([4, 0, 4, 0, 4], 1, 0, 4, edge_margin_s=margin)
    assert [p.time_s for p in points] == expected


@pytest.mark.parametrize("levels,expected", [([8, 2, 8, 1, 8], 0.3), ([8, 1, 8, 1, 8], 0.1)])
def test_close_minima_keep_lowest_then_earliest(levels: list[float], expected: float) -> None:
    points = motion_cut_points(levels, 10, 0, 1, min_gap_s=0.3, edge_margin_s=0)
    assert [p.time_s for p in points] == [expected]


def test_min_gap_boundary_and_multiple_sorted_candidates() -> None:
    points = motion_cut_points([8, 2, 8, 0, 8, 1, 8], 4, 0, 2)
    assert [p.time_s for p in points] == [0.25, 0.75, 1.25]


def test_plateau_uses_first_sample_and_requires_two_higher_neighbours() -> None:
    assert motion_cut_points([5, 1, 1, 1, 5], 1, 0, 5, max_relative_level=1) == [
        CutPoint(time_s=1, kind="motion")
    ]
    assert motion_cut_points([0, 0, 5, 0, 0], 1, 0, 5) == []
    # Do not move to a later plateau sample when its first sample is excluded.
    assert motion_cut_points([5, 0, 0, 5], 1, 0, 4, edge_margin_s=1) == []


@pytest.mark.parametrize("relative,expected", [(0.49, []), (0.5, [1]), (1, [1])])
def test_relative_threshold_is_inclusive(relative: float, expected: list[float]) -> None:
    points = motion_cut_points([4, 2, 4], 1, 0, 3, max_relative_level=relative)
    assert [p.time_s for p in points] == expected


@pytest.mark.parametrize("levels", [[], [1], [1, 0], [3] * 5, [0] * 5, [1, 2, 3], [3, 2, 1]])
def test_no_internal_minima(levels: list[float]) -> None:
    assert motion_cut_points(levels, 1, 0, 10) == []


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), float("-inf")])
def test_invalid_motion_raises_even_for_short_input(bad: float) -> None:
    with pytest.raises(ValueError, match="motion"):
        motion_cut_points([bad], 1, 0, 3)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_sample_rate_raises_even_for_empty_input(fps: float) -> None:
    with pytest.raises(ValueError, match="sample_fps"):
        motion_cut_points([], fps, 0, 3)


@pytest.mark.parametrize("name", ["min_gap_s", "edge_margin_s", "max_relative_level"])
@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_options(name: str, bad: float) -> None:
    with pytest.raises(ValueError, match=name):
        motion_cut_points([4, 0, 4], 1, 0, 3, **{name: bad})


@pytest.mark.parametrize("start,end", [(-1, 3), (3, 2), (0, float("inf")), (float("nan"), 3)])
def test_invalid_bounds(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="bounds"):
        motion_cut_points([4, 0, 4], 1, start, end)


def test_merge_prefers_pauses_on_either_side_and_keeps_distant_motion() -> None:
    pauses = [CutPoint(time_s=2, kind="pause")]
    motion = [CutPoint(time_s=t, kind="motion") for t in [3, 2.05, 1.95]]
    expected = [*pauses, motion[0]]
    assert merge_cut_points(pauses, motion) == expected
    assert merge_cut_points(motion, pauses) == expected
    assert [p.time_s for p in motion] == [3, 2.05, 1.95]


def test_merge_inclusive_tolerance_and_exact_duplicates() -> None:
    pause = CutPoint(time_s=2, kind="pause")
    motion = [CutPoint(time_s=t, kind="motion") for t in [2, 2.125, 1.875]]
    assert merge_cut_points([pause, pause], motion, tolerance_s=0.125) == [pause]
    assert merge_cut_points([pause], motion, tolerance_s=0) == [motion[2], pause, motion[1]]
    assert merge_cut_points([], []) == []


def test_merge_preserves_distinct_same_kind_times() -> None:
    points = [CutPoint(time_s=t, kind="motion") for t in [2, 2.05]]
    points += [CutPoint(time_s=t, kind="pause") for t in [3, 3.05]]
    assert merge_cut_points(points[::-1], points) == points


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_merge_rejects_invalid_tolerance_and_times(bad: float) -> None:
    with pytest.raises(ValueError, match="tolerance_s"):
        merge_cut_points([], [], tolerance_s=bad)
    with pytest.raises(ValueError, match="times"):
        merge_cut_points([CutPoint(time_s=bad, kind="motion")], [])


def test_motion_and_merged_output_satisfy_shot_contract() -> None:
    motion = motion_cut_points([8, 2, 8, 0, 8, 1, 8], 4, 10, 12)
    merged = merge_cut_points(motion, [CutPoint(time_s=10.3, kind="pause")])
    for points in (motion, merged):
        shot = Shot(
            shot_id="s", asset_id="a", start_s=10, end_s=12,
            quality=QualityFlags(sharpness=50, exposure_clip=0, shake=0),
            cut_points=points,
        )  # fmt: skip
        assert shot.cut_points == points
        assert Shot.model_validate_json(shot.model_dump_json()) == shot
