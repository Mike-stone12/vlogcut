from __future__ import annotations

import pytest

from vlogcut.analyze.transitions import merge_black_fades


@pytest.mark.parametrize(
    ("dissolves", "luma", "expected_spans", "expected_cuts"),
    [
        ([], [], [], []),
        ([(0, 1)], [0] * 10, [(0, 1)], [0.5]),
        ([(0, 1), (1.5, 4)], [0.03] * 10, [(0, 4)], [1.25]),
        ([(0, 1), (1, 3)], [0.03] * 10, [(0, 3)], [1]),  # shared junction
        ([(0, 1), (2, 4)], [0.03] * 10, [(0, 4)], [1.5]),  # gap bound inclusive
        ([(0, 1), (1.5, 4)], [0.5] * 10, [(0, 1), (1.5, 4)], [0.5, 2.75]),
        ([(0, 1), (1.5, 4)], [0.06] * 10, [(0, 1), (1.5, 4)], [0.5, 2.75]),
        ([(0, 1), (2.5, 4)], [0] * 10, [(0, 1), (2.5, 4)], [0.5, 3.25]),
        ([(0, 1), (1.5, 4)], [], [(0, 1), (1.5, 4)], [0.5, 2.75]),
        ([(0, 1), (0.5, 4)], [0] * 10, [(0, 1), (0.5, 4)], [0.5, 2.25]),
        # A bright sample inside an otherwise black gap prevents merging.
        ([(0, 1), (2, 4)], [0, 0, 0, 0.1, 0, 0], [(0, 1), (2, 4)], [0.5, 3]),
    ],
)
def test_merge_black_fades(
    dissolves: list[tuple[float, float]],
    luma: list[float],
    expected_spans: list[tuple[float, float]],
    expected_cuts: list[float],
) -> None:
    original = dissolves.copy(), luma.copy()
    assert merge_black_fades(dissolves, luma, fps=2) == (expected_spans, expected_cuts)
    assert (dissolves, luma) == original


def test_merge_black_fades_keeps_other_transitions_and_custom_thresholds() -> None:
    found = [(0, 1), (2.5, 4), (7, 8)]
    assert merge_black_fades(found, [0.1] * 20, 2, max_gap_s=1.5, max_luma=0.2) == (
        [(0, 4), (7, 8)],
        [1.75, 7.5],
    )
