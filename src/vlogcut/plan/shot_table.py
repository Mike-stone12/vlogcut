"""Deterministic planner input: tab-separated columns, one line per shot.

Column order: shot_id, asset_id, start_s, duration_s, usable_range_rel_s,
warnings, dup_group, kept_for_audio, cut_points_rel_s, caption, tags, interest,
rejected, reject_reasons. Interest is an integer when present.
With ranker_scores supplied, rank appears after interest with two decimal places:
higher means more likely chosen, relative within this table only.

Empty or None values use ``-``. A None usable_range means the whole shot, so it
is expanded to 0..duration. Times are seconds with two decimal places: start_s
is source-absolute; usable_range and cut_points are shot-relative, like Clip
in_s/out_s. Full IDs and other text are JSON-quoted (lists use compact JSON),
escaping tabs and newlines so each shot occupies exactly one line.

Shots and list values retain input order. kept_for_audio is derived from the
quality warning of that name. Rejected shots are omitted by default; the last
line always reports the omitted count. No trailing newline is added.
"""

from collections.abc import Mapping, Sequence
from json import dumps
from math import isfinite

from vlogcut.schemas import Shot

_HEADER = (
    "shot_id\tasset_id\tstart_s\tduration_s\tusable_range_rel_s\twarnings\tdup_group\t"
    "kept_for_audio\tcut_points_rel_s\tcaption\ttags\tinterest\trejected\treject_reasons"
)


def _text(value: str | Sequence[str] | None) -> str:
    """Render optional text without introducing row or column separators."""
    return dumps(value, ensure_ascii=True, separators=(",", ":")) if value else "-"


def format_shot_table(
    shots: Sequence[Shot],
    *,
    include_rejected: bool = False,
    ranker_scores: Mapping[str, float] | None = None,
) -> str:
    """Format metadata and optional precomputed scores without changing shots.

    Scores for supplied shots must be finite, including omitted rejected shots.
    Unknown score IDs are ignored; None preserves the original table format.
    """
    header = _HEADER
    if ranker_scores is not None:
        header = header.replace("\trejected\t", "\trank\trejected\t")
    lines = [header]
    omitted = 0
    for shot in shots:
        rank_value = "-"
        if ranker_scores is not None and shot.shot_id in ranker_scores:
            value = ranker_scores[shot.shot_id]
            if not isfinite(value):
                raise ValueError(f"Ranker score for shot {shot.shot_id!r} must be finite")
            rank_value = f"{value:.2f}"
        quality = shot.quality
        if quality.rejected and not include_rejected:
            omitted += 1
            continue
        lo, hi = quality.usable_range or (0.0, shot.duration_s)
        cut_points = ",".join(
            f"{point.time_s - shot.start_s:.2f}:{point.kind}" for point in shot.cut_points
        )
        lines.append(
            "\t".join(
                (
                    _text(shot.shot_id),
                    _text(shot.asset_id),
                    f"{shot.start_s:.2f}",
                    f"{shot.duration_s:.2f}",
                    f"{lo:.2f}..{hi:.2f}",
                    _text(quality.warnings),
                    _text(shot.dup_group),
                    str("kept_for_audio" in quality.warnings).lower(),
                    cut_points or "-",
                    _text(shot.caption),
                    _text(shot.tags),
                    str(shot.interest) if shot.interest is not None else "-",
                    *((rank_value,) if ranker_scores is not None else ()),
                    str(quality.rejected).lower(),
                    _text(quality.reject_reasons),
                )
            )
        )
    lines.append(f"Omitted rejected shots: {omitted}")
    return "\n".join(lines)
