from __future__ import annotations

from pathlib import Path

import pytest

from vlogcut.analyze import ShotDetectionError, ShotParams, detect_shots
from vlogcut.analyze import shots as shots_mod
from vlogcut.analyze.shots import (
    cuts_to_spans,
    keyframe_times,
    merge_short,
    postprocess,
    split_long,
)
from vlogcut.ingest import ffmpeg, make_proxy, probe_asset
from vlogcut.schemas import MediaAsset

# ---------------------------------------------------------------------------
# pure span logic
# ---------------------------------------------------------------------------


def test_cuts_to_spans_is_contiguous_and_drops_out_of_range_cuts():
    spans = cuts_to_spans([4.0, 2.0, 2.0, 0.0, 6.0, 9.0], 6.0)
    assert spans == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert cuts_to_spans([], 3.0) == [(0.0, 3.0)]


def test_merge_short_joins_the_shorter_neighbour():
    spans = [(0.0, 3.0), (3.0, 3.2), (3.2, 4.0), (4.0, 10.0)]
    assert merge_short(spans, 0.5) == [(0.0, 3.0), (3.0, 4.0), (4.0, 10.0)]


def test_merge_short_edges_and_single_span():
    assert merge_short([(0.0, 0.2), (0.2, 5.0), (5.0, 5.3)], 0.5) == [(0.0, 5.3)]
    assert merge_short([(0.0, 0.3)], 0.5) == [(0.0, 0.3)]  # nothing to merge into


def test_split_long_makes_equal_parts_within_max():
    parts = split_long([(0.0, 25.0), (25.0, 30.0)], 10.0)
    assert parts == pytest.approx([(0.0, 25 / 3), (25 / 3, 50 / 3), (50 / 3, 25.0), (25.0, 30.0)])
    assert split_long([(0.0, 10.0)], 10.0) == [(0.0, 10.0)]  # exactly max: untouched


def test_postprocess_covers_whole_asset():
    params = ShotParams(min_shot_s=0.5, max_shot_s=10.0)
    spans = postprocess([2.0, 2.1, 30.0], 45.0, params)
    assert spans[0][0] == 0.0 and spans[-1][1] == 45.0
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert all(0.5 <= e - s <= 10.0 for s, e in spans)


def test_keyframe_times():
    assert keyframe_times((2.0, 6.0)) == [3.0, 4.0, 5.0]


def test_detect_shots_needs_a_proxy(tmp_path):
    asset = MediaAsset(asset_id="a", path="x.mp4", duration_s=1, fps=30, width=2, height=2)
    with pytest.raises(ShotDetectionError, match="proxy"):
        detect_shots(asset, tmp_path)


# ---------------------------------------------------------------------------
# real detector on generated footage
# ---------------------------------------------------------------------------


def _three_shot_clip(tmp_path: Path) -> Path:
    """6 s clip with hard cuts at 2 s and 4 s between visually different sources."""
    out = tmp_path / "three_shots.mp4"
    srcs = [
        "testsrc2=size=320x240:rate=30:duration=2",
        "mandelbrot=size=320x240:rate=30,trim=duration=2",
        "smptehdbars=size=320x240:rate=30:duration=2",
    ]
    args = ["-v", "error", "-y"]
    for s in srcs:
        args += ["-f", "lavfi", "-i", s]
    args += ["-filter_complex", "[0][1][2]concat=n=3:v=1:a=0"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)]
    ffmpeg.run("ffmpeg", args)
    return out


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_detect_shots_finds_hard_cuts_and_writes_keyframes(tmp_path):
    asset = make_proxy(probe_asset(_three_shot_clip(tmp_path)), tmp_path / "ws")
    spans = detect_shots(asset, tmp_path / "ws")
    frame = 1 / 30
    assert len(spans) == 3
    assert [s.start_s for s in spans] == pytest.approx([0.0, 2.0, 4.0], abs=frame)
    assert spans[-1].end_s == pytest.approx(asset.duration_s)
    assert [s.shot_id for s in spans] == [f"{asset.asset_id}_{i:03d}" for i in range(3)]
    for span in spans:
        assert len(span.keyframes) == 3
        assert all(Path(k).stat().st_size > 0 for k in span.keyframes)


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_detect_shots_cache_hit_and_param_change(tmp_path, monkeypatch):
    asset = make_proxy(probe_asset(_three_shot_clip(tmp_path)), tmp_path / "ws")
    first = detect_shots(asset, tmp_path / "ws")

    def boom(*a, **k):
        raise AssertionError("detector should not run on a cache hit")

    monkeypatch.setattr(shots_mod, "detect_cuts", boom)
    assert detect_shots(asset, tmp_path / "ws") == first

    monkeypatch.undo()
    split = detect_shots(asset, tmp_path / "ws", ShotParams(max_shot_s=1.0))
    assert len(split) == 6  # new params -> recomputed, each 2 s shot split in two
