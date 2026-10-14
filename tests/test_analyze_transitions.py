from __future__ import annotations

import pytest

from vlogcut.analyze import ShotParams, detect_shots
from vlogcut.analyze.transitions import merge_cuts
from vlogcut.ingest import ffmpeg, make_proxy, probe_asset

np = pytest.importorskip("numpy")
from vlogcut.analyze.transitions import find_dissolves  # noqa: E402  (needs numpy)

FPS = 15


def _image(seed: int) -> np.ndarray:
    """Smooth 64x64 'scene': blocky noise, like large objects in a frame."""
    rng = np.random.default_rng(seed)
    return np.kron(rng.random((8, 8)), np.ones((8, 8))).astype(np.float32)


def _hold(img: np.ndarray, n: int, seed: int = 0, noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, (n, *img.shape)), 0, 1).astype(np.float32)


def _blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    alphas = np.arange(1, n + 1) / (n + 1)
    return np.stack([(1 - t) * a + t * b for t in alphas]).astype(np.float32)


# ---------------------------------------------------------------------------
# pure detector on synthetic frame stacks
# ---------------------------------------------------------------------------


def test_cross_dissolve_found_with_correct_span():
    a, b = _image(1), _image(2)
    frames = np.concatenate([_hold(a, 30, 1), _blend(a, b, 14), _hold(b, 30, 2)])
    spans = find_dissolves(frames, FPS)
    assert len(spans) == 1
    start, end = spans[0]
    # blend occupies frames 30..43; its ends are the last A (29) and first B (44)
    assert (start + end) / 2 == pytest.approx(36.5 / FPS, abs=1 / FPS)
    assert end - start == pytest.approx(15 / FPS, abs=2 / FPS)


def test_fade_through_black_gives_two_transitions():
    a, b, black = _image(3), _image(4), np.zeros((64, 64), np.float32)
    frames = np.concatenate(
        [
            _hold(a, 20),
            _blend(a, black, 7),
            _hold(black, 10, noise=0),
            _blend(black, b, 7),
            _hold(b, 20),
        ]
    )
    assert len(find_dissolves(frames, FPS)) == 2


def test_hard_cut_is_not_a_dissolve():
    frames = np.concatenate([_hold(_image(5), 30), _hold(_image(6), 30)])
    assert find_dissolves(frames, FPS) == []


def test_exposure_ramp_is_not_a_dissolve():
    a = _image(7) * 0.6 + 0.2
    gains = np.concatenate([np.full(20, 0.5), np.linspace(0.5, 1.4, 15), np.full(20, 1.4)])
    frames = np.clip(np.stack([a * g for g in gains]), 0, 1).astype(np.float32)
    assert find_dissolves(frames, FPS) == []


def test_pan_is_not_a_dissolve():
    wide = np.kron(np.random.default_rng(8).random((8, 40)), np.ones((8, 8)))  # 64x320
    frames = np.stack([wide[:, x : x + 64] for x in range(0, 240, 4)]).astype(np.float32)
    assert find_dissolves(frames, FPS) == []


def test_static_shot_with_noise_is_not_a_dissolve():
    assert find_dissolves(_hold(_image(9), 60, noise=0.03), FPS) == []


def test_merge_cuts_drops_hard_cuts_inside_a_dissolve():
    cuts = merge_cuts([1.0, 3.6, 8.0], [(3.0, 4.0)], min_gap_s=0.5)
    assert cuts == [1.0, 3.5, 8.0]


# ---------------------------------------------------------------------------
# end to end on a generated xfade clip
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_detect_shots_cuts_at_dissolve_midpoint(tmp_path):
    clip = tmp_path / "dissolve.mp4"
    ffmpeg.run(
        "ffmpeg",
        ["-v", "error", "-y"]
        + ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4"]
        + [
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=size=320x240:rate=30,trim=duration=4,setpts=PTS-STARTPTS",
        ]
        + ["-filter_complex", "[0][1]xfade=transition=fade:duration=1:offset=3"]
        + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip)],
    )
    asset = make_proxy(probe_asset(clip), tmp_path / "ws")
    spans = detect_shots(asset, tmp_path / "ws")
    # analysis runs at 15 fps, so the cut is exact to within one sample
    assert [s.start_s for s in spans] == pytest.approx([0.0, 3.5], abs=1.5 / FPS)

    without = detect_shots(asset, tmp_path / "ws", ShotParams(dissolves=False))
    assert len(without) == 1
