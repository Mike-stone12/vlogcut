from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlogcut.analyze.look import LookParams, LookStats, load_rgb_frames, look_stats, pool_look
from vlogcut.ingest import ffmpeg


@pytest.mark.parametrize(
    "rgb,brightness,saturation,warmth",
    [
        ((0, 0, 0), 0, 0, 0),
        ((255, 255, 255), 1, 0, 0),
        ((255, 0, 0), 0.2126, 1, 1),
        ((0, 0, 255), 0.0722, 1, -1),
    ],
)
def test_solid_colours(
    rgb: tuple[int, int, int], brightness: float, saturation: float, warmth: float
) -> None:
    np = pytest.importorskip("numpy")
    result = look_stats(np.full((1, 8, 8, 3), rgb, dtype=np.uint8))
    assert result is not None
    assert result.brightness == pytest.approx(brightness)
    assert result.saturation == pytest.approx(saturation)
    assert result.warmth == pytest.approx(warmth)
    assert result.frames == 1


def test_uint8_matches_float() -> None:
    np = pytest.importorskip("numpy")
    frames = np.full((1, 8, 8, 3), 128, dtype=np.uint8)
    integer = look_stats(frames)
    floating = look_stats(frames.astype(np.float32) / 255)
    assert integer is not None and floating is not None
    assert integer.model_dump() == pytest.approx(floating.model_dump(), abs=1e-6)


def test_float_clipping_without_mutation() -> None:
    np = pytest.importorskip("numpy")
    frames = np.array([[[[1.5, -0.5, 0.0]]]])
    original = frames.copy()
    assert look_stats(frames) == LookStats(brightness=0.2126, saturation=1, warmth=1, frames=1)
    np.testing.assert_array_equal(frames, original)


def test_equal_pixel_and_frame_pooling() -> None:
    np = pytest.importorskip("numpy")
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 255  # Four red pixels, then one blue and three black.
    frames[1, 0, 0, 2] = 255
    result = look_stats(frames)
    assert result is not None
    assert result.brightness == pytest.approx((4 * 0.2126 + 0.0722) / 8)
    assert result.saturation == pytest.approx(5 / 8)
    assert result.warmth == pytest.approx(3 / 8)
    assert result.frames == 2


def test_zero_frames() -> None:
    np = pytest.importorskip("numpy")
    assert look_stats(np.empty((0, 8, 8, 3), dtype=np.uint8)) is None


@pytest.mark.parametrize("shape", [(8, 8, 3), (1, 8, 8), (1, 8, 8, 4), (0,), (1, 0, 8, 3)])
def test_wrong_shape(shape: tuple[int, ...]) -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="shape"):
        look_stats(np.zeros(shape, dtype=np.uint8))


def test_unsupported_dtype() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="uint8 or floating-point"):
        look_stats(np.zeros((1, 1, 1, 3), dtype=np.int32))


def test_frame_weighted_pooling() -> None:
    red = LookStats(brightness=0.2126, saturation=1, warmth=1, frames=1)
    grey = LookStats(brightness=0.5, saturation=0, warmth=0, frames=3)
    pooled = pool_look([None, red, grey, None])
    assert pooled is not None
    assert pooled.brightness == pytest.approx((0.2126 + 3 * 0.5) / 4)
    assert pooled.saturation == pytest.approx(0.25)
    assert pooled.warmth == pytest.approx(0.25)
    assert pooled.frames == 4
    assert pool_look([red]) == red
    assert pool_look([]) is None
    assert pool_look([None, None]) is None


@pytest.mark.parametrize("field", ["brightness", "saturation", "warmth", "frames"])
@pytest.mark.parametrize("value", [-2, 2.5, float("nan"), float("inf")])
def test_stats_validation(field: str, value: float) -> None:
    values = dict(brightness=0.5, saturation=0.5, warmth=0, frames=1)
    values[field] = value
    with pytest.raises(ValidationError):
        LookStats(**values)


@pytest.mark.parametrize(
    "field,value",
    [("sample_fps", 0), ("sample_fps", -1), ("sample_fps", float("inf")), ("size", 7)],
)
def test_params_validation(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        LookParams(**{field: value})


def test_decoder_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")

    def empty(name: str, args: list[str]) -> bytes:
        assert name == "ffmpeg"
        assert "fps=1.0,scale=64:64,format=rgb24" in args
        return b""

    monkeypatch.setattr(ffmpeg, "run_bytes", empty)
    frames = load_rgb_frames("proxy.mp4")
    assert frames.shape == (0, 64, 64, 3)
    assert frames.dtype == np.uint8


def test_decoder_caps_output(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")

    def decode(name: str, args: list[str]) -> bytes:
        assert args[args.index("-frames:v") + 1] == "32"
        return bytes(32 * 64 * 64 * 3)

    monkeypatch.setattr(ffmpeg, "run_bytes", decode)
    assert len(load_rgb_frames("proxy.mp4", max_frames=32)) == 32
    with pytest.raises(ValueError, match="max_frames"):
        load_rgb_frames("proxy.mp4", max_frames=0)


@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    "colour,brightness,warmth_sign",
    [("0xff8000", 0.2126 + 0.7152 * 128 / 255, 1), ("blue", 0.0722, -1)],
)
@pytest.mark.parametrize("params", [LookParams(), LookParams(sample_fps=2.5, size=16)])
def test_decode_solid_colour(
    tmp_path: Path, colour: str, brightness: float, warmth_sign: int, params: LookParams
) -> None:
    np = pytest.importorskip("numpy")
    path = tmp_path / "solid.mp4"
    duration = 2
    ffmpeg.run(
        "ffmpeg",
        [
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s=80x48:r=25:d={duration}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
    )
    frames = load_rgb_frames(path, params)
    assert abs(len(frames) - duration * params.sample_fps) <= 1
    assert frames.shape[1:] == (params.size, params.size, 3)
    assert frames.dtype == np.uint8
    result = look_stats(frames)
    assert result is not None
    assert result.frames == len(frames)
    assert result.warmth * warmth_sign > 0
    assert result.brightness == pytest.approx(brightness, abs=0.05)
