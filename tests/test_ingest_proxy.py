from __future__ import annotations

import json
from fractions import Fraction

import pytest

from vlogcut.ingest import ProxyError, ffmpeg, make_proxy, probe_asset, proxy_args, proxy_path_for
from vlogcut.ingest.probe import rotation_of
from vlogcut.ingest.proxy import proxy_dims, proxy_rate
from vlogcut.schemas import MediaAsset


def _asset(**kw) -> MediaAsset:
    base = dict(asset_id="abc123", path="in.mov", duration_s=10.0, fps=30.0)
    base |= dict(width=1920, height=1080)
    return MediaAsset(**{**base, **kw})


# ---------------------------------------------------------------------------
# pure (no ffmpeg needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dims", "expected"),
    [
        ((1920, 1080), (960, 540)),
        ((1080, 1920), (540, 960)),  # portrait phone footage
        ((3840, 2160), (960, 540)),
        ((886, 1920), (540, 1170)),  # odd aspect ratio stays even
        ((320, 240), (320, 240)),  # never upscale
        ((321, 241), (320, 240)),  # odd dims -> even, rounded down
    ],
)
def test_proxy_dims(dims, expected):
    assert proxy_dims(*dims) == expected


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (60.0, Fraction(30)),
        (59.94, Fraction(30)),
        (30000 / 1001, Fraction(30000, 1001)),
        (29.94, Fraction(1497, 50)),  # iPhone avg_frame_rate stays as-is
        (25.0, Fraction(25)),
        (24.0, Fraction(24)),
    ],
)
def test_proxy_rate(fps, expected):
    assert proxy_rate(fps) == expected


def test_proxy_args_with_audio():
    args = proxy_args(_asset(fps=60.0), "out.mp4")
    assert args[args.index("-vf") + 1] == "fps=30,scale=960:540,setsar=1"
    assert args[args.index("-g") + 1] == "30"
    assert "0:a:0" in args and "-an" not in args
    assert args[-1] == "out.mp4"


def test_proxy_args_without_audio_and_ntsc_rate():
    args = proxy_args(_asset(fps=30000 / 1001, width=1080, height=1920, has_audio=False), "o.mp4")
    assert args[args.index("-vf") + 1] == "fps=30000/1001,scale=540:960,setsar=1"
    assert "-an" in args and "0:a:0" not in args


def test_proxy_path_is_keyed_by_asset_id(tmp_path):
    assert proxy_path_for(_asset(), tmp_path) == tmp_path / "proxies" / "abc123.mp4"


# ---------------------------------------------------------------------------
# real ffmpeg on generated clips
# ---------------------------------------------------------------------------


def _stream_info(path) -> dict:
    out = ffmpeg.run(
        "ffprobe", ["-v", "error", "-print_format", "json", "-show_streams", str(path)]
    )
    return next(s for s in json.loads(out)["streams"] if s["codec_type"] == "video")


@pytest.mark.ffmpeg
def test_proxy_downscales_and_caps_fps(make_clip, tmp_path):
    src = probe_asset(make_clip("hi.mp4", duration=2.0, size="1280x720", fps=60))
    asset = make_proxy(src, tmp_path / "ws")
    assert asset.proxy_path == str(proxy_path_for(src, tmp_path / "ws"))
    proxy = probe_asset(asset.proxy_path)
    assert (proxy.width, proxy.height) == (960, 540)
    assert proxy.fps == 30
    assert proxy.duration_s == pytest.approx(src.duration_s, abs=0.1)  # same timeline
    assert proxy.has_audio
    assert int(_stream_info(asset.proxy_path)["nb_frames"]) == pytest.approx(60, abs=1)


@pytest.mark.ffmpeg
def test_rotated_source_gives_upright_proxy_without_rotation(make_clip, tmp_path):
    src = probe_asset(make_clip("rot.mp4", size="1280x720", rotation=90, audio=False))
    assert (src.width, src.height) == (720, 1280)
    asset = make_proxy(src, tmp_path)
    proxy = probe_asset(asset.proxy_path)
    assert (proxy.width, proxy.height) == (540, 960)
    assert rotation_of(_stream_info(asset.proxy_path)) == 0
    assert not proxy.has_audio


@pytest.mark.ffmpeg
def test_cached_proxy_is_reused_and_broken_one_rebuilt(make_clip, tmp_path):
    src = probe_asset(make_clip("c.mp4"))
    first = make_proxy(src, tmp_path)
    mtime = proxy_path_for(src, tmp_path).stat().st_mtime_ns
    assert make_proxy(src, tmp_path).proxy_path == first.proxy_path
    assert proxy_path_for(src, tmp_path).stat().st_mtime_ns == mtime  # not rebuilt

    proxy_path_for(src, tmp_path).write_bytes(b"truncated")
    rebuilt = make_proxy(src, tmp_path)
    assert probe_asset(rebuilt.proxy_path).duration_s == pytest.approx(1.0, abs=0.1)


@pytest.mark.ffmpeg
def test_failed_encode_raises_and_leaves_nothing(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"\x00" * 128)
    with pytest.raises(ProxyError):
        make_proxy(_asset(path=str(bad)), tmp_path / "ws")
    assert not list((tmp_path / "ws" / "proxies").iterdir())
