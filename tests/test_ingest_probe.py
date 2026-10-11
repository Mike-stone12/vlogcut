from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from vlogcut.ingest import ProbeError, asset_id_for, parse_probe, probe_asset
from vlogcut.ingest.probe import parse_creation_time, parse_iso6709, parse_rate

# ---------------------------------------------------------------------------
# pure parsing (no ffmpeg needed)
# ---------------------------------------------------------------------------


def test_iphone_portrait_uses_display_dims_and_apple_tags(ffprobe_json):
    asset = parse_probe(ffprobe_json("iphone_portrait_hevc"), path="IMG_4821.MOV", asset_id="a1")
    assert (asset.width, asset.height) == (1080, 1920)
    assert asset.fps == pytest.approx(29.94, abs=0.01)
    assert asset.duration_s == pytest.approx(18.033333)
    # Apple creationdate keeps the local offset
    assert asset.created_at == datetime(
        2026, 10, 1, 16, 30, 12, tzinfo=timezone(timedelta(hours=8))
    )
    assert asset.gps == pytest.approx((31.2304, 121.4737))
    assert asset.has_audio


def test_android_legacy_rotate_tag_and_fallbacks(ffprobe_json):
    asset = parse_probe(ffprobe_json("android_vfr_legacy_rotate"), path="v.mp4", asset_id="a2")
    assert (asset.width, asset.height) == (1080, 1920)  # tags.rotate fallback
    assert asset.fps == 60  # avg_frame_rate "0/0" -> r_frame_rate
    assert asset.duration_s == 42.5  # no stream duration -> format duration
    # repeated "a;b" tag; the bogus 1970 stream date is ignored
    assert asset.created_at == datetime(2026, 10, 1, 4, 34, 56, tzinfo=UTC)
    assert asset.gps == pytest.approx((22.5431, 114.0579))


def test_no_audio_and_missing_tags(ffprobe_json):
    asset = parse_probe(ffprobe_json("screen_recording_no_audio"), path="s.mp4", asset_id="a3")
    assert not asset.has_audio
    assert asset.created_at is None and asset.gps is None
    assert (asset.width, asset.height) == (886, 1920)


@pytest.mark.parametrize("name", ["music_with_cover_art", "still_photo"])
def test_non_footage_is_rejected(ffprobe_json, name):
    with pytest.raises(ProbeError):
        parse_probe(ffprobe_json(name), path=name, asset_id="x")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30000/1001", 29.97), ("30/1", 30.0), ("0/0", None), ("", None), (None, None), ("x", None)],
)
def test_parse_rate(value, expected):
    assert parse_rate(value) == (pytest.approx(expected, abs=0.001) if expected else None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+31.2304+121.4737+004.000/", (31.2304, 121.4737)),
        ("-33.8688+151.2093/", (-33.8688, 151.2093)),
        ("+0.0000+0.0000/", None),  # null island = no fix
        ("+91.0+10.0/", None),
        ("garbage", None),
        (None, None),
    ],
)
def test_parse_iso6709(value, expected):
    assert parse_iso6709(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-01T08:30:00.000000Z", datetime(2026, 10, 1, 8, 30, tzinfo=UTC)),
        ("2026-10-01T08:30:00", datetime(2026, 10, 1, 8, 30, tzinfo=UTC)),  # naive -> UTC
        ("1904-01-01T00:00:00.000000Z", None),
        ("not a date", None),
        (None, None),
    ],
)
def test_parse_creation_time(value, expected):
    assert parse_creation_time(value) == expected


def test_asset_id_is_stable_across_rename_and_sensitive_to_content(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x" * 3_000_000)
    first = asset_id_for(a)
    b = a.rename(tmp_path / "renamed.mp4")
    assert asset_id_for(b) == first
    b.write_bytes(b"x" * 2_999_999 + b"y")  # differs only in the tail
    assert asset_id_for(b) != first


# ---------------------------------------------------------------------------
# real ffprobe on generated clips
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
def test_probe_generated_clip(make_clip):
    clip = make_clip(
        "日常vlog 片段.mp4",  # non-ASCII path, as on real phones / Windows
        metadata={"creation_time": "2026-10-01T08:30:00Z", "location": "+31.2304+121.4737/"},
    )
    asset = probe_asset(clip)
    assert (asset.width, asset.height) == (320, 240)
    assert asset.fps == 30
    assert asset.duration_s == pytest.approx(1.0, abs=0.05)
    assert asset.has_audio
    assert asset.created_at == datetime(2026, 10, 1, 8, 30, tzinfo=UTC)
    assert asset.gps == pytest.approx((31.2304, 121.4737))
    assert asset.asset_id == asset_id_for(clip)


@pytest.mark.ffmpeg
def test_probe_rotated_clip_without_audio(make_clip):
    asset = probe_asset(make_clip("rot.mp4", rotation=90, audio=False))
    assert (asset.width, asset.height) == (240, 320)
    assert not asset.has_audio


@pytest.mark.ffmpeg
def test_probe_corrupt_file_raises(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises(ProbeError):
        probe_asset(bad)
