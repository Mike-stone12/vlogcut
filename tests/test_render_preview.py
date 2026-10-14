from __future__ import annotations

import json
import math
from array import array

import pytest

from vlogcut.ingest import ffmpeg, make_proxy, probe_asset
from vlogcut.render import PreviewParams, RenderError, render_preview
from vlogcut.render.preview import (
    atempo_chain,
    canvas_size,
    music_filter,
    resolve,
    segment_args,
    segment_key,
    srt,
)
from vlogcut.schemas import EDL, Clip, MediaAsset, MusicCue, QualityFlags, Shot

Q = QualityFlags(sharpness=100, exposure_clip=0, shake=0)


def _asset(aid: str, w: int = 1080, h: int = 1920, audio: bool = True) -> MediaAsset:
    return MediaAsset(
        asset_id=aid, path=f"/footage/{aid}.MOV", proxy_path=f"/ws/{aid}.mp4", duration_s=60,
        fps=30, width=w, height=h, has_audio=audio,
    )  # fmt: skip


def _shot(aid: str, start: float = 10.0, dur: float = 6.0) -> Shot:
    return Shot(shot_id=f"{aid}_000", asset_id=aid, start_s=start, end_s=start + dur, quality=Q)


def _edl(*clips: Clip, music: list[MusicCue] | None = None) -> EDL:
    return EDL(variant_id="t", target_duration_s=10, clips=list(clips), music=music or [])


# ---------------------------------------------------------------------------
# pure
# ---------------------------------------------------------------------------


def test_canvas_follows_the_majority_of_the_timeline():
    port, land = _asset("p"), _asset("l", w=1920, h=1080)
    shots = [_shot("p"), _shot("l")]
    p = PreviewParams()
    long_land = resolve(
        _edl(Clip(shot_id="p_000", in_s=0, out_s=2), Clip(shot_id="l_000", in_s=0, out_s=5)),
        shots, [port, land], 30,
    )  # fmt: skip
    assert canvas_size(long_land, p) == (960, 540)
    assert canvas_size(long_land, PreviewParams(canvas="portrait")) == (540, 960)
    tie = resolve(
        _edl(Clip(shot_id="p_000", in_s=0, out_s=2), Clip(shot_id="l_000", in_s=0, out_s=2)),
        shots, [port, land], 30,
    )  # fmt: skip
    assert canvas_size(tie, p) == (540, 960)  # ties go to portrait (phone vlogs)


@pytest.mark.parametrize(
    ("speed", "expected"),
    [(1.0, []), (2.0, ["atempo=2.000000"]), (3.0, ["atempo=2.000000", "atempo=1.500000"]),
     (0.3, ["atempo=0.500000", "atempo=0.600000"])],
)  # fmt: skip
def test_atempo_chain(speed, expected):
    assert atempo_chain(speed) == expected


def test_resolve_snaps_cumulative_timeline_to_frames():
    clips = [Clip(shot_id="p_000", in_s=0, out_s=3.18) for _ in range(13)]
    segs = resolve(_edl(*clips), [_shot("p")], [_asset("p")], 30)
    total = sum(s.out_dur_s for s in segs)
    assert abs(total - 13 * 3.18) <= 0.5 / 30
    assert all(abs(s.out_dur_s * 30 - round(s.out_dur_s * 30)) < 1e-9 for s in segs)
    assert segs[0].src_start_s == 10.0  # shot start + in_s


def test_resolve_rejects_bad_references():
    shots, assets = [_shot("p")], [_asset("p")]
    with pytest.raises(RenderError, match="unknown shot"):
        resolve(_edl(Clip(shot_id="x", in_s=0, out_s=1)), shots, assets, 30)
    with pytest.raises(RenderError, match="past the shot end"):
        resolve(_edl(Clip(shot_id="p_000", in_s=0, out_s=7)), shots, assets, 30)
    no_proxy = _asset("p").model_copy(update={"proxy_path": None})
    with pytest.raises(RenderError, match="no proxy"):
        resolve(_edl(Clip(shot_id="p_000", in_s=0, out_s=1)), shots, [no_proxy], 30)


def test_segment_args_silence_letterbox_and_frame_count():
    seg = resolve(_edl(Clip(shot_id="l_000", in_s=1, out_s=3, speed=2.0)), [_shot("l")],
                  [_asset("l", w=1920, h=1080, audio=False)], 30)[0]  # fmt: skip
    args = segment_args(seg, (540, 960), PreviewParams(), "o.mov")
    fc = args[args.index("-filter_complex") + 1]
    assert "anullsrc" in " ".join(args) and "[1:a]" in fc
    assert "pad=540:960" in fc and "boxblur" not in fc
    assert "atempo=2.000000" in fc and "setpts=(PTS-STARTPTS)/2.000000" in fc
    assert args[args.index("-frames:v") + 1] == "30"  # 2 s at 2x = 1 s
    blur = segment_args(seg, (540, 960), PreviewParams(background="blur"), "o.mov")
    assert "boxblur" in blur[blur.index("-filter_complex") + 1]


def test_segment_key_changes_with_what_is_rendered():
    seg = resolve(_edl(Clip(shot_id="p_000", in_s=0, out_s=2)), [_shot("p")], [_asset("p")], 30)[0]
    base = segment_key(seg, (540, 960), PreviewParams())
    assert base == segment_key(seg, (540, 960), PreviewParams(subtitles=False))
    assert base != segment_key(seg, (960, 540), PreviewParams())
    assert base != segment_key(seg, (540, 960), PreviewParams(background="blur"))
    longer = seg.model_copy(update={"out_dur_s": seg.out_dur_s + 1 / 30})
    assert base != segment_key(longer, (540, 960), PreviewParams())


def test_srt_lists_source_and_reason():
    clips = [Clip(shot_id="p_000", in_s=0, out_s=2, reason="rules: bin 1/2"),
             Clip(shot_id="p_000", in_s=2, out_s=4, subtitle="早安")]  # fmt: skip
    text = srt(resolve(_edl(*clips), [_shot("p")], [_asset("p")], 30))
    assert "00:00:00,000 --> 00:00:02,000" in text
    assert "p.MOV 00:10.0–00:12.0\nrules: bin 1/2" in text
    assert "00:00:02,000 --> 00:00:04,000\n早安\np.MOV 00:12.0–00:14.0" in text


def test_music_filter_mixes_under_programme_audio():
    fc = music_filter([MusicCue(track_id="bgm", start_s=1.5, gain_db=-12)], 20.0, 2.0)
    assert "volume=-12.0dB" in fc and "adelay=1500|1500" in fc
    assert "afade=t=out:st=18.000000" in fc
    assert fc.endswith("[0:a][m1]amix=inputs=2:duration=first:normalize=0[a]")


# ---------------------------------------------------------------------------
# real rendering
# ---------------------------------------------------------------------------


def _probe(path) -> dict:
    out = ffmpeg.run(
        "ffprobe", ["-v", "error", "-print_format", "json", "-show_streams", str(path)]
    )
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _rms_db(path, start: float, dur: float) -> float:
    raw = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", "-ss", str(start), "-t", str(dur), "-i", str(path)]
        + ["-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
    )
    pcm = array("h", raw)
    rms = math.sqrt(sum(x * x for x in pcm) / max(1, len(pcm))) / 32768
    return 20 * math.log10(rms + 1e-9)


def _footage(make_clip, tmp_path):
    ws = tmp_path / "ws"
    port = make_proxy(probe_asset(make_clip("p.mp4", duration=4, size="360x640")), ws)
    land = make_proxy(probe_asset(make_clip("l.mp4", duration=4, size="640x360", audio=False)), ws)
    shots = [
        Shot(shot_id="p_000", asset_id=port.asset_id, start_s=0, end_s=4, quality=Q),
        Shot(shot_id="l_000", asset_id=land.asset_id, start_s=0, end_s=4, quality=Q),
    ]
    return ws, [port, land], shots


@pytest.mark.ffmpeg
def test_render_mixed_orientation_speed_and_silence(make_clip, tmp_path):
    ws, assets, shots = _footage(make_clip, tmp_path)
    edl = _edl(
        Clip(shot_id="p_000", in_s=0.5, out_s=2.5, reason="first"),
        Clip(shot_id="l_000", in_s=0, out_s=2, reason="landscape, no audio"),
        Clip(shot_id="p_000", in_s=0, out_s=4, speed=2.0),
    )
    out = render_preview(edl, shots, assets, ws, tmp_path / "out" / "preview.mp4")
    streams = _probe(out)
    v, a = streams["video"], streams["audio"]
    assert (v["width"], v["height"]) == (540, 960)
    assert float(v["duration"]) == pytest.approx(edl.total_duration_s, abs=1 / 30)
    assert float(a["duration"]) == pytest.approx(edl.total_duration_s, abs=0.05)
    assert "subtitle" in streams
    assert _rms_db(out, 0.5, 1.0) > -30  # sine tone from the portrait clip
    assert _rms_db(out, 2.5, 1.0) < -60  # landscape clip had no audio -> silence
    top = ffmpeg.run_bytes(
        "ffmpeg",
        ["-v", "error", "-ss", "3", "-i", str(out), "-frames:v", "1"]
        + ["-vf", "crop=540:100:0:0,format=gray", "-f", "rawvideo", "-"],
    )
    assert max(top) < 30  # landscape clip is letterboxed with black bars

    segs = sorted((ws / "preview" / "segments").glob("*.mov"))
    mtimes = [p.stat().st_mtime_ns for p in segs]
    render_preview(edl, shots, assets, ws, tmp_path / "out" / "again.mp4")
    assert [p.stat().st_mtime_ns for p in segs] == mtimes  # segments reused


@pytest.mark.ffmpeg
def test_render_with_music_and_missing_track(make_clip, tmp_path):
    ws, assets, shots = _footage(make_clip, tmp_path)
    bgm = tmp_path / "bgm.m4a"
    ffmpeg.run(
        "ffmpeg",
        ["-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=10"]
        + ["-c:a", "aac", str(bgm)],
    )
    edl = _edl(Clip(shot_id="l_000", in_s=0, out_s=3), music=[MusicCue(track_id="bgm", gain_db=-6)])
    p = PreviewParams(subtitles=False, music_fade_out_s=0.5)
    out = render_preview(edl, shots, assets, ws, tmp_path / "m.mp4", music={"bgm": bgm}, params=p)
    assert "subtitle" not in _probe(out)
    assert _rms_db(out, 0.5, 1.0) > -35  # music under a silent clip
    with pytest.raises(RenderError, match="bgm"):
        render_preview(edl, shots, assets, ws, tmp_path / "x.mp4", music={})
