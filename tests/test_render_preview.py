from __future__ import annotations

import json
import math
import re
from array import array
from collections.abc import Callable
from pathlib import Path

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
    segment_plan,
    srt,
)
from vlogcut.schemas import (
    EDL,
    AudioMode,
    Clip,
    MediaAsset,
    MusicCue,
    QualityFlags,
    Shot,
    Transition,
)

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


@pytest.mark.parametrize(
    "audio,speed,body_key,blend_key",
    [
        (True, 1.0, "cccf8095801ad9ba", "6f187d0204fb2adf"),
        (True, 2.0, "838241710f70fdbd", "6fd6731f94acb8ce"),
        (False, 1.0, "cccf8095801ad9ba", "6f187d0204fb2adf"),
        (False, 2.0, "3008542caaecd6b1", "b4fcd3d86ba29bc1"),
    ],
)
def test_keep_cache_keys_match_legacy(
    audio: bool, speed: float, body_key: str, blend_key: str
) -> None:
    # Golden keys captured from the renderer before Clip.audio was introduced.
    clips = [
        Clip(shot_id="p_000", in_s=0, out_s=2, speed=speed),
        Clip(shot_id="p_000", in_s=2, out_s=4,
             transition_in=Transition(kind="dissolve", duration_s=0.5)),
    ]  # fmt: skip
    segs = resolve(_edl(*clips), [_shot("p")], [_asset("p", audio=audio)], 30)
    for gain in (-12, -6):
        params = PreviewParams(duck_gain_db=gain)
        assert segment_key(segs[0], (540, 960), params) == body_key
        assert segment_key(segment_plan(segs, 30)[1], (540, 960), params) == blend_key


@pytest.mark.parametrize("incoming", [False, True])
def test_audio_modes_change_source_filters_and_cache_keys(incoming: bool) -> None:
    clips = [Clip(shot_id="p_000", in_s=0, out_s=2, speed=2) for _ in range(2)]
    clips[1].transition_in = Transition(kind="dissolve", duration_s=0.5)
    segs = resolve(_edl(*clips), [_shot("p")], [_asset("p")], 30)
    blend = segment_plan(segs, 30)[1]
    source = blend.incoming if incoming else blend
    params = PreviewParams()
    keys = []
    for mode in ("keep", "mute", "duck"):
        source.clip.audio = mode
        keys.append(segment_key(blend, (540, 960), params))
        changed_gain = segment_key(blend, (540, 960), PreviewParams(duck_gain_db=-6))
        assert (keys[-1] != changed_gain) == (mode == "duck")
        args = segment_args(blend, (540, 960), params, "blend.mov")
        graph = args[args.index("-filter_complex") + 1]
        assert graph.count("volume=-12.00dB") == (mode == "duck")
        if mode == "duck":
            assert graph.index("volume=-12.00dB") < graph.index("acrossfade=")
        if mode == "mute":
            assert "anullsrc" in " ".join(args)
            assert "aformat=sample_fmts=s16,asetpts=PTS-STARTPTS,atempo=2.000000" in graph
        assert "afade=" not in graph
    assert len(set(keys)) == 3


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
# dissolve planning and rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("duration", [0, 0.5, 0.53])
def test_segment_plan_windows_and_padding(duration):
    clips = [
        Clip(shot_id="p_000", in_s=0, out_s=2),
        Clip(
            shot_id="q_000",
            in_s=0,
            out_s=3,
            speed=1.5,
            transition_in=Transition(kind="dissolve", duration_s=duration)
            if duration
            else Transition(),
        ),
    ]
    assets = [_asset("p").model_copy(update={"duration_s": 2}), _asset("q")]
    resolved = resolve(_edl(*clips), [_shot("p", 0, 2), _shot("q", 0, 3)], assets, 30)
    plan = segment_plan(resolved, 30)
    assert sum(round(s.out_dur_s * 30) for s in plan) == 120
    if not duration:
        assert plan == resolved
        return
    before, blend, after = plan
    assert before.out_dur_s == round((2 - duration / 2) * 30) / 30
    assert before.out_dur_s + blend.out_dur_s == round((2 + duration / 2) * 30) / 30
    assert blend.out_dur_s == pytest.approx(duration, abs=1 / 30)
    assert before.src_dur_s == pytest.approx(2 - duration / 2)
    assert after.src_start_s == pytest.approx(duration / 2 * 1.5)
    assert blend.pad_end_s == pytest.approx(duration / 2)
    assert blend.incoming.pad_start_s == pytest.approx(duration / 2)
    assert not before.fade_out and not after.fade_in
    params = PreviewParams()
    key = segment_key(blend, (160, 90), params)
    changed = blend.model_copy(update={"incoming": after})
    assert key != segment_key(changed, (160, 90), params)
    args = segment_args(blend, (160, 90), params, "blend.mov")
    graph = args[args.index("-filter_complex") + 1]
    assert "tpad=start_mode=clone" in graph and "acrossfade=" in graph
    assert "atempo=1.500000" in graph and "afade=" not in graph


def test_segment_plan_adjacent_windows_and_available_handles():
    clips = [
        Clip(
            shot_id="p_000",
            in_s=0,
            out_s=d,
            transition_in=Transition(kind="dissolve", duration_s=0.53) if i else Transition(),
        )
        for i, d in enumerate((1.13, 1.17, 1.19))
    ]
    resolved = resolve(_edl(*clips), [_shot("p")], [_asset("p")], 30)
    plan = segment_plan(resolved, 30)
    assert len(plan) == 5
    assert sum(round(s.out_dur_s * 30) for s in plan) == round(3.49 * 30)
    assert not plan[2].fade_in and not plan[2].fade_out
    for transition in (plan[1], plan[3]):
        assert transition.pad_end_s == transition.incoming.pad_start_s == 0
    clips[1].transition_in.duration_s = 5
    with pytest.raises(RenderError, match="overlap"):
        segment_plan(resolve(_edl(*clips), [_shot("p")], [_asset("p")], 30), 30)


@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    ("speed", "audio", "background", "mode"),
    [
        (1.0, True, "black", "keep"),
        (2.0, False, "blur", "keep"),
        (1.0, True, "black", "mute"),
        (2.0, True, "black", "mute"),
    ],
)
def test_render_centred_dissolve(tmp_path, speed, audio, background, mode):
    assets, shots = [], []
    duration = 3 if mode == "mute" else 2  # keep audio handles for fade-level measurements
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.mp4"
        ffmpeg.run(
            "ffmpeg",
            [
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color={color}:s=160x90:r=30:d={duration}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={duration}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(path),
            ],
        )
        assets.append(
            probe_asset(path).model_copy(update={"proxy_path": str(path), "has_audio": audio})
        )
        shots.append(
            Shot(shot_id=color, asset_id=assets[-1].asset_id, start_s=0, end_s=2, quality=Q)
        )
    edl = _edl(
        Clip(shot_id="red", in_s=0, out_s=2, speed=speed),
        Clip(
            shot_id="blue",
            audio=mode,
            in_s=0,
            out_s=2,
            speed=speed,
            transition_in=Transition(kind="dissolve", duration_s=0.5),
        ),
    )
    params = PreviewParams(short_side=90, background=background)
    out = render_preview(edl, shots, assets, tmp_path, tmp_path / "out.mp4", params=params)
    assert float(_probe(out)["video"]["duration"]) == pytest.approx(
        edl.total_duration_s, abs=0.5 / 30
    )
    colors = []
    for t in (2 / speed - 0.5, 2 / speed, 2 / speed + 0.5):
        pixel = ffmpeg.run_bytes(
            "ffmpeg",
            [
                "-v",
                "error",
                "-ss",
                str(t),
                "-i",
                str(out),
                "-frames:v",
                "1",
                "-vf",
                "scale=1:1,format=rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
        )
        colors.append(tuple(pixel))
    assert colors[0][0] > 230 and colors[0][2] < 20
    assert all(70 < colors[1][channel] < 185 for channel in (0, 2))
    assert colors[2][0] < 20 and colors[2][2] > 230
    if audio:
        assert _rms_db(out, 2 / speed - 0.04, 0.08) > -35
    else:
        assert _rms_db(out, 2 / speed - 0.04, 0.08) < -60
    if mode == "mute":
        before = _rms_db(out, 2 / speed - 0.6, 0.1)
        late = _rms_db(out, 2 / speed + 0.1, 0.08)
        assert late < before - 6  # fades toward silence within the dissolve window
        assert _rms_db(out, 2 / speed + 0.4, 0.2) < -60
    cached = list((tmp_path / "preview" / "segments").glob("*.mov"))
    assert len(cached) == 3
    mtimes = [p.stat().st_mtime_ns for p in cached]
    render_preview(edl, shots, assets, tmp_path, tmp_path / "again.mp4", params=params)
    assert [p.stat().st_mtime_ns for p in cached] == mtimes


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


def _astats_db(path: Path, start: float, duration: float) -> float:
    stats = ffmpeg.run(
        "ffmpeg",
        ["-v", "error", "-ss", str(start), "-t", str(duration), "-i", str(path),
         "-vn", "-af", "astats=metadata=1:reset=0,"
         "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-", "-f", "null", "-"],
    )  # fmt: skip
    levels = re.findall(r"lavfi\.astats\.Overall\.RMS_level=([^\s]+)", stats)
    assert levels, "astats did not measure any audio samples"
    return float(levels[-1])


@pytest.mark.ffmpeg
@pytest.mark.parametrize("speed", [1.0, 2.0])
@pytest.mark.parametrize("gain", [-12.0, -6.0])
def test_render_camera_audio_modes(
    make_clip: Callable[..., Path], tmp_path: Path, speed: float, gain: float
) -> None:
    path = make_clip("tone.mp4", duration=4, size="160x90")  # lavfi 440 Hz sine with video
    asset = probe_asset(path).model_copy(update={"proxy_path": str(path)})
    shot = Shot(shot_id="tone", asset_id=asset.asset_id, start_s=0, end_s=4, quality=Q)
    params = PreviewParams(short_side=90, subtitles=False)
    assert params.duck_gain_db == -12
    if gain != -12:
        params = params.model_copy(update={"duck_gain_db": gain})
    levels = {}
    modes: tuple[AudioMode, ...] = ("keep", "mute", "duck")
    for mode in modes:
        edl = _edl(Clip(shot_id="tone", in_s=0, out_s=4, speed=speed, audio=mode))
        out = render_preview(
            edl, [shot], [asset], tmp_path, tmp_path / f"{mode}.mp4", params=params
        )
        levels[mode] = _astats_db(out, 0.25, edl.total_duration_s - 0.5)
    assert levels["keep"] > -35
    assert levels["mute"] < -80
    assert levels["duck"] - levels["keep"] == pytest.approx(gain, abs=2)


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
@pytest.mark.parametrize("shot_id,mode", [("l_000", "keep"), ("p_000", "mute")])
def test_render_with_music_and_missing_track(make_clip, tmp_path, shot_id, mode):
    ws, assets, shots = _footage(make_clip, tmp_path)
    bgm = tmp_path / "bgm.m4a"
    ffmpeg.run(
        "ffmpeg",
        ["-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=10"]
        + ["-c:a", "aac", str(bgm)],
    )
    edl = _edl(Clip(shot_id=shot_id, in_s=0, out_s=3, audio=mode),
               music=[MusicCue(track_id="bgm", gain_db=-6)])  # fmt: skip
    p = PreviewParams(subtitles=False, music_fade_out_s=0.5)
    out = render_preview(edl, shots, assets, ws, tmp_path / "m.mp4", music={"bgm": bgm}, params=p)
    assert "subtitle" not in _probe(out)
    assert _rms_db(out, 0.5, 1.0) > -35  # music under a silent clip
    with pytest.raises(RenderError, match="bgm"):
        render_preview(edl, shots, assets, ws, tmp_path / "x.mp4", music={})
