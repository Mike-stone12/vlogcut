"""EDL -> low-res preview mp4, rendered from proxies with FFmpeg.

Clip bodies and centred dissolves become cached H.264 + PCM .mov segments, keeping
durations sample-exact. Segments are joined with the concat demuxer (video copied,
audio encoded once) and optional music is mixed under the original sound. A soft
subtitle track shows, per clip, where it came from and why it was picked.

- canvas: orientation of the majority of the timeline (portrait 540x960 or landscape
  960x540); clips of the other orientation are letterboxed (black by default, or a
  blurred fill)
- clip boundaries are snapped to the frame grid on the cumulative timeline, so the
  preview length matches the EDL to within half a frame and audio never drifts
- 30 ms audio fades at every cut avoid clicks; clips without audio get silence

The argument builders and helpers are pure; `render_preview` runs ffmpeg.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vlogcut.ingest import ffmpeg
from vlogcut.schemas import EDL, Clip, MediaAsset, MusicCue, Shot

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
_SEGMENT_VERSION = 1  # bump when segment rendering changes, to invalidate the cache


class PreviewParams(BaseModel):
    canvas: Literal["auto", "portrait", "landscape"] = "auto"
    short_side: int = Field(default=540, gt=0)
    fps: int = Field(default=30, gt=0)
    background: Literal["black", "blur"] = "black"
    audio_fade_s: float = Field(default=0.03, ge=0)
    duck_gain_db: float = Field(default=-12.0, le=0, allow_inf_nan=False)
    music_fade_out_s: float = Field(default=2.0, ge=0)
    subtitles: bool = True
    crf: int = Field(default=26, ge=0, le=51)


class RenderError(RuntimeError):
    pass


class Segment(BaseModel):
    """Resolved source span; an incoming source makes this a transition segment."""

    clip: Clip
    asset: MediaAsset
    src_start_s: float  # absolute time in the asset
    src_dur_s: float
    out_dur_s: float  # on the timeline, snapped to the frame grid
    incoming: Segment | None = None  # second source of a dissolve segment
    dissolve_s: float = 0.0
    pad_start_s: float = 0.0  # timeline seconds of cloned edge video / silent audio
    pad_end_s: float = 0.0
    fade_in: bool = True
    fade_out: bool = True


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def canvas_size(segments: Sequence[Segment], params: PreviewParams) -> tuple[int, int]:
    """(width, height): orientation covering most of the timeline, ties -> portrait."""
    s, long = params.short_side, _even(params.short_side * 16 / 9)
    if params.canvas == "portrait":
        return s, long
    if params.canvas == "landscape":
        return long, s
    portrait = sum(g.out_dur_s for g in segments if g.asset.height >= g.asset.width)
    landscape = sum(g.out_dur_s for g in segments) - portrait
    return (s, long) if portrait >= landscape else (long, s)


def atempo_chain(speed: float) -> list[str]:
    """atempo filters for `speed`; each stage must stay within [0.5, 2]."""
    stages = []
    while speed > 2.0:
        stages.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        stages.append(0.5)
        speed /= 0.5
    stages.append(speed)
    return [f"atempo={x:.6f}" for x in stages if abs(x - 1.0) > 1e-9]


def resolve(edl: EDL, shots: Sequence[Shot], assets: Sequence[MediaAsset], fps: int):
    """Clips -> segments. Durations come from snapping the *cumulative* timeline to the
    frame grid, so rounding never accumulates: the total stays within half a frame."""
    shot_by_id = {s.shot_id: s for s in shots}
    asset_by_id = {a.asset_id: a for a in assets}
    segments, t, frame = [], 0.0, 0
    for clip in edl.clips:
        shot = shot_by_id.get(clip.shot_id)
        if shot is None:
            raise RenderError(f"EDL refers to unknown shot {clip.shot_id}")
        asset = asset_by_id.get(shot.asset_id)
        if asset is None or asset.proxy_path is None:
            raise RenderError(f"no proxy for asset {shot.asset_id} (shot {shot.shot_id})")
        if clip.out_s > shot.duration_s + 1e-3:
            raise RenderError(f"clip {clip.shot_id} out_s {clip.out_s} is past the shot end")
        t += clip.timeline_duration_s
        end_frame = max(frame + 1, round(t * fps))
        segments.append(
            Segment(
                clip=clip,
                asset=asset,
                src_start_s=shot.start_s + clip.in_s,
                src_dur_s=clip.out_s - clip.in_s,
                out_dur_s=(end_frame - frame) / fps,
            )
        )
        frame = end_frame
    return segments


def segment_plan(clips: Sequence[Segment], fps: int) -> list[Segment]:
    """Split resolved clips at cumulative, frame-snapped dissolve window edges.

    Source handles may extend beyond shots, but never require media outside assets.
    Overlapping windows cannot be represented by a two-source dissolve.
    """
    plan, time, frame = [], 0.0, 0
    for i, seg in enumerate(clips):
        left = seg.clip.transition_in.duration_s / 2
        right = clips[i + 1].clip.transition_in.duration_s / 2 if i + 1 < len(clips) else 0
        end = time + seg.clip.timeline_duration_s
        if left + right > seg.clip.timeline_duration_s:
            raise RenderError("dissolve windows overlap or extend beyond the timeline")
        body_start = round((time + left) * fps) if left else frame
        clip_end = frame + round(seg.out_dur_s * fps)
        body_end = round((end - right) * fps) if right else clip_end
        if body_end > body_start:
            plan.append(
                seg
                if not left and not right
                else _span(
                    seg,
                    seg.src_start_s + left * seg.clip.speed,
                    seg.clip.timeline_duration_s - left - right,
                    (body_end - body_start) / fps,
                ).model_copy(update={"fade_in": not left, "fade_out": not right})
            )
        if right:
            duration = 2 * right
            frames = round((end + right) * fps) - body_end
            if frames > 0:
                incoming = clips[i + 1]
                outgoing = _span(
                    seg,
                    seg.src_start_s + seg.src_dur_s - right * seg.clip.speed,
                    duration,
                    frames / fps,
                )
                plan.append(
                    outgoing.model_copy(
                        update={
                            "incoming": _span(
                                incoming,
                                incoming.src_start_s - right * incoming.clip.speed,
                                duration,
                                frames / fps,
                            ),
                            "dissolve_s": duration,
                        }
                    )
                )
        time, frame = end, clip_end
    return plan


def _span(seg: Segment, start: float, duration: float, out_duration: float) -> Segment:
    speed = seg.clip.speed
    stop = start + duration * speed
    lo, hi = max(0.0, start), min(seg.asset.duration_s, stop)
    return seg.model_copy(
        update={
            "src_start_s": lo,
            "src_dur_s": hi - lo,
            "out_dur_s": out_duration,
            "pad_start_s": max(0.0, -start) / speed,
            "pad_end_s": max(0.0, stop - seg.asset.duration_s) / speed,
            "fade_in": False,
            "fade_out": False,
        }
    )


def _filters(
    seg: Segment, canvas: tuple[int, int], params: PreviewParams, index: int, audio_index: int
) -> str:
    w, h = canvas
    d = seg.out_dur_s
    fit = f"scale={w}:{h}:force_original_aspect_ratio=decrease"
    speed = f"setpts=(PTS-STARTPTS)/{seg.clip.speed:.6f},fps={params.fps}"
    if not seg.fade_in or not seg.fade_out:
        speed += (
            f",tpad=start_mode=clone:start_duration={seg.pad_start_s:.6f}"
            f":stop_mode=clone:stop_duration={seg.pad_end_s + 1 / params.fps:.6f}"
        )
    if params.background == "blur":
        fill = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2"
        video = (
            f"[{index}:v]{speed},split[x{index}][y{index}];[x{index}]{fill}[bg{index}];"
            f"[y{index}]{fit}[fg{index}];[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        video = f"[{index}:v]{speed},{fit},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    video += f",setsar=1,format=yuv420p[v{index}]"
    fade = min(params.audio_fade_s, d / 2)
    audio = ",".join(
        (["aformat=sample_fmts=s16"] if not seg.asset.has_audio or seg.clip.audio == "mute" else [])
        + ["asetpts=PTS-STARTPTS", *atempo_chain(seg.clip.speed)]
        + [f"aresample={AUDIO_RATE}", "aformat=sample_fmts=s16:channel_layouts=stereo"]
        + ([f"adelay={round(seg.pad_start_s * AUDIO_RATE)}S:all=1"] if seg.pad_start_s else [])
        + ["apad", f"atrim=0:{d:.6f}"]
        + ([f"volume={params.duck_gain_db}dB"] if seg.clip.audio == "duck" else [])
        + ([f"afade=t=in:d={fade:.3f}"] if fade and seg.fade_in else [])
        + ([f"afade=t=out:st={d - fade:.6f}:d={fade:.3f}"] if fade and seg.fade_out else [])
    )
    return f"{video};[{audio_index}:a]{audio}[a{index}]"


def segment_args(seg: Segment, canvas: tuple[int, int], params: PreviewParams, out) -> list[str]:
    args, filters, labels, index = ["-v", "error", "-y"], [], [], 0
    d = seg.out_dur_s
    for source in [seg, seg.incoming] if seg.incoming else [seg]:
        args += ["-ss", f"{source.src_start_s:.3f}", "-t", f"{source.src_dur_s:.3f}"]
        args += ["-i", str(source.asset.proxy_path)]
        audio_index = index
        if not source.asset.has_audio or source.clip.audio == "mute":
            audio_index += 1
            args += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        filters.append(_filters(source, canvas, params, index, audio_index))
        labels.append(index)
        index = audio_index + 1
    video, audio = "[v0]", "[a0]"
    if seg.incoming:
        other = labels[1]
        filters += [
            f"[v0][v{other}]xfade=transition=fade:duration={d:.6f}:offset=0[v]",
            f"[a0][a{other}]acrossfade=d={d:.6f}:c1=tri:c2=tri[a]",
        ]
        video, audio = "[v]", "[a]"
    args += ["-filter_complex", ";".join(filters), "-map", video, "-map", audio]
    args += ["-pix_fmt", "yuv420p"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(params.crf)]
    args += ["-g", str(params.fps), "-r", str(params.fps), "-frames:v", str(round(d * params.fps))]
    args += ["-c:a", "pcm_s16le", "-f", "mov", str(out)]
    return args


def segment_key(seg: Segment, canvas: tuple[int, int], params: PreviewParams) -> str:
    parts = [
        _SEGMENT_VERSION, seg.asset.asset_id, f"{seg.src_start_s:.3f}",
        f"{seg.src_dur_s:.3f}", f"{seg.out_dur_s:.6f}", f"{seg.clip.speed:.4f}", canvas,
        params.fps, params.background, params.audio_fade_s, params.crf,
    ]  # fmt: skip
    # Keep legacy keys byte-identical, including recursive dissolve source keys.
    if seg.clip.audio != "keep":
        parts += ["audio", seg.clip.audio]
        if seg.clip.audio == "duck":
            parts += [params.duck_gain_db]
    if not seg.asset.has_audio and seg.clip.speed != 1.0:
        parts += ["signed-silence-v1"]  # atempo on unsigned silence can add a DC offset
    if not seg.fade_in or not seg.fade_out or seg.incoming:
        parts += ["dissolve-v1", seg.fade_in, seg.fade_out, seg.pad_start_s, seg.pad_end_s]
        if seg.incoming:
            parts += [seg.dissolve_s, segment_key(seg.incoming, canvas, params)]
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


def srt(segments: Sequence[Segment]) -> str:
    """Debug subtitles: source file, source time and the planner's reason per clip."""
    lines, t = [], 0.0
    for i, seg in enumerate(segments, 1):
        name = Path(seg.asset.path).name
        a, b = seg.src_start_s, seg.src_start_s + seg.src_dur_s
        text = f"{name} {_mmss(a)}–{_mmss(b)}"
        if seg.clip.speed != 1.0:
            text += f" ×{seg.clip.speed:g}"
        if seg.clip.reason:
            text += f"\n{seg.clip.reason}"
        if seg.clip.subtitle:
            text = f"{seg.clip.subtitle}\n{text}"
        lines += [str(i), f"{_srt_time(t)} --> {_srt_time(t + seg.out_dur_s)}", text, ""]
        t += seg.out_dur_s
    return "\n".join(lines)


def music_filter(cues: Sequence[MusicCue], total_s: float, fade_out_s: float) -> str:
    """filter_complex mixing music inputs 1..n under the programme audio [0:a]."""
    chains, labels = [], ["[0:a]"]
    for i, cue in enumerate(cues, 1):
        ms = round(cue.start_s * 1000)
        fade_st = max(0.0, total_s - fade_out_s)
        chain = f"[{i}:a]aresample={AUDIO_RATE},aformat=channel_layouts=stereo"
        chain += f",volume={cue.gain_db}dB,adelay={ms}|{ms},atrim=0:{total_s:.6f}"
        if fade_out_s:
            chain += f",afade=t=out:st={fade_st:.6f}:d={min(fade_out_s, total_s):.3f}"
        chains.append(f"{chain}[m{i}]")
        labels.append(f"[m{i}]")
    mix = f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[a]"
    return ";".join([*chains, mix])


def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_preview(
    edl: EDL,
    shots: Sequence[Shot],
    assets: Sequence[MediaAsset],
    workspace: str | Path,
    out_path: str | Path,
    *,
    music: Mapping[str, str | Path] | None = None,
    params: PreviewParams | None = None,
) -> Path:
    """Render `edl` to `out_path`; `music` maps MusicCue.track_id -> user-supplied file."""
    params = params or PreviewParams()
    segments = resolve(edl, shots, assets, params.fps)
    canvas = canvas_size(segments, params)
    seg_dir = Path(workspace) / "preview" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for seg in segment_plan(segments, params.fps):
        path = seg_dir / f"{segment_key(seg, canvas, params)}.mov"
        if not path.is_file():
            tmp = path.with_name(f"{path.stem}.part.mov")
            try:
                ffmpeg.run("ffmpeg", segment_args(seg, canvas, params, tmp))
            except ffmpeg.FFmpegError as exc:
                tmp.unlink(missing_ok=True)
                raise RenderError(f"segment for {seg.clip.shot_id}: {exc}") from exc
            tmp.replace(path)
        files.append(path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = seg_dir.parent / out_path.stem
    work.mkdir(exist_ok=True)
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in files), encoding="utf-8")

    total = sum(s.out_dur_s for s in segments)
    args = ["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    cues = list(edl.music)
    for cue in cues:
        track = (music or {}).get(cue.track_id)
        if track is None or not Path(track).is_file():
            raise RenderError(f"music track {cue.track_id!r} not supplied or missing")
        args += ["-i", str(track)]
    sub_index = 1 + len(cues)
    if params.subtitles:
        sub = work / "debug.srt"
        sub.write_text(srt(segments), encoding="utf-8")
        args += ["-i", str(sub)]
    if cues:
        args += ["-filter_complex", music_filter(cues, total, params.music_fade_out_s)]
        args += ["-map", "0:v", "-map", "[a]"]
    else:
        args += ["-map", "0:v", "-map", "0:a"]
    if params.subtitles:
        args += ["-map", f"{sub_index}:s", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    args += ["-t", f"{total:.6f}", str(out_path)]
    log.info("rendering preview %s (%d clips, %.1f s)", out_path, len(segments), total)
    try:
        ffmpeg.run("ffmpeg", args)
    except ffmpeg.FFmpegError as exc:
        raise RenderError(f"joining preview: {exc}") from exc
    return out_path
