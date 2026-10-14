"""EDL -> low-res preview mp4, rendered from proxies with FFmpeg.

Each clip becomes a cached segment (H.264 + PCM in .mov, so durations stay
sample-exact), then the segments are joined with the concat demuxer (video copied,
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
    music_fade_out_s: float = Field(default=2.0, ge=0)
    subtitles: bool = True
    crf: int = Field(default=26, ge=0, le=51)


class RenderError(RuntimeError):
    pass


class Segment(BaseModel):
    """One clip resolved against its shot and asset, ready to render."""

    clip: Clip
    asset: MediaAsset
    src_start_s: float  # absolute time in the asset
    src_dur_s: float
    out_dur_s: float  # on the timeline, snapped to the frame grid


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


def segment_args(seg: Segment, canvas: tuple[int, int], params: PreviewParams, out) -> list[str]:
    w, h = canvas
    d = seg.out_dur_s
    args = ["-v", "error", "-y", "-ss", f"{seg.src_start_s:.3f}", "-t", f"{seg.src_dur_s:.3f}"]
    args += ["-i", str(seg.asset.proxy_path)]
    if not seg.asset.has_audio:
        args += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
    fit = f"scale={w}:{h}:force_original_aspect_ratio=decrease"
    speed = f"setpts=(PTS-STARTPTS)/{seg.clip.speed:.6f},fps={params.fps}"
    if params.background == "blur":
        fill = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2"
        video = f"[0:v]{speed},split[a][b];[a]{fill}[bg];[b]{fit}[fg];"
        video += "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]"
    else:
        video = f"[0:v]{speed},{fit},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        video += ",format=yuv420p[v]"
    src_a = "[0:a]" if seg.asset.has_audio else "[1:a]"
    fade = min(params.audio_fade_s, d / 2)
    audio = ",".join(
        ["asetpts=PTS-STARTPTS", *atempo_chain(seg.clip.speed)]
        + [f"aresample={AUDIO_RATE}", "aformat=sample_fmts=s16:channel_layouts=stereo"]
        + ["apad", f"atrim=0:{d:.6f}"]
        + (
            [f"afade=t=in:d={fade:.3f}", f"afade=t=out:st={d - fade:.6f}:d={fade:.3f}"]
            if fade
            else []
        )
    )
    args += ["-filter_complex", f"{video};{src_a}{audio}[a]", "-map", "[v]", "-map", "[a]"]
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
    for seg in segments:
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
