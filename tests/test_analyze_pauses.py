from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlogcut import pipeline
from vlogcut.analyze import (
    Pause,
    PauseParams,
    cut_points_in,
    detect_pauses,
    find_pauses,
    load_mono_audio,
    rms_envelope,
)
from vlogcut.analyze import pauses as pause_module
from vlogcut.analyze.pauses import PauseCache, analyze_pauses, cut_points_for_span
from vlogcut.ingest import ffmpeg
from vlogcut.schemas import CutPoint, MediaAsset, QualityFlags, Shot

try:
    import numpy as np
except ImportError:
    np = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy not available")
RATE = 16000
FRAME = 0.02


def _tone(duration_s: float) -> np.ndarray:
    return np.sin(2 * np.pi * 400 * np.arange(round(duration_s * RATE)) / RATE)


def _pauses(samples: np.ndarray, params: PauseParams | None = None) -> list[Pause]:
    return find_pauses(rms_envelope(samples, RATE, FRAME), FRAME, params or PauseParams())


@needs_numpy
def test_rms_level_frame_count_and_partial_frame() -> None:
    envelope = rms_envelope(_tone(1.01), RATE)
    assert len(envelope) == 51
    assert envelope == pytest.approx(np.full(51, -3.0103), abs=0.01)
    assert rms_envelope(np.zeros(321), RATE).tolist() == [-100.0, -100.0]
    assert rms_envelope(np.empty(0), RATE).size == 0


@needs_numpy
def test_known_gaps() -> None:
    samples = np.concatenate([_tone(1), np.zeros(9600), _tone(0.8), np.zeros(6400), _tone(1)])
    pauses = _pauses(samples)
    assert len(pauses) == 2
    for pause, bounds in zip(pauses, [(1, 1.6), (2.4, 2.8)], strict=True):
        assert (pause.start_s, pause.end_s) == pytest.approx(bounds, abs=FRAME)
        assert pause.mid_s == pytest.approx(sum(bounds) / 2, abs=FRAME)


@pytest.mark.parametrize("gap_s", [0.02, 0.12, 0.24])
@needs_numpy
def test_short_gaps_ignored(gap_s: float) -> None:
    assert _pauses(np.concatenate([_tone(1), np.zeros(round(gap_s * RATE)), _tone(1)])) == []


@needs_numpy
def test_relative_threshold_admits_noise_above_absolute_floor() -> None:
    noise = np.random.default_rng(7).normal(0, 0.01, round(0.6 * RATE))  # about -40 dBFS
    assert rms_envelope(noise, RATE).min() > -50
    samples = np.concatenate([_tone(1), noise, _tone(1)])
    (pause,) = _pauses(samples)
    assert (pause.start_s, pause.end_s) == pytest.approx((1, 1.6), abs=FRAME)
    # Lowering the entire recording retains the same gap through the absolute floor.
    (quieter,) = _pauses(samples * 0.1)
    assert quieter.mid_s == pytest.approx(1.3, abs=FRAME)


@pytest.mark.parametrize("level", [0.0, 0.0001, 1.0])
@needs_numpy
def test_unstructured_audio_has_no_pauses(level: float) -> None:
    assert _pauses(_tone(2) * level) == []
    assert find_pauses(np.empty(0), FRAME, PauseParams()) == []


@needs_numpy
def test_smoothing_removes_isolated_loud_and_quiet_frames() -> None:
    envelope = np.full(150, -3.0)
    envelope[50:80] = -100
    envelope[65] = -3  # must not split the gap into two
    envelope[20] = -100  # must not create an isolated pause
    (pause,) = find_pauses(envelope, FRAME, PauseParams(min_pause_s=0.01))
    assert (pause.start_s, pause.end_s) == pytest.approx((1, 1.6))


@needs_numpy
def test_only_internal_gaps_and_custom_minimum() -> None:
    samples = np.concatenate([np.zeros(RATE), _tone(1), np.zeros(9600), _tone(1), np.zeros(RATE)])
    assert len(_pauses(samples)) == 1
    assert _pauses(samples, PauseParams(min_pause_s=0.7)) == []


def test_cut_points_bounds_margin_and_order() -> None:
    pauses = [Pause(mid - 0.05, mid + 0.05) for mid in [3, 1, 4, 2, 0]]
    assert cut_points_in(pauses, 0, 4, margin_s=0) == [1, 2, 3]
    assert cut_points_in(pauses, 0, 4, margin_s=1) == [2]
    assert cut_points_in(pauses, 0.9, 3.1) == [2]  # default 0.1 s margin, strict bounds
    assert cut_points_in(pauses, 0, 1, margin_s=1) == []
    assert cut_points_in([], 0, 4) == []


def test_cut_points_for_span_default_margin_sorts_and_deduplicates() -> None:
    midpoints = [14, 13.9, 13.95, 12, 9, 10.05, 10.1, 11, 11, 15, 10]
    original = midpoints.copy()
    assert cut_points_for_span(midpoints, 10, 14) == [
        CutPoint(time_s=11, kind="pause"),
        CutPoint(time_s=12, kind="pause"),
    ]
    assert midpoints == original


@pytest.mark.parametrize("margin, expected", [(0, [11, 12, 13]), (1, [12]), (2, []), (3, [])])
def test_cut_points_for_span_margin_edges(margin: float, expected: list[float]) -> None:
    assert cut_points_for_span([10, 11, 12, 13, 14], 10, 14, margin_s=margin) == [
        CutPoint(time_s=t, kind="pause") for t in expected
    ]


def test_cut_points_for_span_empty_input() -> None:
    assert cut_points_for_span([], 10, 14) == []


@pytest.mark.parametrize("midpoints", [[], [12]])
def test_cut_points_for_span_negative_margin(midpoints: list[float]) -> None:
    with pytest.raises(ValueError, match="margin"):
        cut_points_for_span(midpoints, 10, 14, margin_s=-0.1)


def test_pause_records_map_to_cut_point_candidates() -> None:
    pauses = [Pause(12, 13), Pause(10, 11)]
    assert cut_points_for_span(cut_points_in(pauses, 10, 14), 10, 14) == [
        CutPoint(time_s=10.5, kind="pause"),
        CutPoint(time_s=12.5, kind="pause"),
    ]


@pytest.mark.parametrize("kwargs", [{"min_pause_s": 0}, {"frame_s": 0}, {"smooth_s": -1}])
def test_invalid_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        PauseParams(**kwargs)


@needs_numpy
def test_decoder_normalizes_shared_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(name: str, args: list[str]) -> str:
        assert name == "ffprobe" and args[-1] == "proxy.mp4"
        return "1\n"

    def decode(name: str, args: list[str]) -> bytes:
        assert name == "ffmpeg"
        assert args == [
            "-v",
            "error",
            "-i",
            "proxy.mp4",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "-",
        ]
        return np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes()

    monkeypatch.setattr(ffmpeg, "run", probe)
    monkeypatch.setattr(ffmpeg, "run_bytes", decode)
    samples = load_mono_audio(Path("proxy.mp4"), sample_rate=8000)
    assert samples is not None and samples.dtype == np.float32
    assert samples.tolist() == [-1, 0, 0.5, 32767 / 32768]


@needs_numpy
def test_no_audio_skips_decode_and_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg, "run", lambda *args: "")
    monkeypatch.setattr(ffmpeg, "run_bytes", lambda *args: pytest.fail("unexpected decode"))
    assert load_mono_audio("silent.mp4") is None
    assert detect_pauses("silent.mp4") == []

    def fail(*args: object) -> str:
        raise ffmpeg.FFmpegError(["ffprobe"], 1, "broken file")

    monkeypatch.setattr(ffmpeg, "run", fail)
    with pytest.raises(ffmpeg.FFmpegError):
        detect_pauses("broken.mp4")


@pytest.mark.ffmpeg
@needs_numpy
def test_detect_pauses_lavfi(tmp_path: Path) -> None:
    path = tmp_path / "gap.mp4"
    ffmpeg.run(
        "ffmpeg",
        [
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=25:duration=2.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=400:sample_rate=16000:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono,atrim=duration=0.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=400:sample_rate=16000:duration=1",
            "-filter_complex",
            "[1:a][2:a][3:a]concat=n=3:v=0:a=1[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(path),
        ],
    )
    (pause,) = detect_pauses(path)
    assert pause.mid_s == pytest.approx(1.3, abs=0.05)
    assert pause.end_s - pause.start_s == pytest.approx(0.6, abs=0.05)


@pytest.mark.ffmpeg
@needs_numpy
def test_detect_pauses_without_audio(make_clip: Callable[..., Path]) -> None:
    path = make_clip(audio=False)
    assert load_mono_audio(path) is None
    assert detect_pauses(path) == []


@pytest.fixture
def pause_asset() -> MediaAsset:
    return MediaAsset(
        asset_id="asset", path="source.mp4", proxy_path="proxy.mp4",
        duration_s=8, fps=25, width=64, height=64,
    )  # fmt: skip


def test_pause_cache_hit_and_changed_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_asset: MediaAsset
) -> None:
    calls: list[PauseParams] = []

    def detect(path: str, params: PauseParams) -> list[Pause]:
        assert path == pause_asset.proxy_path
        calls.append(params)
        return [Pause(1, 1.6)] if params.min_pause_s < 0.6 else []

    replace = Path.replace
    writes: list[Path] = []

    def replace_part(path: Path, target: Path) -> Path:
        assert path == target.with_suffix(".part")
        PauseCache.model_validate_json(path.read_text(encoding="utf-8"))
        writes.append(target)
        return replace(path, target)

    monkeypatch.setattr(pause_module, "detect_pauses", detect)
    monkeypatch.setattr(Path, "replace", replace_part)
    assert analyze_pauses(pause_asset, tmp_path) == [Pause(1, 1.6)]
    assert analyze_pauses(pause_asset, tmp_path) == [Pause(1, 1.6)]
    assert calls == [PauseParams()]
    changed = PauseParams(min_pause_s=0.7)
    assert analyze_pauses(pause_asset, tmp_path, changed) == []
    assert analyze_pauses(pause_asset, tmp_path, changed) == []  # empty results also hit
    assert calls == [PauseParams(), changed]
    cache_path = tmp_path / "pauses" / "asset.json"
    assert writes == [cache_path, cache_path]
    cache = PauseCache.model_validate_json(cache_path.read_text(encoding="utf-8"))
    assert cache.asset_id == pause_asset.asset_id and cache.params == changed
    assert not cache_path.with_suffix(".part").exists()


@pytest.mark.parametrize(
    "damage", ["json", "encoding", "missing", "asset", "params", "interval", "nonfinite"]
)
def test_invalid_pause_cache_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_asset: MediaAsset, damage: str
) -> None:
    record = PauseCache(asset_id="asset", params=PauseParams(), pauses=[Pause(1, 1.6)])
    data = record.model_dump(mode="json")
    if damage == "missing":
        del data["pauses"]
    elif damage == "asset":
        data["asset_id"] = "other"
    elif damage == "params":
        data["params"]["min_pause_s"] = -1
    elif damage == "interval":
        data["pauses"][0]["end_s"] = 0.5
    elif damage == "nonfinite":
        data["pauses"][0]["end_s"] = float("inf")
    content = {"json": b"{", "encoding": b"\xff"}.get(damage, json.dumps(data).encode())
    cache_path = tmp_path / "pauses" / "asset.json"
    cache_path.parent.mkdir()
    cache_path.write_bytes(content)
    calls = []

    def detect(path: str, params: PauseParams) -> list[Pause]:
        calls.append(path)
        return [Pause(2, 2.6)]

    monkeypatch.setattr(pause_module, "detect_pauses", detect)
    assert analyze_pauses(pause_asset, tmp_path) == [Pause(2, 2.6)]
    assert calls == [pause_asset.proxy_path]
    assert analyze_pauses(pause_asset, tmp_path) == [Pause(2, 2.6)]
    assert len(calls) == 1


@pytest.mark.parametrize("has_audio", [True, False])
def test_pipeline_refreshes_legacy_shots_and_writes_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_asset: MediaAsset, has_audio: bool
) -> None:
    from vlogcut import analyze as analyze_module
    from vlogcut.analyze import quality
    from vlogcut.analyze.shots import ShotSpan

    pause_asset.has_audio = has_audio
    spans = [
        ShotSpan(shot_id=f"asset_{i}", asset_id="asset", start_s=start, end_s=end)
        for i, (start, end) in enumerate([(0, 4), (4, 8)])
    ]
    shots = [
        Shot(**span.model_dump(), quality=QualityFlags(sharpness=100, exposure_clip=0, shake=0))
        for span in spans
    ]
    workspace = tmp_path / "ws"
    quality_path = workspace / "shots" / "asset.quality.json"
    quality_path.parent.mkdir(parents=True)
    cache = quality.QualityCache(
        asset_id="asset", params=quality.QualityParams(),
        spans=[(s.shot_id, s.start_s, s.end_s) for s in spans], shots=shots,
    ).model_dump(mode="json")  # fmt: skip
    for shot in cache["shots"]:
        del shot["cut_points"]  # simulate a quality cache from before cut points existed
    quality_path.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(analyze_module, "detect_shots", lambda *args: spans)
    monkeypatch.setattr(quality, "measure_shot", lambda *args: pytest.fail("re-measured"))
    calls = []

    def detect(path: str, params: PauseParams) -> list[Pause]:
        assert has_audio, "no-audio assets must not decode"
        calls.append(path)
        return [Pause(mid - 0.04, mid + 0.04) for mid in [2, 0.05, 3.95, 1, 1, 4.05]]

    monkeypatch.setattr(pause_module, "detect_pauses", detect)
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([pause_asset], []))
    (tmp_path / "source.mp4").touch()
    progress: list[str] = []
    expected = [CutPoint(time_s=t, kind="pause") for t in [1, 2]] if has_audio else []
    for _ in range(2):
        result = pipeline.cut(
            tmp_path, workspace, target_s=6, render=False, progress=progress.append
        )
        assert [s.cut_points for s in result.shots] == [expected, []]
        saved = json.loads((workspace / "shots.json").read_text(encoding="utf-8"))
        assert [Shot.model_validate(s) for s in saved] == result.shots
    assert calls == ([pause_asset.proxy_path] if has_audio else [])
    assert "[1/1] pauses source.mp4" in progress


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
@needs_numpy
def test_pipeline_finds_generated_audio_pause(
    tmp_path: Path, make_clip: Callable[..., Path]
) -> None:
    from vlogcut.ingest import probe_asset

    source = make_clip(duration=2.6, size="64x64", fps=25)
    proxy = tmp_path / "pause.mp4"
    ffmpeg.run(
        "ffmpeg",
        ["-v", "error", "-y", "-i", str(source), "-c:v", "copy",
         "-af", "volume=0:enable='between(t,1,1.6)'", "-c:a", "aac", str(proxy)],
    )  # fmt: skip
    asset = probe_asset(proxy).model_copy(update={"proxy_path": str(proxy)})
    shots = pipeline.analyze([asset], tmp_path / "ws")
    points = [point for shot in shots for point in shot.cut_points]
    assert len(points) == 1
    assert points[0].kind == "pause"
    assert points[0].time_s == pytest.approx(1.3, abs=0.1)
