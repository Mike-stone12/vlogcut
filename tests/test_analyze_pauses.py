from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlogcut.analyze import (
    Pause,
    PauseParams,
    cut_points_in,
    detect_pauses,
    find_pauses,
    load_mono_audio,
    rms_envelope,
)
from vlogcut.ingest import ffmpeg

np = pytest.importorskip("numpy")
RATE = 16000
FRAME = 0.02


def _tone(duration_s: float) -> np.ndarray:
    return np.sin(2 * np.pi * 400 * np.arange(round(duration_s * RATE)) / RATE)


def _pauses(samples: np.ndarray, params: PauseParams | None = None) -> list[Pause]:
    return find_pauses(rms_envelope(samples, RATE, FRAME), FRAME, params or PauseParams())


def test_rms_level_frame_count_and_partial_frame() -> None:
    envelope = rms_envelope(_tone(1.01), RATE)
    assert len(envelope) == 51
    assert envelope == pytest.approx(np.full(51, -3.0103), abs=0.01)
    assert rms_envelope(np.zeros(321), RATE).tolist() == [-100.0, -100.0]
    assert rms_envelope(np.empty(0), RATE).size == 0


def test_known_gaps() -> None:
    samples = np.concatenate([_tone(1), np.zeros(9600), _tone(0.8), np.zeros(6400), _tone(1)])
    pauses = _pauses(samples)
    assert len(pauses) == 2
    for pause, bounds in zip(pauses, [(1, 1.6), (2.4, 2.8)], strict=True):
        assert (pause.start_s, pause.end_s) == pytest.approx(bounds, abs=FRAME)
        assert pause.mid_s == pytest.approx(sum(bounds) / 2, abs=FRAME)


@pytest.mark.parametrize("gap_s", [0.02, 0.12, 0.24])
def test_short_gaps_ignored(gap_s: float) -> None:
    assert _pauses(np.concatenate([_tone(1), np.zeros(round(gap_s * RATE)), _tone(1)])) == []


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
def test_unstructured_audio_has_no_pauses(level: float) -> None:
    assert _pauses(_tone(2) * level) == []
    assert find_pauses(np.empty(0), FRAME, PauseParams()) == []


def test_smoothing_removes_isolated_loud_and_quiet_frames() -> None:
    envelope = np.full(150, -3.0)
    envelope[50:80] = -100
    envelope[65] = -3  # must not split the gap into two
    envelope[20] = -100  # must not create an isolated pause
    (pause,) = find_pauses(envelope, FRAME, PauseParams(min_pause_s=0.01))
    assert (pause.start_s, pause.end_s) == pytest.approx((1, 1.6))


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


@pytest.mark.parametrize("kwargs", [{"min_pause_s": 0}, {"frame_s": 0}, {"smooth_s": -1}])
def test_invalid_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        PauseParams(**kwargs)


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
def test_detect_pauses_without_audio(make_clip: Callable[..., Path]) -> None:
    path = make_clip(audio=False)
    assert load_mono_audio(path) is None
    assert detect_pauses(path) == []
