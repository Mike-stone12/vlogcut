from __future__ import annotations

import pytest

from vlogcut.analyze.beats import (
    BeatParams,
    beat_aligned_ratio,
    detect_beats,
    energy_curve,
    estimate_tempo,
    onset_envelope,
    track_beats,
)

np = pytest.importorskip("numpy")


def test_energy_curve_ramp_and_gain_invariance() -> None:
    samples = np.arange(103, dtype=float)
    original = samples.copy()
    curve = energy_curve(samples, 22050)
    assert len(curve) == 10
    assert all(a < b for a, b in zip(curve, curve[1:], strict=False))
    assert max(curve) == 1.0
    for gain in (1e-200, -3.0, 1e200):
        assert energy_curve(samples * gain, 16000) == pytest.approx(curve)
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("samples", [np.empty(0), np.ones(9), np.zeros(100)])
def test_energy_curve_unmeasured(samples: np.ndarray) -> None:
    assert energy_curve(samples, 22050) is None


def test_energy_curve_constant_and_rms_bins() -> None:
    assert energy_curve(np.full(103, -0.3), 22050) == pytest.approx([1.0] * 10)
    # Equal spans have boundaries 0, 2, 5; include the final sample and use RMS.
    assert energy_curve(np.array([0, 2, 0, 0, 3]), 22050, bins=2) == pytest.approx(
        [(2 / 3) ** 0.5, 1.0]
    )


def _clicks(
    bpm: float, rate: int, phase: float, noise: float, kind: str = "sine"
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    samples = rng.normal(0, noise, 16 * rate)
    times = np.arange(phase, 16 - 0.04, 60 / bpm)
    t = np.arange(round(0.03 * rate)) / rate
    burst = np.sin(2 * np.pi * 1000 * t) * np.exp(-t / 0.006)
    for time in times:
        if kind == "noise":
            burst = rng.normal(0, 1, len(t)) * np.exp(-t / 0.006)
        start = round(time * rate)
        samples[start : start + len(burst)] += burst
    return samples, times


@pytest.mark.parametrize("bpm", [60, 90, 120, 150, 200])
@pytest.mark.parametrize("kind", ["sine", "noise"])
@pytest.mark.parametrize(
    "rate,phase,noise", [(22050, 0.237, 0), (16000, 0.413, 0.01), (44100, 0.071, 0.005)]
)
def test_click_tempo_and_beats(
    bpm: float, rate: int, phase: float, noise: float, kind: str
) -> None:
    samples, true_beats = _clicks(bpm, rate, phase, noise, kind)
    result = detect_beats(samples, rate)
    assert result.bpm == pytest.approx(bpm, rel=0.02)
    assert result.beats == sorted(set(result.beats))
    assert all(0 <= beat < len(samples) / rate for beat in result.beats)
    distances = np.abs(true_beats[:, None] - np.asarray(result.beats)[None, :])
    assert np.mean(distances.min(axis=1) <= 0.05) >= 0.9
    assert np.sum(distances.min(axis=0) > 0.05) <= 0.1 * len(true_beats)
    assert len(result.beats) <= 1.1 * len(true_beats)


@pytest.mark.parametrize("kind", ["empty", "short", "silence", "constant", "noise"])
def test_unstructured_audio(kind: str) -> None:
    samples = {
        "empty": np.empty(0),
        "short": np.ones(100),
        "silence": np.zeros(22050 * 12),
        "constant": np.ones(22050 * 12),
        "noise": np.random.default_rng(7).normal(0, 0.1, 22050 * 12),
    }[kind]
    result = detect_beats(samples, 22050)
    assert result.bpm is None
    assert result.beats == []


def test_envelope_grid_and_direct_helpers() -> None:
    params = BeatParams(frame_size=512, hop_size=256)
    samples, _ = _clicks(120, 22050, 0.237, 0)
    envelope, frame_rate = onset_envelope(samples, 22050, params)
    assert frame_rate == 22050 / 256
    assert len(envelope) == (len(samples) - 1) // 256 + 1
    assert np.all(envelope >= 0)
    assert estimate_tempo(envelope, frame_rate, params) == pytest.approx(120, rel=0.02)
    assert estimate_tempo(np.ones(1000), frame_rate, params) is None
    assert track_beats(np.zeros(1000), frame_rate, 120) == []


def test_alignment_inclusive_unsorted_and_unmeasured() -> None:
    assert beat_aligned_ratio([1.08], [1.0]) == 1
    assert beat_aligned_ratio([0.92], [1.0]) == 1
    assert beat_aligned_ratio([1.081], [1.0]) == 0
    assert beat_aligned_ratio([0.919], [1.0]) == 0
    assert beat_aligned_ratio([2.081, 0.92, 1.08, 3], [3, 1, 2]) == 0.75
    assert beat_aligned_ratio([1, 1, 2], [1], tolerance_s=0) == 2 / 3
    assert beat_aligned_ratio([], [1]) is None
    assert beat_aligned_ratio([1], []) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"frame_size": 0},
        {"hop_size": -1},
        {"hop_size": 2048},
        {"min_bpm": 200},
        {"min_bpm": 201},
        {"min_bpm": 0},
        {"max_bpm": float("inf")},
        {"align_tolerance_s": -0.01},
    ],
)
def test_invalid_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BeatParams(**kwargs)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate(rate: float) -> None:
    with pytest.raises(ValueError):
        detect_beats(np.zeros(100), rate)
    with pytest.raises(ValueError):
        estimate_tempo(np.zeros(100), rate, BeatParams())
    with pytest.raises(ValueError):
        track_beats(np.zeros(100), rate, 120)


@pytest.mark.parametrize(
    "samples", [np.zeros((2, 20)), np.array([float("nan")]), np.array([float("inf")])]
)
def test_invalid_samples(samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        detect_beats(samples, 22050)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_tolerance_and_tempo(value: float) -> None:
    with pytest.raises(ValueError):
        beat_aligned_ratio([], [], value)
    with pytest.raises(ValueError):
        track_beats(np.zeros(100), 43, value)


def test_invalid_alignment_times() -> None:
    with pytest.raises(ValueError):
        beat_aligned_ratio([float("nan")], [1])
