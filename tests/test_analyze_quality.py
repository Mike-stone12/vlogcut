from __future__ import annotations

import pytest

from vlogcut.analyze import QualityParams, assess_shots, detect_shots
from vlogcut.analyze import quality as quality_mod
from vlogcut.analyze.shots import ShotSpan
from vlogcut.ingest import ffmpeg, make_proxy, probe_asset

np = pytest.importorskip("numpy")
from vlogcut.analyze.quality import (  # noqa: E402  (needs numpy)
    ShotMetrics,
    assess,
    audio_active,
    clipped,
    global_motion,
    jitter,
    laplacian_var,
)

FPS = 30.0
P = QualityParams()


def _texture(h: int = 270, w: int = 480, seed: int = 0) -> np.ndarray:
    """Blocky noise on 0..255: plenty of edges in every tile."""
    rng = np.random.default_rng(seed)
    return np.kron(rng.random((h // 6 + 1, w // 6 + 1)), np.ones((6, 6)))[:h, :w] * 255


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    k = np.ones(2 * r + 1) / (2 * r + 1)
    out = np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 0, img)
    return np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 1, out)


def _metrics(n: int = 150, **kw) -> ShotMetrics:
    base = dict(
        fps=FPS,
        sharpness=[200.0] * n,
        dark=[0.0] * n,
        bright=[0.0] * n,
        motion=[(0.0, 0.0)] * (n - 1),
    )
    return ShotMetrics(**{**base, **kw})


def _shake_motion(n: int, amp: float) -> list[tuple[float, float]]:
    t = np.arange(n) / FPS
    path = np.stack([amp * np.sin(t * 17), amp * np.sin(t * 13 + 1)], axis=1)
    return [tuple(d) for d in np.diff(path, axis=0)]


# ---------------------------------------------------------------------------
# per-frame metrics
# ---------------------------------------------------------------------------


def test_laplacian_var_drops_with_blur():
    img = _texture()
    sharp, soft = laplacian_var(np.stack([img, _box_blur(img, 3)]))
    assert sharp > 10 * soft


def test_clipped_fractions():
    frame = np.zeros((10, 10), np.uint8)
    frame[:3] = 255
    dark, bright = clipped(frame[None])
    assert dark[0] == pytest.approx(0.7) and bright[0] == pytest.approx(0.3)


def test_global_motion_recovers_camera_shift():
    big = _texture(330, 560, seed=1)
    a, b = big[20:290, 20:500], big[23:293, 15:495]  # camera moved: content shifts (-3, +5)
    ((dy, dx),) = global_motion(np.stack([a, b]).astype(np.float32))
    assert (dy, dx) == pytest.approx((-3, 5), abs=0.5)


def test_global_motion_ignores_a_subject_crossing_a_static_frame():
    bg = _texture(seed=2)
    frames = []
    for x in range(0, 200, 20):
        f = bg.copy()
        f[60:200, x : x + 80] = 128  # a "person" walking left to right
        frames.append(f)
    assert np.abs(global_motion(np.stack(frames).astype(np.float32))).max() < 0.5


def test_jitter_is_zero_for_a_steady_pan_and_large_for_shake():
    pan = np.tile([[0.0, 4.0]], (89, 1))
    assert jitter(pan, FPS, 1.0).max() < 1e-6
    shake = np.array(_shake_motion(90, 20.0))
    assert np.sqrt((jitter(shake, FPS, 1.0) ** 2).mean()) > 8


def test_audio_active():
    rate = 8000
    t = np.arange(rate) / rate
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    assert audio_active(tone, rate, -35) == 1.0
    assert audio_active(np.zeros(rate, np.int16), rate, -35) == 0.0


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def test_clean_shot_has_no_flags():
    q = assess(_metrics(), P)
    assert not q.rejected and q.warnings == [] and q.usable_range is None


def test_blurry_start_is_trimmed_not_rejected():
    sharp = [15.0] * 30 + [200.0] * 120  # 1 s of motion blur while raising the phone
    sharp[0] = 200.0  # a stray sharp frame must not anchor the range
    q = assess(_metrics(sharpness=sharp), P)
    assert not q.rejected
    assert q.usable_range == pytest.approx((1.0, 5.0), abs=0.1)


def test_black_frames_are_rejected():
    q = assess(_metrics(sharpness=[0.0] * 150, dark=[1.0] * 150), P)
    assert q.rejected and q.reject_reasons == ["too_dark"] and q.warnings == []


def test_night_scene_and_sky_only_warn():
    night = assess(_metrics(dark=[0.7] * 150), P)
    assert not night.rejected and night.warnings == ["dark"]
    sky = assess(_metrics(sharpness=[0.5] * 150), P)
    assert not sky.rejected and sky.warnings == ["blurry"]


def test_violent_shake_rejected_unless_someone_is_talking():
    motion = _shake_motion(150, 80.0)
    silent = assess(_metrics(motion=motion), P)
    assert silent.rejected and silent.reject_reasons == ["shaky"]
    talking = assess(_metrics(motion=motion, audio_active=0.8), P)
    assert not talking.rejected
    assert talking.warnings == ["shaky", "kept_for_audio"]


def test_mild_shake_only_warns():
    q = assess(_metrics(motion=_shake_motion(150, 8.0)), P)
    assert not q.rejected and q.warnings == ["shaky"] and q.usable_range is None


# ---------------------------------------------------------------------------
# end to end on generated footage
# ---------------------------------------------------------------------------

_STILL = "mandelbrot=size=1620x2400:rate=30,trim=duration=0.04,loop=loop=-1:size=1,trim=duration=4"


def _clip(tmp_path, name: str, vf: str, audio: bool = False):
    out = tmp_path / f"{name}.mp4"
    args = ["-v", "error", "-y", "-f", "lavfi", "-i", _STILL]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=220:duration=4", "-c:a", "aac"]
    args += ["-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    ffmpeg.run("ffmpeg", [*args, "-shortest", str(out)])
    return out


@pytest.mark.ffmpeg
@pytest.mark.scenedetect
def test_assess_shots_end_to_end_and_cache(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    shake = "crop=540:960:x='500+150*sin(t*13)+100*sin(t*31)':y='700+120*sin(t*17)'"
    cases = {
        "steady": ("crop=540:960:500:700", False),
        "black": ("crop=540:960:500:700,eq=brightness=-1", False),
        "wild_silent": (shake, False),
        "wild_talking": (shake, True),
    }
    results = {}
    for name, (vf, audio) in cases.items():
        asset = make_proxy(probe_asset(_clip(tmp_path, name, vf, audio)), ws)
        spans = detect_shots(asset, ws)
        assert len(spans) == 1
        (results[name],) = assess_shots(asset, spans, ws)

    assert not results["steady"].quality.rejected and results["steady"].quality.warnings == []
    assert results["black"].quality.reject_reasons == ["too_dark"]
    assert results["wild_silent"].quality.reject_reasons == ["shaky"]
    assert not results["wild_talking"].quality.rejected
    assert "kept_for_audio" in results["wild_talking"].quality.warnings

    monkeypatch.setattr(quality_mod, "measure_shot", lambda *a: pytest.fail("cache miss"))
    asset = make_proxy(probe_asset(tmp_path / "steady.mp4"), ws)
    assert assess_shots(asset, detect_shots(asset, ws), ws) == [results["steady"]]


def test_assess_shots_recomputes_when_spans_change(tmp_path, monkeypatch):
    from vlogcut.schemas import MediaAsset

    asset = MediaAsset(
        asset_id="a", path="a.mp4", proxy_path="p.mp4", duration_s=6, fps=30, width=540, height=960
    )
    calls = []
    monkeypatch.setattr(quality_mod, "measure_shot", lambda a, s, p: calls.append(s) or _metrics())
    one = [ShotSpan(shot_id="a_000", asset_id="a", start_s=0, end_s=5)]
    assess_shots(asset, one, tmp_path)
    assess_shots(asset, one, tmp_path)
    assert len(calls) == 1
    two = [*one, ShotSpan(shot_id="a_001", asset_id="a", start_s=5, end_s=6)]
    assert len(assess_shots(asset, two, tmp_path)) == 2
    assert len(calls) == 3
