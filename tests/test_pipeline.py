import json
import logging
from pathlib import Path
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut import pipeline, plan
from vlogcut.analyze import pauses, quality
from vlogcut.analyze import shots as shot_module
from vlogcut.ingest import ProbeError
from vlogcut.llm.router import ModelRouter
from vlogcut.schemas import (
    EDL,
    Clip,
    DurationStats,
    MediaAsset,
    MusicCue,
    QualityFlags,
    Shot,
    StyleProfile,
    Transition,
    TransitionMix,
)
from vlogcut.validate import validate_edl


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.outputs.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _router(tmp_path: Path, client: FakeClient) -> ModelRouter:
    return ModelRouter(
        {
            "providers": {"fake": {"base_url": "http://fake/v1"}},
            "roles": {
                "planner": {"provider": "fake", "model": "test-planner"},
                "captioner": {"provider": "fake", "model": "test-captioner"},
            },
        },
        log_path=tmp_path / "calls.jsonl",
        client_factory=lambda _: client,
    )


@pytest.fixture
def stub_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=10, fps=30, width=320, height=240)
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: _shots())


def _shots() -> list[Shot]:
    return [
        Shot(
            shot_id="s", asset_id="a", start_s=0, end_s=10,
            quality=QualityFlags(sharpness=200, exposure_clip=0, shake=0),
        )
    ]  # fmt: skip


def _edl(shot_id: str = "s") -> EDL:
    return EDL(
        variant_id="test", target_duration_s=1,
        clips=[Clip(shot_id=shot_id, in_s=0, out_s=1)],
    )  # fmt: skip


@pytest.mark.parametrize("planner", ["rules", "llm"])
@pytest.mark.parametrize("repair", [False, True])
def test_music_snaps_before_validation_and_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_analysis: None,
    planner: str,
    repair: bool,
) -> None:
    from vlogcut.analyze.beats import BeatParams, BeatResult

    shots = _shots()
    shots.append(shots[0].model_copy(update={"shot_id": "s2"}))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    clips = [Clip(shot_id="s", in_s=1, out_s=1.76), Clip(shot_id="s2", in_s=1, out_s=2)]
    if repair:
        clips.append(Clip(shot_id="missing", in_s=0, out_s=1))
    edl = EDL(variant_id="test", target_duration_s=1.76 + repair, clips=clips)
    target_s = edl.total_duration_s
    profile = StyleProfile(shot_duration_s=DurationStats(p10=1, median=1, p90=1), cuts_per_min=60)
    style = tmp_path / "style.json"
    style.write_text(profile.model_dump_json(), encoding="utf-8")
    assert "clip_too_short" not in {
        v.code for v in validate_edl(edl, shots, profile, target_s=target_s)
    }
    monkeypatch.setattr(plan, "rough_cut", lambda *args, **kwargs: edl)
    router = _router(tmp_path, FakeClient([edl.model_dump_json()])) if planner == "llm" else None
    music = tmp_path / "music.wav"
    music.touch()
    samples = object()
    calls: list[tuple[object, int]] = []

    def load(path: Path, rate: int) -> object:
        assert path == music
        assert rate == BeatParams().sample_rate
        return samples

    def detect(audio: object, rate: int) -> BeatResult:
        calls.append((audio, rate))
        return BeatResult(bpm=120, beats=[0.71])

    monkeypatch.setattr(pauses, "load_mono_audio", load)
    monkeypatch.setattr("vlogcut.analyze.beats.detect_beats", detect)
    repair_if_needed = pipeline.repair_if_needed

    def check_repair_input(planned: EDL, *args: Any) -> tuple[EDL, bool]:
        assert planned.clips[0].timeline_duration_s == pytest.approx(0.71)
        assert "clip_too_short" in {v.code for v in args[1]}
        return repair_if_needed(planned, *args)

    monkeypatch.setattr(pipeline, "repair_if_needed", check_repair_input)
    rendered: list[EDL] = []
    monkeypatch.setattr("vlogcut.render.render_preview", lambda edl, *a, **kw: rendered.append(edl))
    result = pipeline.cut(
        tmp_path, tmp_path, target_s=target_s, music=music, style=style,
        planner=planner, router=router, max_repair_rounds=0,
    )  # fmt: skip
    assert calls == [(samples, BeatParams().sample_rate)]
    assert result.edl.clips[0].timeline_duration_s == pytest.approx(0.71)
    assert result.edl.clips[1].in_s == pytest.approx(0.95)
    assert result.edl.music == [MusicCue(track_id="music", gain_db=-14)]
    assert result.repaired == repair
    assert rendered == [result.edl]
    assert result.violations == validate_edl(result.edl, shots, profile, target_s=target_s)
    assert "clip_too_short" in {v.code for v in result.violations}
    assert EDL.model_validate_json(result.edl_path.read_text(encoding="utf-8")) == result.edl
    assert json.loads(result.edl_path.with_suffix(".violations.json").read_text()) == [
        v.model_dump(mode="json") for v in result.violations
    ]


@pytest.mark.parametrize("explicit_none", [False, True])
def test_no_music_preserves_edl_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_analysis: None, explicit_none: bool
) -> None:
    assets, _ = pipeline.ingest([], tmp_path)
    expected = plan.rough_cut(_shots(), assets, plan.RoughCutParams(target_s=3), None)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(expected.model_dump_json(indent=1), encoding="utf-8")
    for target in (
        "vlogcut.analyze.pauses.load_mono_audio",
        "vlogcut.analyze.beats.detect_beats",
        "vlogcut.plan.snap.snap_to_beats",
    ):
        monkeypatch.setattr(target, lambda *args: pytest.fail("unexpected music analysis/snapping"))
    result = pipeline.cut(
        tmp_path, tmp_path, target_s=3, render=False, **({"music": None} if explicit_none else {})
    )
    assert result.edl == expected
    assert result.edl_path.read_bytes() == baseline.read_bytes()


@pytest.mark.parametrize("no_audio", [False, True])
def test_music_without_beats_keeps_edl_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_analysis: None,
    caplog: pytest.LogCaptureFixture,
    no_audio: bool,
) -> None:
    from vlogcut.analyze.beats import BeatResult

    edl = _edl()
    expected = edl.model_copy(deep=True)
    expected.music = [MusicCue(track_id="music", gain_db=-14)]
    monkeypatch.setattr(plan, "rough_cut", lambda *args, **kwargs: edl)
    monkeypatch.setattr(pauses, "load_mono_audio", lambda *args: None if no_audio else object())
    monkeypatch.setattr("vlogcut.analyze.beats.detect_beats", lambda *args: BeatResult(None, []))
    monkeypatch.setattr(
        "vlogcut.plan.snap.snap_to_beats", lambda *args: pytest.fail("unexpected snapping")
    )
    music = tmp_path / "music.wav"
    music.touch()
    result = pipeline.cut(tmp_path, tmp_path, target_s=1, music=music, render=False)
    assert result.edl == expected
    assert result.violations == []
    assert any(
        r.levelname == "WARNING" and "No beats detected" in r.message for r in caplog.records
    )


def test_missing_music_fails_before_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "find_footage", lambda *args: pytest.fail("scanned footage"))
    monkeypatch.setattr(plan, "rough_cut", lambda *args: pytest.fail("planned footage"))
    with pytest.raises(FileNotFoundError, match="music file.*missing.wav"):
        pipeline.cut(tmp_path, tmp_path, music=tmp_path / "missing.wav")


@pytest.mark.parametrize("stage", ["decode", "detect"])
def test_music_dependency_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_analysis: None, stage: str
) -> None:
    error = ImportError("No module named 'numpy'")

    def fail(*args: object) -> None:
        raise error

    music = tmp_path / "music.wav"
    music.touch()
    monkeypatch.setattr(
        pauses, "load_mono_audio", fail if stage == "decode" else lambda *a: object()
    )
    monkeypatch.setattr("vlogcut.analyze.beats.detect_beats", fail)
    with pytest.raises(ImportError, match="music beat detection needs:.*analyze") as raised:
        pipeline.cut(tmp_path, tmp_path, target_s=3, music=music, render=False)
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("planner", ["rules", "llm"])
@pytest.mark.parametrize("cold_open", [None, False, True])
def test_cut_forwards_cold_open_to_rules_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_analysis: None,
    caplog: pytest.LogCaptureFixture,
    planner: str,
    cold_open: bool | None,
) -> None:
    shots = [
        _shots()[0].model_copy(update={"shot_id": f"s{i}", "start_s": i * 3, "end_s": (i + 1) * 3})
        for i in range(3)
    ]
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    style = tmp_path / "style.json"
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=1, p90=1),
        cuts_per_min=60,
        cold_open_clips=1,
    )
    style.write_text(profile.model_dump_json(), encoding="utf-8")
    rough_cut = plan.rough_cut

    def legacy_cut(*args: Any, **kwargs: Any) -> EDL:
        return rough_cut(*args)  # previous pipeline call, without the new keyword

    monkeypatch.setattr(plan, "rough_cut", legacy_cut)
    baseline = pipeline.cut(tmp_path, tmp_path, target_s=3, style=style, render=False)
    paths = [
        baseline.edl_path, baseline.edl_path.with_suffix(".violations.json"),
        tmp_path / "shots.json",
    ]  # fmt: skip
    before = [path.read_bytes() for path in paths]
    forwarded: list[bool] = []

    def spy(*args: Any, **kwargs: Any) -> EDL:
        forwarded.append(kwargs["cold_open"])
        return rough_cut(*args, **kwargs)

    monkeypatch.setattr(plan, "rough_cut", spy)
    client = FakeClient(["not JSON"] * 3)
    options = {} if cold_open is None else {"cold_open": cold_open}
    result = pipeline.cut(
        tmp_path, tmp_path, target_s=3, style=style, render=False,
        planner=planner, router=_router(tmp_path, client), **options,
    )  # fmt: skip
    assert forwarded == [cold_open is True]
    assert result.planner == "rules" and result.fallback == (planner == "llm")
    assert result.violations == [] and not result.repaired
    assert result.edl_path == baseline.edl_path
    if cold_open:
        assert result.edl.clips[0].shot_id != baseline.edl.clips[0].shot_id
    else:
        assert [path.read_bytes() for path in paths] == before
    warnings = [r for r in caplog.records if "ignores cold_open" in r.message]
    assert len(warnings) == int(cold_open is True and planner == "llm")


@pytest.mark.parametrize("allowance", [None, 0])
def test_cut_warns_for_zero_cold_open_allowance(
    tmp_path: Path,
    stub_analysis: None,
    caplog: pytest.LogCaptureFixture,
    allowance: int | None,
) -> None:
    style = None
    if allowance is not None:
        style = tmp_path / "style.json"
        profile = StyleProfile(
            shot_duration_s=DurationStats(p10=1, median=3, p90=5),
            cuts_per_min=20,
            cold_open_clips=allowance,
        )
        style.write_text(profile.model_dump_json(), encoding="utf-8")
    baseline = pipeline.cut(tmp_path, tmp_path, target_s=3, style=style, render=False)
    assert caplog.records == []
    result = pipeline.cut(tmp_path, tmp_path, target_s=3, style=style, render=False, cold_open=True)
    assert result.edl == baseline.edl and result.violations == []
    assert [(r.levelname, r.message) for r in caplog.records] == [
        ("WARNING", "Cold-open allowance is 0; no cold open is produced")
    ]


def test_cut_llm_ignores_cold_open_without_changing_prompt(
    tmp_path: Path, stub_analysis: None, caplog: pytest.LogCaptureFixture
) -> None:
    style = tmp_path / "style.json"
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=1, p90=1),
        cuts_per_min=60,
        cold_open_clips=1,
    )
    style.write_text(profile.model_dump_json(), encoding="utf-8")
    clients = []
    results = []
    for cold_open in (False, True):
        client = FakeClient(["not JSON", _edl().model_dump_json()])
        clients.append(client)
        results.append(
            pipeline.cut(
                tmp_path,
                tmp_path,
                target_s=1,
                style=style,
                render=False,
                planner="llm",
                router=_router(tmp_path, client),
                cold_open=cold_open,
            )  # fmt: skip
        )
    assert results[0] == results[1]
    assert results[1].planner == "llm" and results[1].repair_rounds == 1
    assert clients[0].requests == clients[1].requests
    warnings = [r for r in caplog.records if "ignores cold_open" in r.message]
    assert len(warnings) == 1 and warnings[0].levelname == "WARNING"


def _assets() -> list[MediaAsset]:
    return [
        MediaAsset(
            asset_id=str(i), path=f"{i}.mp4", proxy_path=f"{i}.mp4",
            duration_s=10, fps=30, width=320, height=240,
        )
        for i in range(3)
    ]  # fmt: skip


@pytest.fixture
def caption_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[MediaAsset]:
    assets = _assets()[:1]
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"tiny fake JPEG")

    def detect(*args: object) -> list[Shot]:
        return [
            _shots()[0].model_copy(
                update={
                    "shot_id": f"s{i}", "asset_id": assets[0].asset_id,
                    "start_s": i * 3, "end_s": (i + 1) * 3,
                    "keyframes": [str(frame)] if i < 2 else [],
                },
                deep=True,
            )
            for i in range(3)
        ]  # fmt: skip

    def assess(asset: MediaAsset, shots: list[Shot], workspace: Path) -> list[Shot]:
        shots[1].quality.rejected = True
        return shots

    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(assets[0].path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **k: (assets, []))
    monkeypatch.setattr("vlogcut.analyze.detect_shots", detect)
    monkeypatch.setattr("vlogcut.analyze.assess_shots", assess)
    monkeypatch.setattr(pauses, "analyze_pauses", lambda *a: [pauses.Pause(1, 2)])
    return assets


@pytest.mark.parametrize("workers", [1, 3])
def test_cut_captions_and_cache_preserve_planning_and_default_outputs(
    tmp_path: Path, caption_analysis: list[MediaAsset], workers: int
) -> None:
    client = FakeClient(['{"description":"Coffee","tags":["cafe"],"interest":4}'])
    router = _router(tmp_path, client)
    baseline = pipeline.cut(tmp_path, tmp_path, target_s=3, render=False, workers=workers)
    paths = [
        tmp_path / "shots.json", baseline.edl_path,
        baseline.edl_path.with_suffix(".violations.json"),
    ]  # fmt: skip
    before = [path.read_bytes() for path in paths]
    assert not (tmp_path / "captions").exists()
    first_bytes = None
    for _ in range(2):
        result = pipeline.cut(
            tmp_path, tmp_path, target_s=3, render=False, workers=workers,
            caption=True, router=router,
        )  # fmt: skip
        assert len(client.requests) == 1
        assert client.requests[0]["model"] == "test-captioner"
        assert [s.caption for s in result.shots] == ["Coffee", None, None]
        assert result.shots[0].tags == ["cafe"] and result.shots[0].interest == 4
        assert result.shots[0].cut_points == baseline.shots[0].cut_points
        assert result.shots[0].cut_points[0].time_s == 1.5
        assert json.loads(paths[0].read_text(encoding="utf-8")) == [
            {**s.model_dump(mode="json"), "keyframes": ["frame.jpg"] if s.keyframes else []}
            for s in result.shots
        ]
        assert [path.read_bytes() for path in paths[1:]] == before[1:]
        if first_bytes is not None:
            assert paths[0].read_bytes() == first_bytes
        first_bytes = paths[0].read_bytes()
    cache = json.loads((tmp_path / "captions" / "s0.json").read_text(encoding="utf-8"))
    assert cache["key"]["model"] == "test-captioner"
    assert len(list((tmp_path / "captions").glob("*.json"))) == 1
    calls = [json.loads(line) for line in router.log_path.read_text().splitlines()]
    assert [call["role"] for call in calls] == ["captioner"]
    for options in ({}, {"caption": False}):
        pipeline.cut(
            tmp_path, tmp_path, target_s=3, render=False, workers=workers,
            router=router, **options,
        )  # fmt: skip
        assert len(client.requests) == 1
        assert [path.read_bytes() for path in paths] == before


def test_analyze_returns_captions(tmp_path: Path, caption_analysis: list[MediaAsset]) -> None:
    client = FakeClient(['{"description":"Coffee","tags":["cafe"],"interest":4}'])
    shots = pipeline.analyze(
        caption_analysis, tmp_path, caption=True, router=_router(tmp_path, client)
    )
    assert [s.caption for s in shots] == ["Coffee", None, None]
    assert shots[0].tags == ["cafe"] and shots[0].interest == 4


def test_caption_requires_router_before_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("started analysis")

    monkeypatch.setattr(pipeline, "find_footage", unexpected)
    monkeypatch.setattr("vlogcut.analyze.detect_shots", unexpected)
    for call in (
        lambda: pipeline.cut(tmp_path, tmp_path, caption=True),
        lambda: pipeline.analyze(_assets(), tmp_path, caption=True),
    ):
        with pytest.raises(ValueError, match="caption=True requires a ModelRouter"):
            call()
    assert not list(tmp_path.iterdir())


def test_parallel_analysis_preserves_outputs_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets()
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(a.path) for a in assets])
    monkeypatch.setattr(pipeline, "probe_asset", lambda path: assets[int(path.stem)])
    monkeypatch.setattr(pipeline, "make_proxy", lambda asset, workspace: asset)
    monkeypatch.setattr(shot_module, "detect_cuts", lambda *a, **k: [5.0])
    monkeypatch.setattr(shot_module, "extract_keyframes", lambda *a: [])
    monkeypatch.setattr(
        "vlogcut.analyze.detect_shots",
        lambda asset, workspace: shot_module.detect_shots(
            asset, workspace, shot_module.ShotParams(dissolves=False)
        ),
    )
    monkeypatch.setattr(quality, "measure_shot", lambda *a: None)
    monkeypatch.setattr(quality, "assess", lambda *a: _shots()[0].quality)
    monkeypatch.setattr(pauses, "detect_pauses", lambda *a: [pauses.Pause(1, 2)])
    analyze_pauses = pauses.analyze_pauses
    completed: list[str] = []
    finished = [Event() for _ in assets]
    parallel = False

    def ordered_pauses(asset: MediaAsset, workspace: Path) -> list[pauses.Pause]:
        i = int(asset.asset_id)
        if parallel and i < 2:
            assert finished[i + 1].wait(5), "assets did not run concurrently"
        result = analyze_pauses(asset, workspace)
        completed.append(asset.asset_id)
        finished[i].set()
        return result

    monkeypatch.setattr(pauses, "analyze_pauses", ordered_pauses)
    with monkeypatch.context() as sequential:
        sequential.setattr(
            pipeline, "ThreadPoolExecutor", lambda **kwargs: pytest.fail("started pool")
        )
        baseline = pipeline.cut(tmp_path, tmp_path, target_s=9, render=False, workers=1)
    assert completed == ["0", "1", "2"]
    expected = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert len(expected) == 12  # shot, quality, pause caches plus shots, EDL and violations
    for path in expected:
        (tmp_path / path).unlink()
    completed.clear()
    for event in finished:
        event.clear()
    parallel = True
    result = pipeline.cut(tmp_path, tmp_path, target_s=9, render=False, workers=3)
    assert completed == ["2", "1", "0"]
    assert result.shots == baseline.shots
    assert result.edl == baseline.edl and result.violations == baseline.violations
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")} == expected
    assert not list(tmp_path.rglob("*.part"))
    assert pipeline.analyze(assets, tmp_path, workers=3) == baseline.shots  # cache hits


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("stage", ["probe", "proxy", "shots", "quality", "pauses"])
def test_asset_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, stage: str
) -> None:
    assets = _assets()
    error = RuntimeError("asset failed")

    def fail(item: Path | MediaAsset, *args: object) -> MediaAsset | list[Shot]:
        asset_id = item.stem if isinstance(item, Path) else item.asset_id
        if asset_id == "1":
            raise error
        return assets[int(asset_id)] if stage in {"probe", "proxy"} else []

    monkeypatch.setattr(pipeline, "probe_asset", lambda path: assets[int(path.stem)])
    monkeypatch.setattr(pipeline, "make_proxy", lambda asset, workspace: asset)
    monkeypatch.setattr("vlogcut.analyze.detect_shots", lambda *a: [])
    monkeypatch.setattr("vlogcut.analyze.assess_shots", lambda *a: [])
    monkeypatch.setattr(pauses, "analyze_pauses", lambda *a: [])
    target = {
        "probe": "vlogcut.pipeline.probe_asset",
        "proxy": "vlogcut.pipeline.make_proxy",
        "shots": "vlogcut.analyze.detect_shots",
        "quality": "vlogcut.analyze.assess_shots",
        "pauses": "vlogcut.analyze.pauses.analyze_pauses",
    }[stage]
    monkeypatch.setattr(target, fail)
    with pytest.raises(RuntimeError) as raised:
        if stage in {"probe", "proxy"}:
            pipeline.ingest([Path(a.path) for a in assets], tmp_path, workers=workers)
        else:
            pipeline.analyze(assets, tmp_path, workers=workers)
    assert raised.value is error


@pytest.mark.parametrize("stage", ["probe", "proxy"])
def test_parallel_ingest_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    assets = _assets()
    finished = [Event() for _ in assets]
    completed: list[str] = []

    def finish(item: Path | MediaAsset, *args: object) -> MediaAsset:
        i = int(item.stem if isinstance(item, Path) else item.asset_id)
        if i < 2:
            assert finished[i + 1].wait(5), "ingest did not run concurrently"
        completed.append(str(i))
        finished[i].set()
        return assets[i]

    monkeypatch.setattr(pipeline, "probe_asset", lambda path: assets[int(path.stem)])
    monkeypatch.setattr(pipeline, "make_proxy", lambda asset, workspace: asset)
    monkeypatch.setattr(pipeline, "probe_asset" if stage == "probe" else "make_proxy", finish)
    result = pipeline.ingest([Path(a.path) for a in assets], tmp_path, workers=3)
    assert completed == ["2", "1", "0"]
    assert result == (assets, [])


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_workers_fail_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    def unexpected(*args: object) -> None:
        pytest.fail("started work")

    monkeypatch.setattr(pipeline, "find_footage", unexpected)
    for call in (
        lambda: pipeline.cut(tmp_path, tmp_path, workers=workers),
        lambda: pipeline.ingest([Path("0.mp4")], tmp_path, unexpected, workers=workers),
        lambda: pipeline.analyze(_assets(), tmp_path, unexpected, workers=workers),
    ):
        with pytest.raises(ValueError, match="workers must be at least 1"):
            call()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("workers", [1, 3])
def test_ingest_skip_reports_keep_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, workers: int
) -> None:
    assets = _assets()

    def probe(path: Path) -> MediaAsset:
        i = int(path.stem)
        sleep(0.02 * (2 - i))
        if i != 1:
            raise ProbeError(f"bad {i}")
        return assets[i]

    monkeypatch.setattr(pipeline, "probe_asset", probe)
    monkeypatch.setattr(pipeline, "make_proxy", lambda asset, workspace: asset)
    found, skipped = pipeline.ingest([Path(a.path) for a in assets], tmp_path, workers=workers)
    assert found == [assets[1]]
    assert skipped == [("0.mp4", "bad 0"), ("2.mp4", "bad 2")]
    assert [r.message for r in caplog.records] == ["skipping 0.mp4: bad 0", "skipping 2.mp4: bad 2"]


@pytest.mark.parametrize("stage", ["ingest", "analyze"])
def test_duplicate_ids_do_not_write_caches_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    asset = _assets()[0]
    active = Lock()

    def write(*args: object) -> MediaAsset | list[Shot]:
        assert active.acquire(blocking=False), "concurrent writes to the same asset cache"
        try:
            sleep(0.02)
            return asset if stage == "ingest" else []
        finally:
            active.release()

    monkeypatch.setattr(pipeline, "probe_asset", lambda path: asset)
    monkeypatch.setattr(pipeline, "make_proxy", write)
    monkeypatch.setattr("vlogcut.analyze.detect_shots", write)
    monkeypatch.setattr("vlogcut.analyze.assess_shots", lambda *a: [])
    monkeypatch.setattr(pauses, "analyze_pauses", lambda *a: [])
    if stage == "ingest":
        assert pipeline.ingest([Path(asset.path)] * 3, tmp_path, workers=3) == ([asset] * 3, [])
    else:
        assert pipeline.analyze([asset] * 3, tmp_path, workers=3) == []


def test_validate_and_write_valid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "edl" / "coverage_1s.json"
    with caplog.at_level(logging.INFO, logger="vlogcut.pipeline"):
        assert pipeline.validate_and_write(_edl(), _shots(), path, target_s=1) == []
    assert json.loads(path.with_suffix(".violations.json").read_text(encoding="utf-8")) == []
    assert [(r.levelname, r.message) for r in caplog.records] == [("INFO", "validation: OK")]


def test_validate_and_write_unknown_then_overwrite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "edl" / "coverage_1s_fast.json"
    violations = pipeline.validate_and_write(_edl("missing"), _shots(), path, target_s=1)
    report = path.with_suffix(".violations.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data == [v.model_dump(mode="json") for v in violations]
    assert data == [
        {"code": "unknown_shot", "clip_index": 0, "message": "Shot 'missing' is absent from shots"}
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "unknown_shot" in caplog.records[0].message
    assert pipeline.validate_and_write(_edl(), _shots(), path, target_s=1) == []
    assert json.loads(report.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("styled", [False, True])
@pytest.mark.parametrize("render", [False, True])
def test_cut_validates_with_requested_target_and_style_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    styled: bool,
    render: bool,
) -> None:
    shots, edl = _shots(), _edl()
    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=10, fps=30, width=320, height=240)
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    monkeypatch.setattr("vlogcut.plan.rough_cut", lambda *args, **kwargs: edl)
    monkeypatch.setattr(pipeline, "repair_edl", lambda *args: pytest.fail("unexpected repair"))
    profile = StyleProfile(shot_duration_s=DurationStats(p10=2, median=3, p90=4), cuts_per_min=20)
    style = tmp_path / "fast.json" if styled else None
    if style is not None:
        style.write_text(profile.model_dump_json(), encoding="utf-8")
    stem = "coverage_3s_fast" if styled else "coverage_3s"
    report = tmp_path / "edl" / f"{stem}.violations.json"
    rendered = []

    def render_preview(*args: object, **kwargs: object) -> None:
        assert json.loads(report.read_text(encoding="utf-8"))
        rendered.append(args[0])

    monkeypatch.setattr("vlogcut.render.render_preview", render_preview)
    result = pipeline.cut(tmp_path, tmp_path, target_s=3, style=style, render=render)
    expected = ["clip_too_short", "total_duration"] if styled else ["total_duration"]
    assert [v.code for v in result.violations] == expected
    assert result.violations[-1].clip_index is None
    assert len(caplog.records) == len(expected)
    assert all(r.levelname == "WARNING" for r in caplog.records)
    assert result.profile == (profile if styled else None)
    assert result.edl_path == report.with_name(f"{stem}.json")
    assert json.loads(report.read_text(encoding="utf-8")) == [
        v.model_dump(mode="json") for v in result.violations
    ]
    assert rendered == ([edl] if render else [])
    assert not result.repaired


@pytest.mark.parametrize("case", ["clean", "mixed", "all_unknown"])
def test_cut_repair_updates_files_and_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    shots = _shots()
    rejected = shots[0].model_copy(deep=True)
    rejected.shot_id = "rejected"
    rejected.quality.rejected = True
    shots.append(rejected)
    retained = Clip(shot_id="s", in_s=2, out_s=4, speed=2, audio="duck")
    clips = [retained]
    if case == "mixed":
        clips += [Clip(shot_id="rejected", in_s=0, out_s=1)]
    if case != "clean":
        clips += [Clip(shot_id="missing", in_s=0, out_s=1)]
    if case == "all_unknown":
        clips = clips[-1:]
    edl = EDL(variant_id="test", target_duration_s=len(clips), clips=clips)
    original_json = edl.model_dump_json(indent=1)
    target_s = edl.total_duration_s
    original_violations = validate_edl(edl, shots, target_s=target_s)
    # Baseline serialization uses the same platform newline handling as cut.
    baseline = tmp_path / "original.json"
    baseline.write_text(original_json, encoding="utf-8")
    pipeline.validate_and_write(edl, shots, baseline, target_s=target_s)
    original_report = baseline.with_suffix(".violations.json").read_bytes()
    caplog.clear()

    asset = MediaAsset(asset_id="a", path="a.mp4", duration_s=10, fps=30, width=320, height=240)
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    monkeypatch.setattr("vlogcut.plan.rough_cut", lambda *args, **kwargs: edl)
    if case == "clean":
        monkeypatch.setattr(
            pipeline, "repair_edl", lambda *args: pytest.fail("clean EDL must bypass repair")
        )
    edl_path = tmp_path / "edl" / f"coverage_{target_s:.0f}s.json"
    report = edl_path.with_suffix(".violations.json")
    rendered: list[EDL] = []

    def render_preview(final_edl: EDL, *args: object, **kwargs: object) -> None:
        assert EDL.model_validate_json(edl_path.read_text(encoding="utf-8")) == final_edl
        assert json.loads(report.read_text(encoding="utf-8")) == [
            v.model_dump(mode="json") for v in validate_edl(final_edl, shots, target_s=target_s)
        ]
        rendered.append(final_edl)

    monkeypatch.setattr("vlogcut.render.render_preview", render_preview)
    result = pipeline.cut(tmp_path, tmp_path, target_s=target_s)
    assert result.repaired == (case == "mixed")
    assert rendered == [result.edl]
    assert json.loads(report.read_text(encoding="utf-8")) == [
        v.model_dump(mode="json") for v in result.violations
    ]
    assert edl.model_dump_json(indent=1) == original_json
    assert not edl_path.with_suffix(".json.part").exists()
    if case == "mixed":
        assert {v.code for v in original_violations} == {"unknown_shot", "rejected_shot"}
        assert result.edl.clips == [retained]  # in/out, speed and audio all preserved
        assert [v.code for v in result.violations] == ["total_duration"]
    else:
        assert result.edl == edl
        assert result.violations == original_violations
        assert edl_path.read_bytes() == baseline.read_bytes()
        assert report.read_bytes() == original_report
    if case == "all_unknown":
        assert any(
            r.levelname == "WARNING" and "keeping original EDL and violations" in r.message
            for r in caplog.records
        )


@pytest.mark.parametrize(
    ("dissolve_share", "duration", "code", "expected"),
    [
        (0.0, 0.5, "transition_not_allowed", Transition()),
        (1.0, 4.0, "dissolve_too_long", Transition(kind="dissolve", duration_s=1)),
    ],
)
def test_repair_if_needed_fixes_transitions(
    dissolve_share: float, duration: float, code: str, expected: Transition
) -> None:
    shots = _shots()
    shots.append(shots[0].model_copy(update={"shot_id": "s2"}))
    edl = _edl()
    edl.clips.append(
        Clip(
            shot_id="s2",
            in_s=1,
            out_s=2,
            transition_in=Transition(kind="dissolve", duration_s=duration),
        )
    )
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=1, p90=1),
        cuts_per_min=60,
        transitions=TransitionMix(hard_cut=1 - dissolve_share, dissolve=dissolve_share),
    )
    violations = validate_edl(edl, shots, profile, target_s=2)
    assert code in {v.code for v in violations}
    final_edl, repaired = pipeline.repair_if_needed(edl, shots, violations, profile)
    assert repaired
    assert final_edl.clips[1].transition_in == expected
    assert validate_edl(final_edl, shots, profile, target_s=2) == []
    assert [(c.in_s, c.out_s) for c in final_edl.clips] == [(0, 1), (1, 2)]


@pytest.mark.parametrize("styled", [False, True])
@pytest.mark.parametrize("case", ["clean", "retry", "deterministic_repair"])
def test_cut_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_analysis: None,
    styled: bool,
    case: str,
) -> None:
    valid = _edl()
    bad = valid.model_copy(deep=True)
    bad.clips.append(Clip(shot_id="missing", in_s=0, out_s=1))
    replies = {
        "clean": [valid.model_dump_json()],
        "retry": [bad.model_dump_json(), valid.model_dump_json()],
        "deterministic_repair": [bad.model_dump_json()],
    }
    client = FakeClient(replies[case])
    router = _router(tmp_path, client)
    style = tmp_path / "fast.json" if styled else None
    if style:
        profile = StyleProfile(
            shot_duration_s=DurationStats(p10=1, median=1, p90=1), cuts_per_min=60
        )
        style.write_text(profile.model_dump_json(), encoding="utf-8")
    stem = "coverage_1s" + ("_fast" if styled else "") + "_llm"
    edl_path = tmp_path / "edl" / f"{stem}.json"
    rendered: list[EDL] = []

    def render_preview(edl: EDL, *args: object, **kwargs: object) -> None:
        assert EDL.model_validate_json(edl_path.read_text(encoding="utf-8")) == edl
        assert json.loads(edl_path.with_suffix(".violations.json").read_text()) == []
        rendered.append(edl)

    monkeypatch.setattr("vlogcut.render.render_preview", render_preview)
    monkeypatch.setattr("vlogcut.plan.rough_cut", lambda *args: pytest.fail("unexpected rules"))
    result = pipeline.cut(
        tmp_path, tmp_path, target_s=1, planner="llm", router=router, style=style,
        brief="A quiet morning", max_repair_rounds=0 if case == "deterministic_repair" else 2,
    )  # fmt: skip
    assert result.planner == "llm" and not result.fallback
    assert result.repair_rounds == (1 if case == "retry" else 0)
    assert result.repaired == (case == "deterministic_repair")
    assert result.edl_path == edl_path
    assert result.preview_path == tmp_path / f"preview_{stem}.mp4"
    assert result.violations == [] and result.edl.clips == valid.clips
    assert rendered == [result.edl]
    assert len(client.requests) == result.repair_rounds + 1
    prompt = client.requests[0]["messages"][-1]["content"]
    assert "A quiet morning" in prompt and "Target duration: 1 seconds" in prompt
    assert ("Profile pacing:" in prompt) == styled
    if case == "retry":
        assert "unknown_shot" in client.requests[1]["messages"][-1]["content"]


def test_cut_llm_fallback_preserves_rules_outputs(
    tmp_path: Path, stub_analysis: None, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = pipeline.cut(tmp_path, tmp_path, target_s=3, render=False)
    edl_bytes = baseline.edl_path.read_bytes()
    report_bytes = baseline.edl_path.with_suffix(".violations.json").read_bytes()
    assert (baseline.planner, baseline.fallback, baseline.repair_rounds) == ("rules", False, 0)
    client = FakeClient(["not JSON"] * 3)
    result = pipeline.cut(
        tmp_path,
        tmp_path,
        target_s=3,
        render=False,
        planner="llm",
        router=_router(tmp_path, client),
    )
    assert (result.planner, result.fallback, result.repair_rounds) == ("rules", True, 0)
    assert result.edl_path == baseline.edl_path == tmp_path / "edl" / "coverage_3s.json"
    assert result.edl_path.read_bytes() == edl_bytes
    assert result.edl_path.with_suffix(".violations.json").read_bytes() == report_bytes
    assert len(client.requests) == 3
    assert any(
        r.levelname == "WARNING" and "falling back to rules" in r.message for r in caplog.records
    )


def test_cut_llm_requires_router_before_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "find_footage", lambda *args: pytest.fail("scanned footage"))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: pytest.fail("analyzed footage"))
    with pytest.raises(ValueError, match='planner="llm" requires a ModelRouter'):
        pipeline.cut(tmp_path, tmp_path, planner="llm")
