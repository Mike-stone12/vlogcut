"""Workspace cache portability without media tools or model calls."""

import json
from pathlib import Path
from typing import Any

import pytest

from vlogcut import pipeline
from vlogcut.analyze import quality, shots, transitions
from vlogcut.paths import from_workspace, to_workspace_relative
from vlogcut.schemas import MediaAsset, QualityFlags
from vlogcut.style import profile


def test_inside_path_is_posix_relative_and_round_trips(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    path = workspace / "keyframes" / "a" / "frame.jpg"
    value = to_workspace_relative(path, workspace)
    assert value == "keyframes/a/frame.jpg"
    assert from_workspace(value, workspace) == path


def test_outside_path_stays_absolute(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    path = tmp_path / "workspace-other" / "source.mp4"
    assert to_workspace_relative(path, workspace) == str(path)
    assert from_workspace(str(path), workspace) == path


def test_relative_value_uses_new_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    value = to_workspace_relative(Path("old/keyframes/frame.jpg"), Path("old"))
    assert from_workspace(value, Path("new")) == tmp_path / "new/keyframes/frame.jpg"


def _fail(*args: Any, **kwargs: Any) -> Any:
    pytest.fail("expensive analysis ran on a cache hit")


def _asset(source: Path, workspace: Path) -> MediaAsset:
    return MediaAsset(
        asset_id="a", path=str(source), proxy_path=str(workspace / "proxies/a.mp4"),
        duration_s=4, fps=30, width=64, height=64, has_audio=False,
    )  # fmt: skip


@pytest.mark.parametrize("legacy_absolute", [False, True])
def test_analysis_caches_relocate_or_load_legacy_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_absolute: bool
) -> None:
    workspace = tmp_path / "original"
    source = tmp_path / "source.mp4"
    source.touch()
    external_frame = tmp_path / "external.jpg"
    external_frame.touch()
    monkeypatch.setattr(pipeline, "probe_asset", lambda path: _asset(path, workspace))
    monkeypatch.setattr(pipeline, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots, "detect_cuts", lambda *a, **kw: [])
    monkeypatch.setattr(transitions, "load_frames", lambda *a: None)
    monkeypatch.setattr(transitions, "find_dissolves", lambda *a: [])

    def extract(video: str, times: list[float], stem: Path) -> list[str]:
        path = stem.with_suffix(".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return [str(path), str(external_frame)]

    monkeypatch.setattr(shots, "extract_keyframes", extract)
    monkeypatch.setattr(quality, "measure_shot", lambda *a: None)
    monkeypatch.setattr(
        quality, "assess", lambda *a: QualityFlags(sharpness=200, exposure_clip=0, shake=0)
    )
    first = pipeline.cut(tmp_path, workspace, target_s=4, render=False)
    relative = "keyframes/a/a_000.jpg"
    assert first.shots[0].keyframes == [str(workspace / relative), str(external_frame)]
    cache_paths = [workspace / "shots/a.json", workspace / "shots/a.quality.json"]
    for cache_path, field in zip(cache_paths, ["spans", "shots"], strict=True):
        document = json.loads(cache_path.read_text(encoding="utf-8"))
        assert document[field][0]["keyframes"] == [relative, str(external_frame)]
        if legacy_absolute:
            document[field][0]["keyframes"] = first.shots[0].keyframes
            cache_path.write_text(json.dumps(document), encoding="utf-8")
    snapshot = json.loads((workspace / "shots.json").read_text(encoding="utf-8"))
    assert snapshot[0]["keyframes"] == [relative, str(external_frame)]
    before = [path.read_bytes() for path in cache_paths]

    if not legacy_absolute:
        moved = tmp_path / "renamed"
        workspace.rename(moved)
        workspace = moved
    for module, name in (
        (shots, "detect_cuts"),
        (shots, "extract_keyframes"),
        (transitions, "load_frames"),
        (quality, "measure_shot"),
        (quality, "assess"),
    ):
        monkeypatch.setattr(module, name, _fail)
    second = pipeline.cut(tmp_path, workspace, target_s=4, render=False)
    assert second.shots[0].keyframes == [str(workspace / relative), str(external_frame)]
    assert all(Path(k).is_absolute() and Path(k).is_file() for k in second.shots[0].keyframes)
    assert second.assets[0].path == str(source)
    assert second.assets[0].proxy_path == str(workspace / "proxies/a.mp4")
    assert second.edl == first.edl
    assert [
        (workspace / "shots" / path.name).read_bytes() for path in cache_paths
    ] == before  # hits do not rewrite or migrate caches


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("legacy_absolute", [False, True])
@pytest.mark.parametrize("relative_input", [False, True])
def test_reference_cache_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    internal: bool,
    legacy_absolute: bool,
    relative_input: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "original"
    workspace.mkdir()
    source = (workspace if internal else tmp_path) / "reference.mp4"
    source.touch()
    monkeypatch.setattr(profile, "probe_asset", lambda path: _asset(path, workspace))
    monkeypatch.setattr(profile, "make_proxy", lambda asset, ws: asset)
    monkeypatch.setattr(shots, "detect_cuts", lambda *a, **kw: [2.0])
    params = profile.ReferenceParams(dissolves=False, beats=False)
    supplied = str(source.relative_to(tmp_path) if relative_input else source)
    first = profile.analyze_reference(supplied, workspace, params)
    cache_path = workspace / "style/a.cuts.json"
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    assert document["source"] == ("reference.mp4" if internal else str(source))
    assert first.source == supplied
    if legacy_absolute:
        document["source"] = str(source)
        cache_path.write_text(json.dumps(document), encoding="utf-8")
    else:
        moved = tmp_path / "renamed"
        workspace.rename(moved)
        workspace = moved
        if internal:
            source = workspace / "reference.mp4"
    cache_path = workspace / "style/a.cuts.json"
    before = cache_path.read_bytes()
    monkeypatch.setattr(shots, "detect_cuts", _fail)
    loaded = profile._load_cache(cache_path, "a", params, workspace)
    assert loaded is not None and loaded.source == str(source)
    supplied = str(source.relative_to(tmp_path) if relative_input else source)
    second = profile.analyze_reference(supplied, workspace, params)
    assert second.source == supplied
    assert profile.aggregate([second], params).source_refs == [supplied]
    assert second.cuts == first.cuts
    assert cache_path.read_bytes() == before
