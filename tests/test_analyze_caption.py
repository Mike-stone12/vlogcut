"""Offline captioner tests using tiny byte files and a fake provider client."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut.analyze import (
    CaptionError,
    CaptionResult,
    build_caption_messages,
    caption,
    caption_shot,
    caption_shots,
    parse_caption,
)
from vlogcut.analyze.caption import CaptionCache
from vlogcut.llm.router import ModelRouter
from vlogcut.schemas import QualityFlags, Shot


@pytest.fixture
def keyframes(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / f"frame{i}.jpg" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([0, i, 255]))
    return paths


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("context", ["", "At the cafe"])
def test_messages(keyframes: list[Path], count: int, context: str) -> None:
    frames = keyframes[:count]
    messages = build_caption_messages(frames, context=context)
    assert messages == build_caption_messages(frames, context=context)
    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"]
    assert "Return only JSON" in system
    assert all(field in system for field in ("description", "tags", "interest"))
    expected = [{"type": "text", "text": context}] if context else []
    for path in frames:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        expected.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
        )
    assert messages[1]["content"] == expected


@pytest.mark.parametrize("count", [0, 4])
def test_invalid_keyframe_count(tmp_path: Path, count: int) -> None:
    # Count validation happens before reading even nonexistent files.
    with pytest.raises(ValueError, match="keyframes"):
        build_caption_messages([tmp_path / "missing.jpg"] * count)


@pytest.mark.parametrize("fence", [None, "", "json", "JSON"])
@pytest.mark.parametrize("interest", [1, 3, 5])
def test_parse_plain_and_fenced_json(fence: str | None, interest: int) -> None:
    payload = json.dumps({"description": "  Coffee brewing  ", "tags": [], "interest": interest})
    if fence is not None:
        payload = f"```{fence}\n{payload}\n```"
    assert parse_caption(f" \n{payload}\n ") == CaptionResult(
        description="Coffee brewing", tags=[], interest=interest
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"description":"Coffee","tags":[],"interest":0}',
        '{"description":"Coffee","tags":[],"interest":6}',
        '{"description":" \\n\\t ","tags":[],"interest":3}',
        "not JSON\nsecond line",
        "{}",
        '{"description":"Coffee","tags":[1],"interest":3}',
        'Here it is: {"description":"Coffee","tags":[],"interest":3}',
        '```json\n{"description":"Coffee","tags":[],"interest":3}\n```\nDone.',
    ],
)
def test_invalid_output_raises_one_line_error(payload: str) -> None:
    with pytest.raises(CaptionError) as exc:
        parse_caption(payload)
    assert len(str(exc.value).splitlines()) == 1


def test_tag_normalization_and_limit() -> None:
    tags = [" cafe ", "", "coffee", "cafe", " \n ", " coffee "]
    result = parse_caption(json.dumps({"description": "Coffee", "tags": tags, "interest": 3}))
    assert result.tags == ["cafe", "coffee"]
    ten_tags = [f"tag{i}" for i in range(10)]
    result = CaptionResult(description="Coffee", tags=ten_tags + ["tag0", " "], interest=3)
    assert result.tags == ten_tags
    with pytest.raises(CaptionError, match="at most 10"):
        parse_caption(
            json.dumps({"description": "Coffee", "tags": ten_tags + ["extra"], "interest": 3})
        )


class FakeClient:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        message = SimpleNamespace(content=self.output)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.mark.parametrize("valid", [True, False])
def test_caption_shot_calls_captioner_once(
    tmp_path: Path, keyframes: list[Path], valid: bool
) -> None:
    expected = CaptionResult(description="Coffee brewing", tags=["cafe"], interest=4)
    client = FakeClient(expected.model_dump_json() if valid else "not JSON")
    log_path = tmp_path / "calls.jsonl"
    router = ModelRouter(
        {
            "providers": {"fake": {"base_url": "http://fake/v1"}},
            "roles": {"captioner": {"provider": "fake", "model": "m-cap"}},
        },
        log_path=log_path,
        client_factory=lambda _: client,
    )
    if valid:
        assert caption_shot(router, keyframes, context="Morning") == expected
    else:
        with pytest.raises(CaptionError):
            caption_shot(router, keyframes, context="Morning")
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "m-cap"
    assert request["messages"] == build_caption_messages(keyframes, context="Morning")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "captioner"


@pytest.fixture
def batch_router(tmp_path: Path) -> tuple[ModelRouter, FakeClient]:
    client = FakeClient(
        CaptionResult(description="Coffee", tags=["cafe"], interest=4).model_dump_json()
    )
    router = ModelRouter(
        {
            "providers": {"fake": {"base_url": "http://fake/v1"}},
            "roles": {"captioner": {"provider": "fake", "model": "m-cap"}},
        },
        log_path=tmp_path / "calls.jsonl",
        client_factory=lambda _: client,
    )
    return router, client


@pytest.fixture
def shots(keyframes: list[Path]) -> list[Shot]:
    return [
        Shot(
            shot_id=shot_id,
            asset_id="asset",
            start_s=i,
            end_s=i + 1,
            keyframes=[str(path) for path in keyframes],
            quality=QualityFlags(sharpness=100, exposure_clip=0, shake=0),
        )
        for i, shot_id in enumerate(["s2", "s1"])
    ]


def test_batch_cache_miss_hit_and_order(
    tmp_path: Path, shots: list[Shot], batch_router: tuple[ModelRouter, FakeClient]
) -> None:
    router, client = batch_router
    before = [shot.model_dump() for shot in shots]
    progress: list[str] = []
    first = caption_shots(shots, router, tmp_path, model="m-cap", progress=progress.append)
    assert list(first) == ["s2", "s1"]
    assert len(client.requests) == 2
    assert all(request["model"] == "m-cap" for request in client.requests)
    for shot in shots:
        path = tmp_path / "captions" / f"{shot.shot_id}.json"
        cache = CaptionCache.model_validate_json(path.read_text(encoding="utf-8"))
        assert cache.shot_id == shot.shot_id
        assert cache.result == first[shot.shot_id]
        assert cache.key.model == "m-cap"
        assert cache.key.keyframe_sha1 == [
            hashlib.sha1(Path(frame).read_bytes()).hexdigest() for frame in shot.keyframes
        ]
    client.requests.clear()
    second = caption_shots(shots[::-1], router, tmp_path, model="m-cap", progress=progress.append)
    assert second == first
    assert list(second) == ["s1", "s2"]
    assert client.requests == []
    assert progress == ["Captioned s2", "Captioned s1", "Captioned s1", "Captioned s2"]
    assert not list((tmp_path / "captions").glob("*.part"))
    assert [shot.model_dump() for shot in shots] == before
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [record["role"] for record in records] == ["captioner", "captioner"]


@pytest.mark.parametrize("change", ["model", "bytes", "order", "context", "prompt"])
def test_batch_cache_key_invalidation(
    tmp_path: Path,
    shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    router, client = batch_router
    shot = shots[0]
    caption_shots([shot], router, tmp_path, model="m-cap")
    model, context = "m-cap", ""
    if change == "model":
        model = "another-model"
    elif change == "bytes":
        Path(shot.keyframes[0]).write_bytes(b"changed")
    elif change == "order":
        shot.keyframes.reverse()
    elif change == "context":
        context = "Morning"
    else:
        monkeypatch.setattr(caption, "CAPTION_PROMPT_VERSION", 2)
    client.output = '{"description":"New caption","tags":[],"interest":2}'
    result = caption_shots([shot], router, tmp_path, model=model, context=context)
    assert len(client.requests) == 2
    assert result[shot.shot_id].description == "New caption"
    assert caption_shots([shot], router, tmp_path, model=model, context=context) == result
    assert len(client.requests) == 2


@pytest.mark.parametrize("damage", ["json", "encoding", "shot_id", "result", "key"])
def test_batch_invalid_cache_recomputed(
    tmp_path: Path,
    shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    damage: str,
) -> None:
    router, client = batch_router
    shot = shots[0]
    expected = caption_shots([shot], router, tmp_path, model="m-cap")
    path = tmp_path / "captions" / f"{shot.shot_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if damage == "json":
        path.write_text("broken JSON", encoding="utf-8")
    elif damage == "encoding":
        path.write_bytes(b"\xff")
    else:
        if damage == "shot_id":
            data["shot_id"] = "wrong-shot"
        elif damage == "result":
            data["result"]["interest"] = 6
        else:
            del data["key"]["prompt_version"]
        path.write_text(json.dumps(data), encoding="utf-8")
    assert caption_shots([shot], router, tmp_path, model="m-cap") == expected
    assert len(client.requests) == 2
    CaptionCache.model_validate_json(path.read_text(encoding="utf-8"))


def test_batch_skip_rules(
    tmp_path: Path,
    shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    router, client = batch_router
    shots[0].quality.rejected = True
    shots[1].keyframes = []
    assert caption_shots(shots, router, tmp_path, model="m-cap") == {}
    assert client.requests == []
    assert "Skipping shot s1: no keyframes" in caplog.text
    assert not (tmp_path / "captions").exists()
    result = caption_shots(shots, router, tmp_path, model="m-cap", include_rejected=True)
    assert list(result) == ["s2"]
    assert len(client.requests) == 1
    assert caption_shots(shots, router, tmp_path, model="m-cap") == {}
    assert len(client.requests) == 1  # Rejection also excludes a cached result.


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), "invalid JSON"])
def test_batch_core_error_leaves_no_cache(
    tmp_path: Path,
    shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    failure: Exception | str,
) -> None:
    router, client = batch_router
    client.output = failure
    with pytest.raises(RuntimeError) as exc:
        caption_shots(shots, router, tmp_path, model="m-cap")
    if isinstance(failure, Exception):
        assert exc.value is failure
    else:
        assert isinstance(exc.value, CaptionError)
    assert len(client.requests) == 1
    assert not list((tmp_path / "captions").glob("*"))


def test_batch_atomic_write_cleanup(
    tmp_path: Path,
    shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router, _ = batch_router

    def fail_replace(source: Path, target: Path) -> None:
        assert source == target.with_suffix(".part")
        CaptionCache.model_validate_json(source.read_text(encoding="utf-8"))
        assert not target.exists()
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        caption_shots(shots, router, tmp_path, model="m-cap")
    assert not list((tmp_path / "captions").glob("*"))


@pytest.fixture
def grouped_shots(shots: list[Shot]) -> list[Shot]:
    representative, ungrouped = shots
    representative.dup_group = "take"
    peers = [
        representative.model_copy(deep=True, update={"shot_id": shot_id})
        for shot_id in ("s3", "s4")
    ]
    # Interleave an ungrouped shot to check global input order is preserved.
    return [representative, ungrouped, *peers]


def test_batch_dedup_copies_results(
    tmp_path: Path,
    grouped_shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    router, client = batch_router
    before = [shot.model_dump() for shot in grouped_shots]
    progress: list[str] = []
    with caplog.at_level("DEBUG", logger="vlogcut.analyze.caption"):
        results = caption_shots(
            grouped_shots, router, tmp_path, model="m-cap", progress=progress.append
        )
    assert len(client.requests) == 2
    assert list(results) == [shot.shot_id for shot in grouped_shots]
    assert progress == [f"Captioned {shot.shot_id}" for shot in grouped_shots]
    assert "Dedup saved 2 captioner calls" in caplog.text
    assert [shot.model_dump() for shot in grouped_shots] == before
    assert all(result == parse_caption(client.output) for result in results.values())
    assert len({id(result.tags) for result in results.values()}) == 4
    results["s3"].tags.append("changed")
    assert results["s2"].tags == results["s4"].tags == ["cafe"]


def test_batch_dedup_rejected_first(
    tmp_path: Path,
    grouped_shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
) -> None:
    router, client = batch_router
    rejected, _, representative, peer = grouped_shots
    rejected.quality.rejected = True
    # Distinct frames prove that the first eligible member is sent to the router.
    representative.keyframes = representative.keyframes[1:2]
    peer.keyframes = peer.keyframes[2:]
    results = caption_shots([rejected, representative, peer], router, tmp_path, model="m-cap")
    assert list(results) == [representative.shot_id, peer.shot_id]
    assert len(client.requests) == 1
    assert client.requests[0]["messages"] == build_caption_messages(
        [Path(path) for path in representative.keyframes]
    )
    assert results[representative.shot_id] == results[peer.shot_id]
    assert rejected.caption is None and rejected.tags == [] and rejected.interest is None
    # Rejection also prevents copying after a successful representative.
    assert (
        caption_shots([representative, rejected, peer], router, tmp_path, model="m-cap") == results
    )
    assert len(client.requests) == 1


def test_batch_ungrouped_still_captioned_individually(
    tmp_path: Path, shots: list[Shot], batch_router: tuple[ModelRouter, FakeClient]
) -> None:
    router, client = batch_router
    results = caption_shots(shots, router, tmp_path, model="m-cap")
    assert all(shot.dup_group is None for shot in shots)
    assert len(client.requests) == len(shots)
    assert list(results) == [shot.shot_id for shot in shots]
    assert all(result == parse_caption(client.output) for result in results.values())


def test_batch_dedup_warm_cache(
    tmp_path: Path,
    grouped_shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
) -> None:
    router, client = batch_router
    first = caption_shots(grouped_shots, router, tmp_path, model="m-cap")
    assert len(client.requests) == 2
    assert {path.stem for path in (tmp_path / "captions").glob("*.json")} == {"s2", "s1"}
    client.output = RuntimeError("warm cache must not call the provider")
    second = caption_shots(grouped_shots, router, tmp_path, model="m-cap")
    assert second == first
    assert len(client.requests) == 2
    assert len({id(result.tags) for result in second.values()}) == 4


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), "invalid JSON", ""])
def test_batch_dedup_failure_does_not_retry_peer(
    tmp_path: Path,
    grouped_shots: list[Shot],
    batch_router: tuple[ModelRouter, FakeClient],
    failure: Exception | str,
) -> None:
    router, client = batch_router
    client.output = failure
    before = [shot.model_dump() for shot in grouped_shots]
    with pytest.raises(RuntimeError):
        caption_shots(grouped_shots, router, tmp_path, model="m-cap")
    assert len(client.requests) == 1
    assert [shot.model_dump() for shot in grouped_shots] == before
    assert not list((tmp_path / "captions").glob("*"))


def test_batch_dedup_missing_representative_frames_does_not_retry_peer(
    tmp_path: Path, shots: list[Shot], batch_router: tuple[ModelRouter, FakeClient]
) -> None:
    router, client = batch_router
    for shot in shots:
        shot.dup_group = ""  # Empty strings are groups too; only None is ungrouped.
    shots[0].keyframes = []
    assert caption_shots(shots, router, tmp_path, model="m-cap") == {}
    assert client.requests == []
    assert not (tmp_path / "captions").exists()
