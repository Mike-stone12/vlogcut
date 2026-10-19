"""Offline captioner tests using tiny byte files and a fake provider client."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut.analyze import (
    CaptionError,
    CaptionResult,
    build_caption_messages,
    caption_shot,
    parse_caption,
)
from vlogcut.llm.router import ModelRouter


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
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
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
