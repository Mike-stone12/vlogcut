"""Offline tests for ModelRouter. FakeClient stands in for any provider SDK."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vlogcut.llm.router import ModelRouter, StructuredOutputError, extract_json
from vlogcut.schemas import EDL

CONFIG = {
    "providers": {"fake": {"base_url": "http://fake/v1"}},
    "roles": {
        "planner": {"provider": "fake", "model": "m-plan", "json_mode": "schema"},
        "captioner": {"provider": "fake", "model": "m-cap", "json_mode": "object"},
    },
}

VALID_EDL = {
    "variant_id": "v1",
    "target_duration_s": 10,
    "clips": [{"shot_id": "s1", "in_s": 0, "out_s": 4}, {"shot_id": "s2", "in_s": 1, "out_s": 7}],
}


class FakeClient:
    """Returns scripted outputs in order and records every request."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = SimpleNamespace(content=self.outputs.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def make_router(tmp_path, outputs: list[str]) -> tuple[ModelRouter, FakeClient]:
    client = FakeClient(outputs)
    router = ModelRouter(CONFIG, log_path=tmp_path / "calls.jsonl", client_factory=lambda _: client)
    return router, client


def test_structured_valid_first_try(tmp_path):
    router, client = make_router(tmp_path, [json.dumps(VALID_EDL)])
    edl = router.structured("planner", [{"role": "user", "content": "plan"}], EDL)
    assert edl.total_duration_s == pytest.approx(10)
    req = client.requests[0]
    assert req["model"] == "m-plan"
    assert req["response_format"]["type"] == "json_schema"


def test_repair_after_schema_error(tmp_path):
    bad = {**VALID_EDL, "clips": [{"shot_id": "s1", "in_s": 5, "out_s": 2}]}
    router, client = make_router(tmp_path, [json.dumps(bad), json.dumps(VALID_EDL)])
    edl = router.structured("planner", [{"role": "user", "content": "plan"}], EDL)
    assert edl.variant_id == "v1"
    repair_msg = client.requests[1]["messages"][-1]["content"]
    assert "out_s must be > in_s" in repair_msg


def test_semantic_check_drives_repair(tmp_path):
    known = {"s1"}

    def check(edl: EDL) -> list[str]:
        return [f"unknown shot_id {c.shot_id}" for c in edl.clips if c.shot_id not in known]

    fixed = {**VALID_EDL, "clips": [{"shot_id": "s1", "in_s": 0, "out_s": 4}]}
    router, client = make_router(tmp_path, [json.dumps(VALID_EDL), json.dumps(fixed)])
    edl = router.structured("planner", [{"role": "user", "content": "p"}], EDL, check=check)
    assert [c.shot_id for c in edl.clips] == ["s1"]
    assert "unknown shot_id s2" in client.requests[1]["messages"][-1]["content"]


def test_gives_up_after_max_repairs(tmp_path):
    router, _ = make_router(tmp_path, ["not json"] * 3)
    with pytest.raises(StructuredOutputError) as exc:
        router.structured("planner", [{"role": "user", "content": "p"}], EDL, max_repairs=2)
    assert "not valid JSON" in exc.value.errors[0]


def test_json_object_mode_and_logging_redacts_images(tmp_path):
    router, client = make_router(tmp_path, [json.dumps(VALID_EDL)])
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
        }
    ]
    router.structured("captioner", msgs, EDL)
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    record = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert record["role"] == "captioner"
    assert "base64" not in json.dumps(record)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 2} hope this helps') == {"a": 2}


def test_unknown_role_and_provider():
    router = ModelRouter(CONFIG, client_factory=lambda _: None)
    with pytest.raises(KeyError):
        router.complete("nope", [])
    bad = {**CONFIG, "roles": {"x": {"provider": "missing", "model": "m"}}}
    with pytest.raises(ValueError):
        ModelRouter(bad)
