"""Offline planner tests using the real router with a scripted provider client."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut.llm.router import ModelRouter
from vlogcut.plan import (
    PlannerError,
    build_planner_messages,
    format_shot_table,
    parse_planner_response,
    plan_with_llm,
)
from vlogcut.schemas import (
    EDL,
    CutPoint,
    DurationStats,
    QualityFlags,
    Shot,
    StyleProfile,
    TransitionMix,
)
from vlogcut.validate.checks import validate_edl


class FakeClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.output)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def make_router(tmp_path: Path, output: str) -> tuple[ModelRouter, FakeClient]:
    client = FakeClient(output)
    config = {
        "providers": {"fake": {"base_url": "http://fake/v1"}},
        "roles": {"planner": {"provider": "fake", "model": "m-plan"}},
    }
    router = ModelRouter(config, log_path=tmp_path / "calls.jsonl", client_factory=lambda _: client)
    return router, client


@pytest.fixture
def shots() -> list[Shot]:
    return [
        Shot(
            shot_id=shot_id,
            asset_id="asset",
            start_s=start,
            end_s=start + 10,
            quality=QualityFlags(
                sharpness=100,
                exposure_clip=0,
                shake=0,
                rejected=rejected,
                usable_range=(1, 9),
            ),
            cut_points=[CutPoint(time_s=start + 2, kind="pause")],
        )
        for shot_id, start, rejected in [
            ("s1", 20, False),
            ("reject-me", 30, True),
            ("s2", 40, False),
        ]
    ]


@pytest.fixture
def valid_json() -> str:
    return json.dumps(
        {
            "variant_id": "llm-test",
            "target_duration_s": 10,
            "clips": [
                {"shot_id": "s1", "in_s": 1, "out_s": 6},
                {"shot_id": "s2", "in_s": 2, "out_s": 7},
            ],
        }
    )


def test_messages_are_deterministic_and_include_filtered_table(shots: list[Shot]) -> None:
    before = [shot.model_dump() for shot in shots]
    messages = build_planner_messages(shots, target_s=10)
    assert messages == build_planner_messages(shots, target_s=10)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert format_shot_table(shots) in user
    assert '"s1"' in user and '"s2"' in user
    assert "reject-me" not in str(messages)
    assert "Target duration: 10 seconds" in user
    assert "No style profile." in user
    assert "Brief: none." in user
    assert [shot.model_dump() for shot in shots] == before


@pytest.mark.parametrize("dissolve_share", [None, 0.0, 0.25])
def test_messages_include_profile_and_brief(
    shots: list[Shot], dissolve_share: float | None
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1.5, median=3, p90=7.5),
        cuts_per_min=18,
        transitions=(
            TransitionMix(hard_cut=1 - dissolve_share, dissolve=dissolve_share)
            if dissolve_share is not None
            else None
        ),
    )
    messages = build_planner_messages(
        shots, target_s=10, profile=profile, brief="Focus on the cafe."
    )
    assert messages == build_planner_messages(
        shots, target_s=10, profile=profile, brief="Focus on the cafe."
    )
    user = messages[1]["content"]
    for value in ("p10=1.5 s", "median=3.0 s", "p90=7.5 s", "cuts per minute=18.0"):
        assert value in user
    assert "Brief: Focus on the cafe." in user
    assert "No style profile." not in user
    expected = (
        "Dissolve share: not measured."
        if dissolve_share is None
        else f"Dissolve share={dissolve_share}."
    )
    assert expected in user


@pytest.mark.parametrize("wrapper", [" {} \n", "```json\n{}\n```", "\n```\r\n{}\r\n```\n"])
def test_parse_plain_and_fenced_json(valid_json: str, wrapper: str) -> None:
    assert parse_planner_response(wrapper.format(valid_json)) == EDL.model_validate_json(valid_json)


@pytest.mark.parametrize("kind", ["garbage", "prose", "bad_dissolve", "empty_clips"])
def test_parse_rejects_invalid_output(valid_json: str, kind: str) -> None:
    data = json.loads(valid_json)
    if kind == "garbage":
        response = "This is not JSON.\nTry something else."
    elif kind == "prose":
        response = f"Here is your EDL: {valid_json}"
    else:
        if kind == "bad_dissolve":
            data["clips"][1]["transition_in"] = {"kind": "dissolve", "duration_s": 0}
        else:
            data["clips"] = []
        response = json.dumps(data)
    with pytest.raises(PlannerError) as exc:
        parse_planner_response(response)
    assert len(str(exc.value).splitlines()) == 1
    if kind == "bad_dissolve":
        assert "a dissolve must have duration > 0" in str(exc.value)


@pytest.mark.parametrize("unknown", [False, True])
def test_plan_routes_once_and_leaves_semantic_checks_to_validator(
    tmp_path: Path, shots: list[Shot], valid_json: str, unknown: bool
) -> None:
    data = json.loads(valid_json)
    if unknown:
        data["clips"][1]["shot_id"] = "unknown-shot"
    router, client = make_router(tmp_path, json.dumps(data))
    edl = plan_with_llm(router, shots, target_s=10, brief="A quiet day.")
    assert edl == EDL.model_validate(data)
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "m-plan"
    assert request["messages"] == build_planner_messages(shots, target_s=10, brief="A quiet day.")
    record = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert record["role"] == "planner"
    violations = validate_edl(edl, shots, target_s=10)
    assert [v.code for v in violations] == (["unknown_shot"] if unknown else [])
    if unknown:
        assert violations[0].clip_index == 1


def test_plan_garbage_raises_without_retry(tmp_path: Path, shots: list[Shot]) -> None:
    router, client = make_router(tmp_path, "garbage")
    with pytest.raises(PlannerError):
        plan_with_llm(router, shots, target_s=10)
    assert len(client.requests) == 1
