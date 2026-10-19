"""Offline planner tests using the real router with a scripted provider client."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut.llm.router import ModelRouter
from vlogcut.plan import (
    LLMPlanResult,
    PlannerError,
    build_planner_messages,
    format_shot_table,
    format_violations,
    parse_planner_response,
    plan_with_llm,
    plan_with_repair,
)
from vlogcut.schemas import (
    EDL,
    CutPoint,
    DurationStats,
    QualityFlags,
    Shot,
    StructureStats,
    StyleProfile,
    TransitionMix,
)
from vlogcut.validate.checks import Violation, validate_edl


class FakeClient:
    def __init__(self, output: str | list[str]) -> None:
        self.outputs = [output] if isinstance(output, str) else list(output)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(deepcopy(kwargs))
        message = SimpleNamespace(content=self.outputs.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def make_router(tmp_path: Path, output: str | list[str]) -> tuple[ModelRouter, FakeClient]:
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
    assert "caption, tags and interest (1–5)" in messages[0]["content"]
    user = messages[1]["content"]
    assert format_shot_table(shots) in user
    assert '"s1"' in user and '"s2"' in user
    assert "reject-me" not in str(messages)
    assert "Target duration: 10 seconds" in user
    assert "No style profile." in user
    assert "Brief: none." in user
    assert [shot.model_dump() for shot in shots] == before


@pytest.mark.parametrize("planner", [plan_with_llm, plan_with_repair])
def test_planner_forwards_ranker_scores(
    tmp_path: Path, shots: list[Shot], valid_json: str, planner: Any
) -> None:
    router, client = make_router(tmp_path, valid_json)
    scores = {"s1": 1.25, "s2": -2.5, "reject-me": 0.0}
    planner(router, shots, target_s=10, ranker_scores=scores)
    assert (
        format_shot_table(shots, ranker_scores=scores)
        in client.requests[0]["messages"][1]["content"]
    )


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


@pytest.mark.parametrize("intro_ratio", [0.5, 0.75])
def test_messages_describe_measured_structure(shots: list[Shot], intro_ratio: float) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=StructureStats(montage_share=0.3, intro_pace_ratio=intro_ratio),
    )
    before = profile.model_dump()
    messages = build_planner_messages(shots, target_s=60, profile=profile)
    assert messages == build_planner_messages(shots, target_s=60, profile=profile)
    user = messages[1]["content"]
    assert "Montage share=0.30: aim for 0.30 of clips in montage bursts." in user
    assert "at least three consecutive clips" in user
    assert "strictly shorter than half the profile median (2.50 s)" in user
    assert f"Intro pace ratio={intro_ratio:.2f}" in user
    assert "clips starting in the first 20 s of the timeline" in user
    assert f"median length about {intro_ratio:.2f} times the body median" in user
    assert f"approximately {intro_ratio * 5:.2f} s" in user
    assert profile.model_dump() == before


def test_messages_describe_zero_montage_and_omit_unmeasured_intro(shots: list[Shot]) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=StructureStats(montage_share=0),
    )
    user = build_planner_messages(shots, target_s=60, profile=profile)[1]["content"]
    assert "Montage share=0.00: no montage bursts are wanted." in user
    assert "at least three consecutive clips" in user
    assert "half the profile median (2.50 s)" in user
    assert "Intro pace" not in user
    assert "first 20 s" not in user


@pytest.mark.parametrize("alternation, formatted", [(0.62, "0.62"), (0.0, "0.00")])
def test_messages_describe_measured_kind_alternation(
    shots: list[Shot], alternation: float, formatted: str
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=StructureStats(
            montage_share=0.3, intro_pace_ratio=0.5, kind_alternation=alternation
        ),
    )
    before = profile.model_dump()
    messages = build_planner_messages(shots, target_s=60, profile=profile)
    line = (
        f"Talking/b-roll alternation: {formatted} of adjacent talking/b-roll clip pairs "
        "switch kind (1 = always alternate, 0 = long same-kind runs)."
    )
    assert messages[1]["content"].count(line) == 1
    unmeasured = profile.model_copy(
        update={"structure": StructureStats(montage_share=0.3, intro_pace_ratio=0.5)}
    )
    baseline = build_planner_messages(shots, target_s=60, profile=unmeasured)
    assert messages[0] == baseline[0]
    assert messages[1]["content"].replace(line + "\n", "") == baseline[1]["content"]
    assert profile.model_dump() == before


def test_unmeasured_kind_alternation_preserves_structure_prompt_bytes(shots: list[Shot]) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=StructureStats(montage_share=0.3, intro_pace_ratio=0.5),
    )
    baseline = build_planner_messages(shots, target_s=60, profile=profile)
    explicit_none = profile.model_copy(
        update={
            "structure": StructureStats(
                montage_share=0.3, intro_pace_ratio=0.5, kind_alternation=None
            )
        }
    )
    assert build_planner_messages(shots, target_s=60, profile=explicit_none) == baseline
    expected_user = (
        "Target duration: 60 seconds (target_s).\n"
        "Profile pacing: p10=1.0 s / median=5.0 s / p90=8.0 s; cuts per minute=12.0."
        " Dissolve share: not measured.\n"
        "Montage share=0.30: aim for 0.30 of clips in montage bursts. A burst is a run "
        "of at least three consecutive clips, each with a timeline duration strictly "
        "shorter than half the profile median (2.50 s).\n"
        "Intro pace ratio=0.50: clips starting in the first 20 s of the timeline "
        "should have a median length about 0.50 times the body median "
        "(approximately 2.50 s, using the profile median as the body estimate).\n"
        f"Brief: none.\nShot table (chronological order):\n{format_shot_table(shots)}"
    )
    assert baseline[1]["content"].encode("utf-8") == expected_user.encode("utf-8")
    assert baseline[0] == build_planner_messages(shots, target_s=60)[0]


@pytest.mark.parametrize("parse_error", [False, True])
def test_repair_retains_kind_alternation_once_in_conversation(
    tmp_path: Path, shots: list[Shot], valid_json: str, parse_error: bool
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=StructureStats(montage_share=0.3, kind_alternation=0.62),
    )
    raw = "not JSON" if parse_error else valid_json.replace('"s2"', '"missing"')
    router, client = make_router(tmp_path, [raw, valid_json])
    result = plan_with_repair(router, shots, target_s=10, profile=profile)
    assert result.rounds == 1
    assert result.violations == []
    assert len(client.requests) == 2
    initial = client.requests[0]["messages"]
    for request in client.requests:
        messages = request["messages"]
        assert messages[:2] == initial
        conversation = "\n".join(message["content"] for message in messages)
        assert conversation.count("Talking/b-roll alternation: 0.62") == 1
    assert "Talking/b-roll alternation" not in client.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("speech_ratio, formatted", [(0.35, "0.35"), (0.0, "0.00")])
@pytest.mark.parametrize(
    "structure",
    [None, StructureStats(montage_share=0.3, intro_pace_ratio=0.5, kind_alternation=0.62)],
)
def test_messages_describe_measured_speech_ratio(
    shots: list[Shot], speech_ratio: float, formatted: str, structure: StructureStats | None
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=structure,
        speech_ratio=speech_ratio,
    )
    before = profile.model_dump()
    messages = build_planner_messages(shots, target_s=60, profile=profile)
    assert messages == build_planner_messages(shots, target_s=60, profile=profile)
    line = (
        f"Speech ratio={formatted}: aim for roughly this share of the timeline duration "
        "from talking clips, with the rest from b-roll."
    )
    assert messages[1]["content"].splitlines().count(line) == 1
    baseline = build_planner_messages(
        shots, target_s=60, profile=profile.model_copy(update={"speech_ratio": None})
    )
    assert messages[0] == baseline[0]
    assert messages[1]["content"].replace(line + "\n", "") == baseline[1]["content"]
    assert profile.model_dump() == before


@pytest.mark.parametrize(
    "structure",
    [
        None,
        StructureStats(montage_share=0),
        StructureStats(montage_share=0.3, kind_alternation=0.62),
    ],
)
@pytest.mark.parametrize("cold_open_clips", [0, 2])
def test_unmeasured_speech_ratio_preserves_prompt_bytes(
    shots: list[Shot], structure: StructureStats | None, cold_open_clips: int
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=structure,
        cold_open_clips=cold_open_clips,
    )
    baseline = build_planner_messages(shots, target_s=60, profile=profile)
    explicit_none = profile.model_copy(update={"speech_ratio": None})
    messages = build_planner_messages(shots, target_s=60, profile=explicit_none)
    assert json.dumps(messages).encode("utf-8") == json.dumps(baseline).encode("utf-8")
    assert "Speech ratio" not in str(messages)


@pytest.mark.parametrize("speech_ratio", [0.35, 0.0])
@pytest.mark.parametrize("parse_error", [False, True])
def test_repair_retains_speech_ratio_once_in_conversation(
    tmp_path: Path, shots: list[Shot], valid_json: str, speech_ratio: float, parse_error: bool
) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        speech_ratio=speech_ratio,
    )
    raw = "not JSON" if parse_error else valid_json.replace('"s2"', '"missing"')
    router, client = make_router(tmp_path, [raw, valid_json])
    result = plan_with_repair(router, shots, target_s=10, profile=profile)
    assert result.rounds == 1
    assert result.violations == []
    assert len(client.requests) == 2
    initial = client.requests[0]["messages"]
    for request in client.requests:
        messages = request["messages"]
        assert messages[:2] == initial
        conversation = "\n".join(message["content"] for message in messages)
        assert conversation.count(f"Speech ratio={speech_ratio:.2f}") == 1
    assert "Speech ratio" not in client.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("structure", [None, StructureStats(montage_share=0.3)])
def test_messages_allow_cold_open(shots: list[Shot], structure: StructureStats | None) -> None:
    profile = StyleProfile(
        shot_duration_s=DurationStats(p10=1, median=5, p90=8),
        cuts_per_min=12,
        structure=structure,
        cold_open_clips=2,
    )
    system = build_planner_messages(shots, target_s=60, profile=profile)[0]["content"]
    assert (
        "Cold open: at most 2 leading clips may come from later in the day as a hook "
        "before the chronological timeline."
    ) in system
    assert "Keep the remaining clips in chronological order as given by the table" in system
    assert "select each shot at most once." in system
    assert "- Keep chronological order as given by the table" not in system


@pytest.mark.parametrize("profile_given", [False, True])
@pytest.mark.parametrize("dissolve_share", [None, 0.0, 0.25])
@pytest.mark.parametrize("constraints", [None, [], ["", " \t\n"]])
def test_unmeasured_profile_preserves_original_prompt_bytes(
    shots: list[Shot],
    profile_given: bool,
    dissolve_share: float | None,
    constraints: list[str] | None,
) -> None:
    profile = None
    pacing = "No style profile."
    if profile_given:
        profile = StyleProfile(
            shot_duration_s=DurationStats(p10=1.5, median=3, p90=7.5),
            cuts_per_min=18,
            transitions=(
                TransitionMix(hard_cut=1 - dissolve_share, dissolve=dissolve_share)
                if dissolve_share is not None
                else None
            ),
        )
        pacing = "Profile pacing: p10=1.5 s / median=3.0 s / p90=7.5 s; cuts per minute=18.0."
        pacing += (
            " Dissolve share: not measured."
            if dissolve_share is None
            else f" Dissolve share={dissolve_share}."
        )
    messages = build_planner_messages(shots, target_s=10, profile=profile, constraints=constraints)
    # Captured from the original system prompt before adding structure guidance.
    assert hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest() == (
        "a9923fc0954efa03f365e943650429602202bfa058be3b3b66612d9362f7394b"
    )
    expected_user = (
        f"Target duration: 10 seconds (target_s).\n{pacing}\nBrief: none.\n"
        f"Shot table (chronological order):\n{format_shot_table(shots)}"
    )
    assert messages[1]["content"].encode("utf-8") == expected_user.encode("utf-8")
    for wording in ("montage", "Intro pace", "first 20 s", "Cold open"):
        assert wording not in str(messages)


def test_constraints_are_ordered_and_one_per_line(shots: list[Shot]) -> None:
    constraints = ["  Lengthen the café part  ", "", " \t\n", "Keep the\narrival"]
    messages = build_planner_messages(shots, target_s=10, constraints=constraints)
    assert (
        "Edit constraints from feedback on a previous cut (follow them where possible):\n"
        "Lengthen the café part\nKeep the arrival\nShot table"
    ) in messages[1]["content"]
    assert messages[0] == build_planner_messages(shots, target_s=10)[0]
    assert constraints == ["  Lengthen the café part  ", "", " \t\n", "Keep the\narrival"]


@pytest.mark.parametrize("repair", [False, True])
def test_planning_retains_constraints_once_in_conversation(
    tmp_path: Path, shots: list[Shot], valid_json: str, repair: bool
) -> None:
    bad = json.loads(valid_json)
    bad["clips"][0]["shot_id"] = "missing"
    replies = ["not JSON", json.dumps(bad), valid_json] if repair else [valid_json]
    router, client = make_router(tmp_path, replies)
    planner = plan_with_repair if repair else plan_with_llm
    planner(router, shots, target_s=10, constraints=["Lengthen the café part"])
    assert len(client.requests) == (3 if repair else 1)
    initial = client.requests[0]["messages"]
    for request in client.requests:
        messages = request["messages"]
        assert messages[:2] == initial
        conversation = "\n".join(message["content"] for message in messages)
        assert conversation.count("Edit constraints from feedback") == 1
        assert conversation.count("Lengthen the café part") == 1


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


def test_format_violations_is_deterministic() -> None:
    violations = [
        Violation(code="unknown_shot", clip_index=0, message="Unknown shot 'missing'"),
        Violation(code="total_duration", clip_index=None, message="Total too short\nfor target"),
        Violation(code="out_of_bounds", clip_index=2, message="Offsets exceed bounds"),
    ]
    before = [v.model_dump() for v in violations]
    expected = (
        "clip 0 | unknown_shot | Unknown shot 'missing'\n"
        "clip - | total_duration | Total too short for target\n"
        "clip 2 | out_of_bounds | Offsets exceed bounds"
    )
    assert format_violations(violations) == format_violations(violations) == expected
    assert [v.model_dump() for v in violations] == before
    assert format_violations([]) == "No violations."


def test_repair_clean_first_reply(tmp_path: Path, shots: list[Shot], valid_json: str) -> None:
    router, client = make_router(tmp_path, valid_json)
    result = plan_with_repair(router, shots, target_s=10, brief="A quiet day.")
    assert isinstance(result, LLMPlanResult)
    assert result.edl == EDL.model_validate_json(valid_json)
    assert result.rounds == 0
    assert result.violations == []
    assert len(client.requests) == 1
    assert client.requests[0]["messages"] == build_planner_messages(
        shots, target_s=10, brief="A quiet day."
    )


def test_repair_unknown_shot(tmp_path: Path, shots: list[Shot], valid_json: str) -> None:
    raw = valid_json.replace('"s2"', '"unknown-shot"')
    router, client = make_router(tmp_path, [raw, valid_json])
    result = plan_with_repair(router, shots, target_s=10)
    assert result.edl == EDL.model_validate_json(valid_json)
    assert result.rounds == 1
    assert result.violations == []
    assert len(client.requests) == 2
    messages = client.requests[1]["messages"]
    assert messages[:-2] == client.requests[0]["messages"]
    assert messages[-2] == {"role": "assistant", "content": raw}
    assert messages[-1]["role"] == "user"
    assert "unknown_shot" in messages[-1]["content"]
    assert (
        format_violations(validate_edl(EDL.model_validate_json(raw), shots, target_s=10))
        in messages[-1]["content"]
    )


@pytest.mark.parametrize("raw", ["not JSON\nat all", '{"clips": []}'])
def test_repair_parse_error(tmp_path: Path, shots: list[Shot], valid_json: str, raw: str) -> None:
    router, client = make_router(tmp_path, [raw, valid_json])
    result = plan_with_repair(router, shots, target_s=10)
    assert result.rounds == 1
    assert result.violations == []
    assert len(client.requests) == 2
    messages = client.requests[1]["messages"]
    assert messages[-2] == {"role": "assistant", "content": raw}
    with pytest.raises(PlannerError) as exc:
        parse_planner_response(raw)
    assert messages[-1]["role"] == "user"
    assert str(exc.value) in messages[-1]["content"]


@pytest.mark.parametrize("max_rounds", [0, 1, 2])
def test_repair_exhausted_returns_last_parsed(
    tmp_path: Path, shots: list[Shot], valid_json: str, max_rounds: int
) -> None:
    replies = [valid_json.replace('"s2"', f'"missing-{i}"') for i in range(max_rounds + 1)]
    router, client = make_router(tmp_path, replies)
    result = plan_with_repair(router, shots, target_s=10, max_rounds=max_rounds)
    assert result.rounds == max_rounds
    assert result.edl == EDL.model_validate_json(replies[-1])
    assert result.violations == validate_edl(result.edl, shots, target_s=10)
    assert [v.code for v in result.violations] == ["unknown_shot"]
    assert len(client.requests) == max_rounds + 1
    for i, request in enumerate(client.requests):
        assert request["model"] == "m-plan"
        assert len(request["messages"]) == 2 + 2 * i
        if i:
            assert request["messages"][:-2] == client.requests[i - 1]["messages"]
            assert request["messages"][-2] == {"role": "assistant", "content": replies[i - 1]}


@pytest.mark.parametrize("max_rounds", [0, 2])
def test_repair_all_unparseable_raises(tmp_path: Path, shots: list[Shot], max_rounds: int) -> None:
    router, client = make_router(tmp_path, ["garbage"] * (max_rounds + 1))
    with pytest.raises(PlannerError, match="Invalid planner EDL"):
        plan_with_repair(router, shots, target_s=10, max_rounds=max_rounds)
    assert len(client.requests) == max_rounds + 1


def test_repair_retains_parsed_edl_after_later_parse_failure(
    tmp_path: Path, shots: list[Shot], valid_json: str
) -> None:
    raw = valid_json.replace('"s2"', '"unknown-shot"')
    router, client = make_router(tmp_path, ["garbage", raw, "garbage again"])
    result = plan_with_repair(router, shots, target_s=10)
    assert result.rounds == 2
    assert result.edl == EDL.model_validate_json(raw)
    assert [v.code for v in result.violations] == ["unknown_shot"]
    assert len(client.requests) == 3


def test_repair_checks_supplied_target_and_profile(
    tmp_path: Path, shots: list[Shot], valid_json: str
) -> None:
    profile = StyleProfile(shot_duration_s=DurationStats(p10=1, median=2, p90=3), cuts_per_min=30)
    router, client = make_router(tmp_path, [valid_json, valid_json])
    result = plan_with_repair(router, shots, target_s=20, profile=profile, max_rounds=1)
    assert result.rounds == 1
    assert [v.code for v in result.violations] == [
        "clip_too_long",
        "clip_too_long",
        "total_duration",
    ]
    assert result.violations == validate_edl(result.edl, shots, profile=profile, target_s=20)
    assert client.requests[0]["messages"] == build_planner_messages(
        shots, target_s=20, profile=profile
    )


def test_repair_negative_rounds_raises_before_call(tmp_path: Path, shots: list[Shot]) -> None:
    router, client = make_router(tmp_path, [])
    with pytest.raises(ValueError, match="max_rounds"):
        plan_with_repair(router, shots, target_s=10, max_rounds=-1)
    assert client.requests == []
