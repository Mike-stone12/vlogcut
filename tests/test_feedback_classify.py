"""Offline feedback classification through a real router with a fake provider."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut.feedback import (
    FeedbackClassification,
    FeedbackClassificationError,
    ProfileDelta,
    apply_profile_delta,
    build_feedback_messages,
    classify_feedback,
    parse_classification,
)
from vlogcut.llm.router import ModelRouter
from vlogcut.schemas import (
    DurationStats,
    Feedback,
    LookStats,
    StructureStats,
    StyleProfile,
    TransitionMix,
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


def make_router(tmp_path: Path, output: str) -> tuple[ModelRouter, FakeClient]:
    client = FakeClient(output)
    config = {
        "providers": {"fake": {"base_url": "http://fake/v1"}},
        "roles": {"feedback": {"provider": "fake", "model": "m-feedback"}},
    }
    router = ModelRouter(config, log_path=tmp_path / "calls.jsonl", client_factory=lambda _: client)
    return router, client


@pytest.fixture
def profile() -> StyleProfile:
    return StyleProfile(
        version=2,
        source_refs=["reference.mp4"],
        shot_duration_s=DurationStats(p10=1.5, median=3, p90=7.5),
        cuts_per_min=18,
        transitions=TransitionMix(hard_cut=0.75, dissolve=0.25),
        structure=StructureStats(montage_share=0.3, intro_pace_ratio=0.8),
        cold_open_clips=2,
        beat_aligned_ratio=0.5,
        bpm_range=(100, 120),
        intro_s=12,
        speech_ratio=0.4,
        look=LookStats(brightness=0.6, saturation=0.2, warmth=0.1, frames=4),
        notes="Original notes. ",
    )


@pytest.mark.parametrize("share", [None, 0.0, 0.25])
@pytest.mark.parametrize("with_profile", [False, True])
def test_messages(profile: StyleProfile, share: float | None, with_profile: bool) -> None:
    profile.transitions = (
        None if share is None else TransitionMix(hard_cut=1 - share, dissolve=share)
    )
    supplied = profile if with_profile else None
    feedback = Feedback(variant_id="v1", verdict="reject", text="too fast overall")
    before = profile.model_dump(), feedback.model_dump()
    messages = build_feedback_messages(feedback, supplied)
    assert messages == build_feedback_messages(feedback, supplied)
    assert [m["role"] for m in messages] == ["system", "user"]
    system, user = [m["content"] for m in messages]
    for text in ("the café part was too short", "too fast overall", "[0.25, 4.0]", "[0, 1]"):
        assert text in system
    assert "Return only one JSON object" in system
    assert 'Verdict: "reject"' in user and 'Text: "too fast overall"' in user
    if with_profile:
        for text in ("p10=1.5 s", "median=3.0 s", "p90=7.5 s"):
            assert text in user
        expected = "Dissolve share: not measured." if share is None else f"Dissolve share={share}."
        assert expected in user
    else:
        assert "No style profile." in user
    assert (profile.model_dump(), feedback.model_dump()) == before
    text_only = Feedback(variant_id="v1", text="the café part was too short")
    assert "Verdict: null" in build_feedback_messages(text_only, None)[1]["content"]


@pytest.mark.parametrize(
    "wrapper", [" {} \n", "```json\n{}\n```", "```\r\n{}\r\n```", "```JSON\n{}\n```"]
)
@pytest.mark.parametrize(
    "data",
    [
        {"scope": "edit", "constraint": "Lengthen the café part", "delta": None},
        {"scope": "style", "constraint": None, "delta": {"pace_factor": 2, "dissolve_share": 0}},
    ],
)
def test_parse(data: dict[str, Any], wrapper: str) -> None:
    assert parse_classification(
        wrapper.format(json.dumps(data))
    ) == FeedbackClassification.model_validate(data)


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON\nat all",
        'Here is JSON: {"scope":"edit","constraint":"longer"}',
        '{"scope":"other"}',
        '{"scope":"edit","constraint":" \\n "}',
        '{"scope":"edit"}',
        '{"scope":"edit","constraint":"longer","delta":{"pace_factor":2}}',
        '{"scope":"style","constraint":"longer","delta":{"pace_factor":2}}',
        '{"scope":"style"}',
        '{"scope":"style","delta":{"pace_factor":0}}',
        '{"scope":"style","delta":{"pace_factor":0.24}}',
        '{"scope":"style","delta":{"pace_factor":4.01}}',
        '{"scope":"style","delta":{"dissolve_share":-0.1}}',
        '{"scope":"style","delta":{"dissolve_share":1.1}}',
        '{"scope":"style","delta":{}}',
        '{"scope":"style","delta":{"pace_factor":null,"dissolve_share":null}}',
    ],
)
def test_invalid_reply_has_one_line_error(raw: str) -> None:
    with pytest.raises(FeedbackClassificationError) as exc:
        parse_classification(raw)
    assert len(str(exc.value).splitlines()) == 1


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_inclusive_delta_bounds(factor: float) -> None:
    assert ProfileDelta(pace_factor=factor).pace_factor == factor


@pytest.mark.parametrize("valid", [False, True])
def test_classify_calls_feedback_once(tmp_path: Path, profile: StyleProfile, valid: bool) -> None:
    raw = '{"scope":"style","delta":{"pace_factor":2}}' if valid else "garbage"
    router, client = make_router(tmp_path, raw)
    feedback = Feedback(variant_id="v1", text="too fast overall")
    if valid:
        assert classify_feedback(router, feedback, profile) == parse_classification(raw)
    else:
        with pytest.raises(FeedbackClassificationError):
            classify_feedback(router, feedback, profile)
    assert len(client.requests) == 1
    assert client.requests[0]["model"] == "m-feedback"
    assert client.requests[0]["messages"] == build_feedback_messages(feedback, profile)
    record = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert record["role"] == "feedback"


@pytest.mark.parametrize("text", ["", " \t\r\n"])
def test_blank_feedback_never_calls(tmp_path: Path, text: str) -> None:
    router, client = make_router(tmp_path, "unused")
    with pytest.raises(FeedbackClassificationError, match="blank"):
        classify_feedback(router, Feedback(variant_id="v1", verdict="pick", text=text))
    assert client.requests == []
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize(
    ("factor", "share"),
    [(2.0, None), (None, 0.0), (None, 0.4), (None, 1.0), (2.0, 0.0), (2.0, 0.4), (2.0, 1.0)],
)
def test_apply_delta(profile: StyleProfile, factor: float | None, share: float | None) -> None:
    before = profile.model_dump()
    delta = ProfileDelta(pace_factor=factor, dissolve_share=share)
    result = apply_profile_delta(profile, delta)
    assert result == apply_profile_delta(profile, delta)
    assert result is not profile
    assert result.shot_duration_s == DurationStats(
        p10=1.5 * (factor or 1), median=3 * (factor or 1), p90=7.5 * (factor or 1)
    )
    assert result.cuts_per_min == 18 / (factor or 1)
    assert result.transitions == (
        profile.transitions if share is None else TransitionMix(hard_cut=1 - share, dissolve=share)
    )
    changed = {"shot_duration_s", "cuts_per_min", "transitions", "notes"}
    assert result.model_dump(exclude=changed) == profile.model_dump(exclude=changed)
    assert result.notes.startswith(profile.notes + "\nApplied profile delta: ")
    assert StyleProfile.model_validate_json(result.model_dump_json()) == result
    result.source_refs.append("new.mp4")
    assert result.look is not None
    result.look.brightness = 0
    assert result.structure is not None
    result.structure.montage_share = 0
    assert profile.model_dump() == before


def test_apply_to_unmeasured_profile_without_notes(profile: StyleProfile) -> None:
    profile.transitions = None
    profile.notes = ""
    result = apply_profile_delta(profile, ProfileDelta(dissolve_share=0))
    assert result.transitions == TransitionMix(hard_cut=1, dissolve=0)
    assert result.notes == "Applied profile delta: dissolve_share=0."
