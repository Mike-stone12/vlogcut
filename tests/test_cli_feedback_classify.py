"""Offline CLI classification through ModelRouter with a scripted provider."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlogcut import cli
from vlogcut.llm.router import ModelRouter
from vlogcut.schemas import DurationStats, StyleProfile
from vlogcut.style import load_profile


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.outputs.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def stub_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outputs: list[str],
    config_path: Path = Path("config.yaml"),
) -> FakeClient:
    client = FakeClient(outputs)
    router = ModelRouter(
        {
            "providers": {"fake": {"base_url": "http://fake/v1"}},
            "roles": {"feedback": {"provider": "fake", "model": "m-feedback"}},
        },
        log_path=tmp_path / "calls.jsonl",
        client_factory=lambda _: client,
    )

    def load(path: Path) -> ModelRouter:
        assert path == config_path
        return router

    monkeypatch.setattr(ModelRouter, "from_yaml", load)
    return client


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    feedback = tmp_path / "feedback.json"
    feedback.write_text('[{"variant_id":"v1","text":"too fast overall"}]', encoding="utf-8")
    profile = tmp_path / "style.v1.json"
    profile.write_text(
        StyleProfile(
            shot_duration_s=DurationStats(p10=1, median=2, p90=4),
            cuts_per_min=20,
            notes="Original",
        ).model_dump_json(),
        encoding="utf-8",
    )
    return feedback, profile


def command(inputs: tuple[Path, Path]) -> list[str]:
    feedback, profile = inputs
    return ["feedback", "classify", str(feedback), "--profile", str(profile)]


@pytest.mark.parametrize("destination", ["default", "custom", "input"])
def test_delta_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    destination: str,
) -> None:
    _, profile = inputs
    before = profile.read_bytes()
    config = tmp_path / "models.yaml"
    client = stub_router(
        tmp_path, monkeypatch, ['{"scope":"style","delta":{"pace_factor":2}}'], config
    )
    out = profile.with_name("style.v1.updated.json")
    argv = command(inputs) + ["--config", str(config)]
    if destination != "default":
        out = profile if destination == "input" else tmp_path / "custom.json"
        argv += ["-o", str(out)]
    assert cli.main(argv) == 0
    result = load_profile(out)
    assert result.shot_duration_s == DurationStats(p10=2, median=4, p90=8)
    assert result.cuts_per_min == 10
    assert "Applied profile delta: pace_factor=2." in result.notes
    if destination != "input":
        assert profile.read_bytes() == before
    captured = capsys.readouterr()
    assert "classified 1 records, applied 1 deltas, 0 constraints" in captured.out
    assert captured.err == ""
    assert len(client.requests) == 1 and client.requests[0]["model"] == "m-feedback"
    assert json.loads((tmp_path / "calls.jsonl").read_text())["role"] == "feedback"


@pytest.mark.parametrize("write_constraints", [False, True])
def test_mixed_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    write_constraints: bool,
) -> None:
    feedback, profile = inputs
    feedback.write_text(
        json.dumps(
            [
                {"variant_id": "v1", "text": "too fast overall"},
                {"variant_id": "v2", "verdict": "pick", "text": " \t\n"},
                {"variant_id": "v3", "text": "the café part was too short"},
                {"variant_id": "v4", "text": "more dissolves"},
            ]
        ),
        encoding="utf-8",
    )
    constraint = "Lengthen the café part"
    client = stub_router(
        tmp_path,
        monkeypatch,
        [
            '{"scope":"style","delta":{"pace_factor":2}}',
            json.dumps({"scope": "edit", "constraint": constraint}),
            '{"scope":"style","delta":{"dissolve_share":0.4}}',
        ],
    )
    argv = command(inputs)
    constraints = tmp_path / "constraints.json"
    if write_constraints:
        argv += ["--constraints-out", str(constraints)]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert "classified 3 records, applied 2 deltas, 1 constraints" in captured.out
    assert captured.err == ""
    if write_constraints:
        assert json.loads(constraints.read_text()) == [constraint]
        assert constraint not in captured.out
    else:
        assert constraint in captured.out.splitlines()
    assert len(client.requests) == 3
    assert all("median=2.0 s" in req["messages"][1]["content"] for req in client.requests)
    result = load_profile(profile.with_name("style.v1.updated.json"))
    assert result.shot_duration_s.median == 4
    assert result.transitions is not None and result.transitions.dissolve == 0.4


@pytest.mark.parametrize("records", [[], [{"variant_id": "v1", "verdict": "pick"}]])
def test_no_text_never_calls_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    records: list[dict[str, str]],
) -> None:
    feedback, profile = inputs
    feedback.write_text(json.dumps(records), encoding="utf-8")
    client = stub_router(tmp_path, monkeypatch, [])
    assert cli.main(command(inputs)) == 0
    assert client.requests == []
    assert load_profile(profile.with_name("style.v1.updated.json")) == load_profile(profile)
    assert "classified 0 records, applied 0 deltas, 0 constraints" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["feedback", "profile"])
@pytest.mark.parametrize("content", [None, "{", "{}", "[1]", "\xff", "directory"])
def test_bad_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    target: str,
    content: str | None,
) -> None:
    path = inputs[0 if target == "feedback" else 1]
    if content == "directory":
        monkeypatch.chdir(path.parent)
        inputs = (Path("."), inputs[1]) if target == "feedback" else (inputs[0], Path("."))
    elif content is None:
        path.unlink()
    else:
        path.write_bytes(content.encode("latin-1"))
    monkeypatch.setattr(ModelRouter, "from_yaml", lambda _: pytest.fail("loaded router"))
    assert cli.main(command(inputs)) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert captured.err.startswith("vlogcut: error:")


@pytest.mark.parametrize("content", [None, "[", "", "[]", "providers: {}\nroles: {}"])
def test_bad_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    content: str | None,
) -> None:
    config = tmp_path / "models.yaml"
    if content is not None:
        config.write_text(content, encoding="utf-8")
    assert cli.main(command(inputs) + ["--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert "invalid model config" in captured.err


@pytest.mark.parametrize("reply", ["not JSON\nat all", '{"scope":"style","delta":{}}'])
def test_classifier_error_preserves_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    reply: str,
) -> None:
    before = inputs[1].read_bytes()
    client = stub_router(tmp_path, monkeypatch, [reply])
    assert cli.main(command(inputs) + ["-o", str(inputs[1])]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert "Invalid feedback classification" in captured.err
    assert inputs[1].read_bytes() == before
    assert len(client.requests) == 1


@pytest.mark.parametrize("input_profile", [False, True])
def test_constraints_cannot_overwrite_profiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inputs: tuple[Path, Path],
    input_profile: bool,
) -> None:
    profile = inputs[1]
    before = profile.read_bytes()
    constraints = profile if input_profile else profile.with_name("style.v1.updated.json")
    monkeypatch.setattr(ModelRouter, "from_yaml", lambda _: pytest.fail("loaded router"))
    assert cli.main(command(inputs) + ["--constraints-out", str(constraints)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert "--constraints-out must differ" in captured.err
    assert profile.read_bytes() == before
