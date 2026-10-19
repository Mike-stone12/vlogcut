from hashlib import sha256
from pathlib import Path

import pytest

from vlogcut import cli
from vlogcut.run import format_run, load_run, run_id
from vlogcut.schemas import Run, RunInputs


@pytest.fixture
def manifest() -> Run:
    inputs = RunInputs(
        asset_ids=["clip-b", "clip-a"],
        target_s=180,
        strategy="highlights",
        planner="llm",
        planner_model="test-planner",
        profile_sha256="a" * 64,
        music_sha256="b" * 64,
        ranker_sha256="c" * 64,
        brief="A day at the café\nKeep it cozy",
        constraints=["Keep the arrival", "End at home"],
        cold_open=True,
        caption=True,
    )
    return Run(
        run_id=run_id(inputs),
        inputs=inputs,
        variant_id="highlights_180s",
        edl="edl/highlights_180s.json",
        violations="edl/highlights_180s.violations.json",
        preview="preview_highlights_180s.mp4",
    )


def test_load_run_round_trip(tmp_path: Path, manifest: Run) -> None:
    path = tmp_path / "manifest.json"
    original = manifest.model_dump_json(indent=1).encode("utf-8")
    path.write_bytes(original)
    assert load_run(path) == manifest
    assert load_run(str(path)) == manifest
    assert path.read_bytes() == original


def test_format_run_all_fields(manifest: Run) -> None:
    original = manifest.model_dump_json()
    expected = [
        f'run_id (inputs fingerprint): "{manifest.run_id}"',
        "inputs:",
        '  asset_ids: ["clip-a", "clip-b"]',
        "  target_s: 180.0",
        '  strategy: "highlights"',
        '  planner: "llm"',
        '  planner_model: "test-planner"',
        f'  profile_sha256: "{"a" * 64}"',
        f'  music_sha256: "{"b" * 64}"',
        f'  ranker_sha256: "{"c" * 64}"',
        '  brief: "A day at the café\\nKeep it cozy"',
        '  constraints: ["Keep the arrival", "End at home"]',
        "  cold_open: true",
        "  caption: true",
        'variant_id: "highlights_180s"',
        'edl: "edl/highlights_180s.json"',
        'violations: "edl/highlights_180s.violations.json"',
        'preview: "preview_highlights_180s.mp4"',
    ]
    assert format_run(manifest) == "\n".join(expected)
    assert format_run(Run.model_validate_json(original)) == format_run(manifest)
    assert manifest.model_dump_json() == original


def test_format_run_optional_values(manifest: Run) -> None:
    inputs = RunInputs(asset_ids=[], target_s=3, strategy="coverage", planner="rules")
    run = manifest.model_copy(update={"inputs": inputs, "preview": None})
    summary = format_run(run)
    for field in ("planner_model", "profile_sha256", "music_sha256", "ranker_sha256", "preview"):
        assert f"{field}: null" in summary
    assert 'brief: ""' in summary
    assert "constraints: []" in summary
    assert "cold_open: false" in summary
    assert "caption: false" in summary


@pytest.mark.parametrize("payload", [None, b"{", b"{}", b"null", b"\xff"])
def test_load_run_errors_name_path(tmp_path: Path, payload: bytes | None) -> None:
    path = tmp_path / "manifest.json"
    if payload is not None:
        path.write_bytes(payload)
    with pytest.raises(ValueError) as exc:
        load_run(path)
    message = str(exc.value)
    assert repr(str(path)) in message
    assert message.startswith("cannot load run manifest")
    assert len(message.splitlines()) == 1


def test_run_show(tmp_path: Path, manifest: Run, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    assert cli.main(["run", "show", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == format_run(manifest) + "\n"
    assert captured.err == ""


@pytest.mark.parametrize("payload", [None, b"{", b"{}", b"\xff"])
def test_run_show_errors(
    tmp_path: Path, payload: bytes | None, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "manifest.json"
    if payload is not None:
        path.write_bytes(payload)
    assert cli.main(["run", "show", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("vlogcut: error: cannot load run manifest")
    assert repr(str(path)) in captured.err
    assert len(captured.err.splitlines()) == 1


def test_run_show_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "show", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "MANIFEST_JSON" in help_text
    assert "<workspace>/runs/<run_id>.json" in help_text
    assert "not a directory" in help_text


def test_run_id_is_exact_deterministic_fingerprint() -> None:
    inputs = RunInputs(asset_ids=["b", "a"], target_s=3, strategy="coverage", planner="rules")
    expected = sha256(inputs.model_dump_json().encode("utf-8")).hexdigest()[:16]
    assert run_id(inputs) == expected
    assert len(expected) == 16 and int(expected, 16) >= 0
    assert run_id(RunInputs.model_validate_json(inputs.model_dump_json())) == expected
    assert run_id(inputs.model_copy(update={"asset_ids": ["a", "b"]})) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("asset_ids", ["a", "c"]),
        ("target_s", 4),
        ("strategy", "highlights"),
        ("planner", "llm"),
        ("planner_model", "test-planner"),
        ("profile_sha256", "a" * 64),
        ("music_sha256", "b" * 64),
        ("ranker_sha256", "c" * 64),
        ("brief", "A day at the café"),
        ("constraints", ["Keep the arrival"]),
        ("cold_open", True),
        ("caption", True),
    ],
)
def test_run_id_changes_with_each_input(field: str, value: object) -> None:
    inputs = RunInputs(asset_ids=["a", "b"], target_s=3, strategy="coverage", planner="rules")
    changed = RunInputs.model_validate({**inputs.model_dump(), field: value})
    assert run_id(changed) != run_id(inputs)
