from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from vlogcut import cli
from vlogcut.schemas import PreferencePair


@pytest.mark.parametrize("default_output", [False, True])
def test_page_relative_previews(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    default_output: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    previews = Path("previews")
    previews.mkdir()
    names = ["preview_coverage_180s_fast.mp4", "preview_slow cut.mp4"]
    for name in names:
        (previews / name).touch()
    argv = ["feedback", "page", *(str(previews / name) for name in names)]
    output = Path("variants.html")
    prefix = ""
    if not default_output:
        Path("pages").mkdir()
        output = Path("pages") / "comparison.html"
        argv += ["-o", str(output)]
        prefix = "../"
    assert cli.main(argv) == 0
    document = output.read_text(encoding="utf-8")
    assert "<h2>coverage_180s_fast</h2>" in document
    assert "<h2>slow cut</h2>" in document
    assert f'src="{prefix}previews/preview_coverage_180s_fast.mp4"' in document
    assert f'src="{prefix}previews/preview_slow%20cut.mp4"' in document
    captured = capsys.readouterr()
    assert str(output) in captured.out and "2 variants" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("duplicate", [False, True])
def test_page_errors_preserve_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], duplicate: bool
) -> None:
    preview = tmp_path / "preview_a.mp4"
    plain = tmp_path / "a.mp4"
    plain.touch()
    if duplicate:
        preview.touch()
    output = tmp_path / "variants.html"
    output.write_text("existing page", encoding="utf-8")
    assert cli.main(["feedback", "page", str(preview), str(plain), "-o", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert ("duplicate variant ID" if duplicate else "preview file not found") in captured.err
    assert output.read_text(encoding="utf-8") == "existing page"


@pytest.fixture
def feedback_json(tmp_path: Path) -> Path:
    path = tmp_path / "feedback.json"
    path.write_text(
        json.dumps([
            {"variant_id": "reject_b", "verdict": "reject"},
            {"variant_id": "pick", "verdict": "pick"},
            {"variant_id": "comment", "text": "Slower, please"},
            {"variant_id": "reject_a", "verdict": "reject"},
        ]),
        encoding="utf-8",
    )  # fmt: skip
    return path


@pytest.mark.parametrize("to_file", [False, True])
def test_pairs_output(
    tmp_path: Path,
    feedback_json: Path,
    capsys: pytest.CaptureFixture[str],
    to_file: bool,
) -> None:
    output = tmp_path / "pairs.json"
    argv = ["feedback", "pairs", str(feedback_json)]
    context = "A quiet café" if to_file else ""
    if to_file:
        output.write_text("old output", encoding="utf-8")
        argv += ["--context", context, "-o", str(output)]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    document = output.read_text(encoding="utf-8") if to_file else captured.out
    pairs = TypeAdapter(list[PreferencePair]).validate_json(document)
    assert pairs == [
        PreferencePair(context=context, chosen="pick", rejected="reject_b"),
        PreferencePair(context=context, chosen="pick", rejected="reject_a"),
    ]
    assert json.loads(document) == [pair.model_dump(mode="json") for pair in pairs]
    if to_file:
        assert str(output) in captured.out and "2 pairs" in captured.out
        assert captured.err == ""
        assert document == json.dumps([pair.model_dump(mode="json") for pair in pairs], indent=1)
        assert set(tmp_path.iterdir()) == {feedback_json, output}
    else:
        assert captured.err == "pairs     2 pairs\n"


@pytest.mark.parametrize(
    "content, message",
    [
        (None, "feedback.json"),
        ("{", "invalid feedback"),
        ('[{"variant_id": "a"}]', "invalid feedback"),
        ("{}", "invalid feedback"),
        (
            '[{"variant_id":"a","verdict":"pick"},{"variant_id":"a","verdict":"reject"}]',
            "conflicting verdicts",
        ),
    ],
)
def test_pairs_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str | None,
    message: str,
) -> None:
    path = tmp_path / "feedback.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert cli.main(["feedback", "pairs", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("vlogcut: error: ")
    assert captured.err.count("\n") == 1 and message in captured.err


@pytest.mark.parametrize("to_file", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '[{"variant_id":"a","text":"Slower"}]',
        '[{"variant_id":"a","verdict":"pick"}]',
        '[{"variant_id":"a","verdict":"reject"}]',
    ],
)
def test_pairs_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, to_file: bool
) -> None:
    path = tmp_path / "feedback.json"
    path.write_text(content, encoding="utf-8")
    output = tmp_path / "pairs.json"
    argv = ["feedback", "pairs", str(path)]
    if to_file:
        argv += ["-o", str(output)]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    if to_file:
        assert output.read_text(encoding="utf-8") == "[]"
        assert "0 pairs" in captured.out and captured.err == ""
    else:
        assert captured.out == "[]\n" and "0 pairs" in captured.err


def test_pairs_failed_replace_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    feedback_json: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "pairs.json"
    output.write_text("existing pairs", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> Path:
        assert target == output and source != output
        assert source.parent == output.parent
        assert len(TypeAdapter(list[PreferencePair]).validate_json(source.read_bytes())) == 2
        assert output.read_text(encoding="utf-8") == "existing pairs"
        raise OSError("cannot replace\ndestination")

    monkeypatch.setattr(Path, "replace", fail_replace)
    assert cli.main(["feedback", "pairs", str(feedback_json), "-o", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "vlogcut: error: cannot replace destination\n"
    assert output.read_text(encoding="utf-8") == "existing pairs"
    assert set(tmp_path.iterdir()) == {feedback_json, output}
