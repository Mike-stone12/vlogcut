from pathlib import Path

import pytest
from pydantic import TypeAdapter

from vlogcut import cli, pipeline, plan
from vlogcut.learn.ranker import RankerModel, score, train_pairwise
from vlogcut.learn.train_cli import RankerTrainingError, train_from_feedback
from vlogcut.schemas import EDL, Clip, Feedback, MediaAsset, QualityFlags, Shot

TrainingData = tuple[list[Feedback], dict[str, EDL], list[Shot]]


@pytest.fixture
def training_data() -> TrainingData:
    shots = [
        Shot(
            shot_id=sid, asset_id="asset", start_s=0, end_s=4, interest=interest,
            quality=QualityFlags(sharpness=100, exposure_clip=0, shake=0),
        )
        for sid, interest in [("chosen", 5), ("skipped", 1), ("shared", 3)]
    ]  # fmt: skip
    edls = {
        name: EDL(
            variant_id="internal-label", target_duration_s=2,
            clips=[Clip(shot_id=sid, in_s=0, out_s=1) for sid in (shot.shot_id, "shared")],
        )
        for name, shot in zip(("coverage_2s", "highlights_2s"), shots[:2], strict=True)
    }  # fmt: skip
    feedback = [
        Feedback(variant_id="coverage_2s", verdict="pick"),
        Feedback(variant_id="highlights_2s", verdict="reject"),
        Feedback(variant_id="coverage_2s", verdict="pick"),
        Feedback(variant_id="comment", text="Slower please"),
    ]
    return feedback, edls, shots


def test_train_and_json_round_trip(training_data: TrainingData) -> None:
    feedback, edls, shots = training_data
    model, count = train_from_feedback(feedback, edls, shots, context="A quiet day")
    assert count == 1
    assert model == train_pairwise([(shots[0], shots[1])])
    restored = RankerModel.model_validate_json(model.model_dump_json())
    assert restored == model
    assert score(restored, shots[0]) > score(restored, shots[1])


@pytest.mark.parametrize("reason", ["empty", "shared", "unknown", "rejected-pick", "rejected-skip"])
def test_zero_pairs(training_data: TrainingData, reason: str) -> None:
    feedback, edls, shots = training_data
    if reason == "empty":
        feedback = []
    elif reason == "shared":
        edls["highlights_2s"] = edls["coverage_2s"]
    elif reason == "unknown":
        shots = shots[2:]
    else:
        shots[0 if reason == "rejected-pick" else 1].quality.rejected = True
    with pytest.raises(RankerTrainingError, match="no training shot pairs"):
        train_from_feedback(feedback, edls, shots)


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_data: TrainingData
) -> tuple[Path, Path]:
    feedback, edls, shots = training_data
    asset = MediaAsset(
        asset_id="asset", path="asset.mp4", duration_s=4, fps=30, width=320, height=240
    )
    monkeypatch.setattr(pipeline, "find_footage", lambda folder: [Path(asset.path)])
    monkeypatch.setattr(pipeline, "ingest", lambda *args: ([asset], []))
    monkeypatch.setattr(pipeline, "analyze", lambda *args: shots)
    monkeypatch.setattr(
        plan, "rough_cut", lambda s, a, params, profile, **kw: edls[f"{params.strategy}_2s"]
    )
    ws = tmp_path / "workspace"
    for strategy in ("coverage", "highlights"):
        pipeline.cut(tmp_path, ws, strategy=strategy, target_s=2, render=False)
    path = tmp_path / "feedback.json"
    path.write_bytes(TypeAdapter(list[Feedback]).dump_json(feedback))
    return ws, path


@pytest.mark.parametrize("default_output", [False, True])
def test_cli_success(
    workspace: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    training_data: TrainingData,
    default_output: bool,
) -> None:
    ws, feedback = workspace
    monkeypatch.chdir(tmp_path)
    # Even an EDL-shaped violation report must not be treated as a variant.
    for name, content in [
        ("extra.violations.json", training_data[1]["coverage_2s"].model_dump_json()),
        ("notes.json", "{}"),
        ("broken.json", "{"),
        ("notes.txt", "notes"),
    ]:
        (ws / "edl" / name).write_text(content, encoding="utf-8")
    (ws / "edl" / "directory.json").mkdir()
    output = Path("ranker.json") if default_output else tmp_path / "custom.json"
    output.write_text("old model", encoding="utf-8")
    original_replace = Path.replace
    replacements = []

    def replace(source: Path, target: Path) -> Path:
        assert source.suffix == ".part" and source.parent.resolve() == target.parent.resolve()
        RankerModel.model_validate_json(source.read_bytes())
        assert target.read_text(encoding="utf-8") == "old model"
        replacements.append(target)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    argv = ["ranker", "train", str(feedback), "-w", str(ws), "--context", "A quiet day", "-v"]
    if not default_output:
        argv += ["-o", str(output)]
    assert cli.main(argv) == 0
    model = RankerModel.model_validate_json(output.read_text(encoding="utf-8"))
    assert score(model, training_data[2][0]) > score(model, training_data[2][1])
    assert replacements == [output]
    assert not list(tmp_path.glob("*.part"))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        f"ranker    4 feedback records, 1 preference pairs, 1 shot pairs -> {output}\n"
    )


@pytest.mark.parametrize(
    "problem, message",
    [
        ("missing-feedback", "feedback.json"),
        ("invalid-feedback", "invalid feedback"),
        ("zero-pairs", "no training shot pairs"),
        ("missing-workspace", "workspace not found"),
        ("missing-edls", "no EDLs found"),
        ("missing-variant", "missing EDLs"),
        ("missing-shots", "cannot load analyzed shots"),
        ("invalid-shots", "cannot load analyzed shots"),
        ("empty-shots", "no analyzed shots"),
        ("write-failure", "cannot replace"),
    ],
)
def test_cli_errors(
    workspace: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    problem: str,
    message: str,
) -> None:
    ws, feedback = workspace
    if problem == "missing-feedback":
        feedback.unlink()
    elif problem in {"invalid-feedback", "zero-pairs"}:
        feedback.write_text("{" if problem == "invalid-feedback" else "[]", encoding="utf-8")
    elif problem == "missing-workspace":
        ws = tmp_path / "absent"
    elif problem in {"missing-edls", "missing-variant"}:
        (ws / "edl" / "coverage_2s.json").unlink()
        if problem == "missing-edls":
            (ws / "edl" / "highlights_2s.json").unlink()
    elif problem == "missing-shots":
        (ws / "shots.json").unlink()
    elif problem in {"invalid-shots", "empty-shots"}:
        (ws / "shots.json").write_text(
            "{}" if problem == "invalid-shots" else "[]", encoding="utf-8"
        )
    else:

        def fail_replace(source: Path, target: Path) -> Path:
            raise OSError("cannot replace\noutput")

        monkeypatch.setattr(Path, "replace", fail_replace)
    output = tmp_path / "ranker.json"
    output.write_text("old model", encoding="utf-8")
    assert cli.main(["ranker", "train", str(feedback), "-w", str(ws), "-o", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("vlogcut: error: ")
    assert len(captured.err.splitlines()) == 1 and message in captured.err
    assert output.read_text(encoding="utf-8") == "old model"
    assert not list(tmp_path.glob("*.part"))
