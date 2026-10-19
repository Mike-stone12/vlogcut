from hashlib import sha256

import pytest

from vlogcut.run import run_id
from vlogcut.schemas import RunInputs


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
