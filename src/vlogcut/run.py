"""Deterministic run identity and manifest inspection, independent of media services."""

import json
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from vlogcut.schemas import Run, RunInputs


def run_id(inputs: RunInputs) -> str:
    """First 16 hex characters of SHA-256 over compact model JSON in UTF-8."""
    return sha256(inputs.model_dump_json().encode("utf-8")).hexdigest()[:16]


def load_run(path: str | Path) -> Run:
    """Read a manifest file; I/O, JSON and schema errors become one-line ValueErrors."""
    path = Path(path)
    try:
        return Run.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        detail = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
        detail = " ".join(detail.split())
        raise ValueError(f"cannot load run manifest {str(path)!r}: {detail}") from exc


def format_run(run: Run) -> str:
    """Summarize all fields in schema order, preserving stored paths and input values."""
    lines: list[str] = []
    for name, value in run.model_dump(mode="json").items():
        if name == "inputs":
            lines.append("inputs:")
            lines.extend(
                f"  {key}: {json.dumps(item, ensure_ascii=False)}" for key, item in value.items()
            )
        else:
            label = "run_id (inputs fingerprint)" if name == "run_id" else name
            lines.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)
