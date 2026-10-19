"""Deterministic run identity, independent of media and model services."""

from hashlib import sha256

from vlogcut.schemas import RunInputs


def run_id(inputs: RunInputs) -> str:
    """First 16 hex characters of SHA-256 over compact model JSON in UTF-8."""
    return sha256(inputs.model_dump_json().encode("utf-8")).hexdigest()[:16]
