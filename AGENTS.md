# Instructions for coding agents

Shared by Claude Code (imported from CLAUDE.md), Codex and any other agent.

## Project

`vlogcut` turns raw daily-vlog footage into editable rough cuts that match a
creator's style learned from reference videos. Read `ARCHITECTURE.md` before any
non-trivial change. `src/vlogcut/schemas.py` defines the data contracts.

## Commands

```bash
pip install -e ".[dev]"   # install with dev tools
pytest -q                 # tests (offline, no GPU, no API keys)
ruff check . && ruff format --check .
```

## Hard rules

- Models never generate FFmpeg/shell commands and never touch media files.
  All model output is JSON validated by pydantic models in `schemas.py`.
- Every model call goes through `vlogcut.llm.router.ModelRouter`.
  No direct SDK calls anywhere else.
- `schemas.py` is the contract. Changing a schema means updating
  `ARCHITECTURE.md` and the affected tests in the same change.
- Tests must run offline: no network, no GPU, no API keys. Use tiny fixtures and
  `FakeClient`-style stubs (see `tests/test_router.py`).
- Never commit media, `workspace/`, `config.yaml`, or anything with keys.

## How to work

- Beyond a small fix: propose a plan first (files to touch, approach, tests),
  then wait for approval.
- One module per change; keep diffs reviewable (roughly < 400 lines).
- Every change adds or updates tests. Run `pytest` and `ruff` before saying done.
- If a design choice is unclear, ask instead of guessing. Record decisions in the
  "Decisions log" section of `ARCHITECTURE.md`.

## Code style

- Python 3.11+, type hints everywhere, pydantic v2, `pathlib`
- `logging`, not `print`; no global mutable state
- Analysis and style code: small pure functions over arrays/records, easy to unit-test
- Heavy dependencies (torch, model weights) are imported lazily inside the
  functions that need them, so the core stays importable without them
