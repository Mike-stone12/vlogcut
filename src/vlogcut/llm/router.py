"""The only place in the codebase that talks to models.

Calls are addressed by *role* (captioner, planner, feedback, critic), not by
vendor. Each role maps to a provider + model in config.yaml. Any endpoint that
speaks the OpenAI chat-completions protocol works: local vLLM/Ollama, OpenAI,
DeepSeek, Qwen (DashScope), OpenRouter, ...

`structured()` is the main entry point: it asks for JSON, validates it with a
pydantic model, runs an optional semantic check, and feeds any errors back to
the model for a bounded number of repair rounds.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
JsonMode = Literal["schema", "object", "none"]
# A semantic check returns a list of human-readable problems; empty = OK.
Check = Callable[[Any], list[str]]


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str | None = None  # name of the env var holding the key


@dataclass(frozen=True)
class RoleConfig:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    # "schema": send the pydantic JSON schema (best, if the provider supports it)
    # "object": generic JSON mode; "none": rely on the prompt only
    json_mode: JsonMode = "schema"


class StructuredOutputError(RuntimeError):
    def __init__(self, role: str, errors: list[str], last_output: str):
        super().__init__(f"[{role}] no valid output after repairs: {errors[:3]}")
        self.role = role
        self.errors = errors
        self.last_output = last_output


class ModelRouter:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        log_path: str | Path | None = None,
        client_factory: Callable[[ProviderConfig], Any] | None = None,
    ):
        self.providers = {k: ProviderConfig(**v) for k, v in config["providers"].items()}
        self.roles = {k: RoleConfig(**v) for k, v in config["roles"].items()}
        for name, role in self.roles.items():
            if role.provider not in self.providers:
                raise ValueError(f"role '{name}' uses unknown provider '{role.provider}'")
        self.log_path = Path(log_path or config.get("log_path", "workspace/llm_calls.jsonl"))
        self._client_factory = client_factory or _openai_client
        self._clients: dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> ModelRouter:
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f), **kwargs)

    # ------------------------------------------------------------------ calls

    def complete(self, role: str, messages: list[dict[str, Any]]) -> str:
        """Plain text completion for a role."""
        return self._call(role, messages, json_mode="none", schema=None)

    def structured(
        self,
        role: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        check: Check | None = None,
        max_repairs: int = 2,
    ) -> T:
        """Return a validated `schema` instance, repairing up to `max_repairs` times.

        `check` runs after pydantic validation for rules that need outside context
        (e.g. "every shot_id must exist in the shot table").
        """
        cfg = self._role(role)
        convo = list(messages)
        errors: list[str] = []
        raw = ""
        for attempt in range(max_repairs + 1):
            raw = self._call(role, convo, json_mode=cfg.json_mode, schema=schema)
            result, errors = _parse_and_validate(raw, schema, check)
            if result is not None:
                if attempt:
                    log.info("[%s] valid output after %d repair(s)", role, attempt)
                return result
            log.warning("[%s] attempt %d invalid: %s", role, attempt + 1, errors[:3])
            convo = [
                *convo,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _repair_prompt(errors)},
            ]
        raise StructuredOutputError(role, errors, raw)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def image_part(path: str | Path, detail: str = "low") -> dict[str, Any]:
        """Message content part for one image (OpenAI vision format)."""
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        data = base64.b64encode(Path(path).read_bytes()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}", "detail": detail},
        }

    def _role(self, role: str) -> RoleConfig:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"unknown role '{role}'; configured: {sorted(self.roles)}") from None

    def _client(self, provider: str) -> Any:
        if provider not in self._clients:
            self._clients[provider] = self._client_factory(self.providers[provider])
        return self._clients[provider]

    def _call(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: JsonMode,
        schema: type[BaseModel] | None,
    ) -> str:
        cfg = self._role(role)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if json_mode == "schema" and schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        elif json_mode == "object":
            kwargs["response_format"] = {"type": "json_object"}

        call_id, t0 = uuid.uuid4().hex[:12], time.monotonic()
        text, usage, error = "", None, None
        try:
            resp = self._client(cfg.provider).chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return text
        except Exception as exc:  # logged, then re-raised
            error = repr(exc)
            raise
        finally:
            self._log(
                {
                    "id": call_id,
                    "ts": time.time(),
                    "role": role,
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "latency_s": round(time.monotonic() - t0, 3),
                    "usage": _usage_dict(usage),
                    "messages": _redact_images(messages),
                    "output": text,
                    "error": error,
                }
            )

    def _log(self, record: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            log.exception("could not write LLM call log")


# ---------------------------------------------------------------------------
# module-level helpers (pure, unit-tested)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(raw: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    text = _FENCE.sub("", raw.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _parse_and_validate(
    raw: str, schema: type[T], check: Check | None
) -> tuple[T | None, list[str]]:
    try:
        data = extract_json(raw)
    except json.JSONDecodeError as exc:
        return None, [f"output is not valid JSON: {exc}"]
    try:
        obj = schema.model_validate(data)
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        ]
    problems = check(obj) if check else []
    return (None, problems) if problems else (obj, [])


def _repair_prompt(errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "Your previous output was rejected for these reasons:\n"
        f"{bullet_list}\n\n"
        "Return the complete corrected JSON only, with no explanation."
    )


def _redact_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [
                {"type": "image_url", "image_url": "<image omitted>"}
                if part.get("type") == "image_url"
                else part
                for part in content
            ]
        out.append({**m, "content": content})
    return out


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {"raw": str(usage)}


def _openai_client(provider: ProviderConfig) -> Any:
    from openai import OpenAI  # lazy: tests never need the SDK

    key = "not-needed"  # local servers ignore the key but the SDK requires one
    if provider.api_key_env:
        key = os.environ.get(provider.api_key_env, "")
        if not key:
            raise RuntimeError(f"environment variable {provider.api_key_env} is not set")
    return OpenAI(base_url=provider.base_url, api_key=key)
