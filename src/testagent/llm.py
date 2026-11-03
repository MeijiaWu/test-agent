"""LLM access: one OpenAI-compatible client for a local llama.cpp server or a
hosted API, structured (JSON) calls with validation and retry, a response
cache for reproducibility, and per-call usage accounting for tracing.

``ScriptedLLM`` replays canned replies keyed by task name; the tests use it to
exercise the whole graph without a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    task: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    attempts: int = 1


@dataclass
class LLMConfig:
    model: str = "qwen"
    base_url: str = "http://localhost:8080/v1"      # llama.cpp server default
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 600.0
    cache_dir: str | None = ".cache/llm"
    max_retries: int = 2                             # re-asks after an invalid reply
    json_mode: bool = False                          # response_format={"type": "json_object"}


SYSTEM = ("You are a senior software test engineer. Answer with a single JSON object "
          "that matches the requested schema. No prose outside the JSON.")


TEXT_SYSTEM = "You are a senior software engineer. Follow the requested output format exactly."


def extract_json(text: str) -> str:
    """The first balanced top-level JSON object in ``text`` (fences tolerated)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)       # reasoning models
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON object in reply")


def schema_hint(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


class BaseLLM:
    """Subclasses implement ``_complete(messages) -> (text, prompt_tokens, completion_tokens)``."""

    def __init__(self, on_usage: Callable[[Usage], None] | None = None):
        self.on_usage = on_usage
        self.calls: list[Usage] = []

    def _complete(self, messages: list[dict]) -> tuple[str, int, int]:
        raise NotImplementedError

    def _record(self, u: Usage):
        self.calls.append(u)
        if self.on_usage:
            self.on_usage(u)

    def structured(self, task: str, prompt: str, schema: type[T], max_retries: int | None = None) -> T:
        """Ask for JSON matching ``schema``; on an invalid reply, show the model the
        error and ask again (``max_retries`` times)."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TASK: {task}\n\n{prompt}\n\n"
                                        f"Reply with JSON matching this schema:\n{schema_hint(schema)}"},
        ]
        usage = Usage(task=task)
        t0 = time.perf_counter()
        last_err = None
        if max_retries is None:
            max_retries = getattr(getattr(self, "cfg", None), "max_retries", 2)
        for attempt in range(max_retries + 1):
            text, pt, ct, cached = self._complete_cached(messages)
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cached = usage.cached or cached
            usage.attempts = attempt + 1
            try:
                obj = schema.model_validate_json(extract_json(text))
                usage.seconds = time.perf_counter() - t0
                self._record(usage)
                return obj
            except (ValueError, ValidationError) as e:
                last_err = e
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That reply was invalid: {str(e)[:1500]}\n"
                                                f"Reply again with only the corrected JSON."},
                ]
        usage.seconds = time.perf_counter() - t0
        self._record(usage)
        raise LLMOutputError(task, last_err)

    def text(self, task: str, prompt: str, parse: Callable[[str], Any], max_retries: int | None = None):
        """Free-form reply turned into an object by ``parse`` (which raises ValueError on a bad
        reply); for outputs that are mostly code, where JSON escaping is error-prone."""
        messages = [{"role": "system", "content": TEXT_SYSTEM},
                    {"role": "user", "content": f"TASK: {task}\n\n{prompt}"}]
        usage = Usage(task=task)
        t0 = time.perf_counter()
        last_err = None
        if max_retries is None:
            max_retries = getattr(getattr(self, "cfg", None), "max_retries", 2)
        for attempt in range(max_retries + 1):
            text, pt, ct, cached = self._complete_cached(messages)
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cached = usage.cached or cached
            usage.attempts = attempt + 1
            try:
                obj = parse(re.sub(r"<think>.*?</think>", "", text, flags=re.S))
                usage.seconds = time.perf_counter() - t0
                self._record(usage)
                return obj
            except ValueError as e:
                last_err = e
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That reply could not be used: {str(e)[:1500]}\n"
                                                f"Reply again in the requested format."}]
        usage.seconds = time.perf_counter() - t0
        self._record(usage)
        raise LLMOutputError(task, last_err)

    def _complete_cached(self, messages):
        text, pt, ct = self._complete(messages)
        return text, pt, ct, False


class LLMOutputError(RuntimeError):
    def __init__(self, task, err):
        super().__init__(f"{task}: no valid reply after retries: {err}")
        self.task = task


class OpenAICompatLLM(BaseLLM):
    """Works with llama.cpp ``llama-server``, vLLM, Ollama's /v1, or a hosted API."""

    def __init__(self, cfg: LLMConfig, on_usage=None):
        super().__init__(on_usage)
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, timeout=cfg.timeout,
                             api_key=os.environ.get(cfg.api_key_env, "sk-local"))
        self.cache = Path(cfg.cache_dir) if cfg.cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _key(self, messages) -> str:
        blob = json.dumps([self.cfg.model, self.cfg.temperature, self.cfg.max_tokens, messages],
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _complete_cached(self, messages):
        if self.cache is not None:
            f = self.cache / f"{self._key(messages)}.json"
            if f.exists():
                d = json.loads(f.read_text())
                return d["text"], d["prompt_tokens"], d["completion_tokens"], True
        text, pt, ct = self._complete(messages)
        if self.cache is not None:
            f.write_text(json.dumps({"text": text, "prompt_tokens": pt, "completion_tokens": ct},
                                    ensure_ascii=False))
        return text, pt, ct, False

    def _complete(self, messages):
        extra = {"response_format": {"type": "json_object"}} if self.cfg.json_mode else {}
        r = self.client.chat.completions.create(
            model=self.cfg.model, messages=messages, temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens, **extra)
        u = r.usage
        return (r.choices[0].message.content or "",
                getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


@dataclass
class ScriptedLLM(BaseLLM):
    """Replays replies per task name, in order (the last one repeats).

    ``script = {"analyze_issue": [json_str, ...], ...}``; values may also be
    callables ``prompt -> json_str`` for replies that depend on the prompt.
    """

    script: dict = field(default_factory=dict)
    on_usage: Callable | None = None

    def __post_init__(self):
        BaseLLM.__init__(self, self.on_usage)
        self._i: dict[str, int] = {}
        self.prompts: list[tuple[str, str]] = []

    def _complete(self, messages):
        user = messages[1]["content"]
        task = user.split("\n", 1)[0].removeprefix("TASK: ").strip()
        self.prompts.append((task, user))
        replies = self.script[task]
        i = self._i.get(task, 0)
        self._i[task] = i + 1
        reply = replies[min(i, len(replies) - 1)]
        if callable(reply):
            reply = reply(user)
        return reply, len(user) // 4, len(reply) // 4


def llm_health(cfg) -> str | None:
    """None if the OpenAI-compatible server answers ``GET /models``, else what went wrong."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(cfg.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {os.environ.get(cfg.api_key_env, 'sk-local')}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return None if r.status == 200 else f"HTTP {r.status} from {cfg.base_url}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} from {cfg.base_url}"
    except Exception as e:                       # noqa: BLE001
        return f"{cfg.base_url} unreachable: {e}"

