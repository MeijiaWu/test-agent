"""Run tracing: every node, LLM call and test execution as one JSON event.

Events go to an optional JSONL file and to optional subscribers (the API
streams them to clients). ``summary()`` aggregates what the evaluation
reports per run: tokens, LLM time, test executions, wall time.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable


class Tracer:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.subscribers: list[Callable[[dict], None]] = []
        self.t0 = time.time()
        self._lock = threading.Lock()

    def emit(self, type_: str, **data):
        ev = {"t": round(time.time() - self.t0, 3), "type": type_, **data}
        with self._lock:
            self.events.append(ev)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        for s in list(self.subscribers):
            s(ev)

    @contextmanager
    def span(self, node: str):
        self.emit("node_start", node=node)
        t = time.perf_counter()
        try:
            yield
            self.emit("node_end", node=node, seconds=round(time.perf_counter() - t, 3))
        except Exception as e:
            self.emit("node_error", node=node, error=f"{type(e).__name__}: {e}")
            raise

    def on_llm(self, usage):
        self.emit("llm", **asdict(usage))

    def summary(self) -> dict:
        llm = [e for e in self.events if e["type"] == "llm"]
        ex = [e for e in self.events if e["type"] == "exec"]
        return {
            "llm_calls": len(llm),
            "prompt_tokens": sum(e["prompt_tokens"] for e in llm),
            "completion_tokens": sum(e["completion_tokens"] for e in llm),
            "llm_seconds": round(sum(e["seconds"] for e in llm), 2),
            "llm_retries": sum(e["attempts"] - 1 for e in llm),
            "test_runs": len(ex),
            "exec_seconds": round(sum(e.get("seconds", 0) for e in ex), 2),
            "wall_seconds": round(time.time() - self.t0, 2),
            "per_task_tokens": _group(llm),
        }


def _group(llm):
    out: dict[str, int] = {}
    for e in llm:
        out[e["task"]] = out.get(e["task"], 0) + e["prompt_tokens"] + e["completion_tokens"]
    return out
