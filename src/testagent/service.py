"""HTTP API around the agent.

    POST /runs               {"instance_id": ...}  (an SWT-bench task from the tasks file), or
                             {"repo_dir": ..., "issue": ..., "repo": ...}  (a local repository)
                             optional "method": agent | agent_norag | ... | oneshot
    GET  /runs               list
    GET  /runs/{id}          status, verdict, test patch, bug report, attempts, usage
    GET  /runs/{id}/events   server-sent events: every node / LLM call / test run, live

Runs execute in background threads; state is in memory (a demo service, not
a job queue).
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import Config, load
from .llm import BaseLLM
from .runner import METHODS, run_task
from .tracing import Tracer


class RunRequest(BaseModel):
    instance_id: str | None = None
    repo_dir: str | None = None
    issue: str | None = None
    repo: str = "local/repo"
    method: str = "agent"


class _Run:
    def __init__(self, rid, task, method):
        self.id, self.task, self.method = rid, task, method
        self.status, self.result, self.error = "running", None, None
        self.tracer = Tracer()
        self.done = threading.Event()


def create_app(cfg: Config | None = None, tasks_path: str | None = "data/swt_lite.jsonl",
               llm_factory: Callable[[], BaseLLM] | None = None, workdir: str = "work") -> FastAPI:
    cfg = cfg or load()
    app = FastAPI(title="testagent")
    runs: dict[str, _Run] = {}

    def lookup(iid):
        p = Path(tasks_path or "")
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip() and json.loads(line)["instance_id"] == iid:
                    return json.loads(line)
        raise HTTPException(404, f"instance {iid} not in {tasks_path}")

    def work(run: _Run):
        try:
            llm = llm_factory() if llm_factory else None
            run.result = run_task(run.task, cfg, run.method, workdir, llm=llm, tracer=run.tracer)
            run.status = "done"
        except Exception as e:                      # surfaced through GET /runs/{id}
            run.status, run.error = "failed", f"{type(e).__name__}: {e}"
            run.tracer.emit("run_failed", error=run.error)
        finally:
            run.tracer.emit("end", status=run.status)
            run.done.set()

    @app.post("/runs")
    def start(req: RunRequest):
        if req.method not in METHODS:
            raise HTTPException(400, f"method must be one of {sorted(METHODS)}")
        if req.instance_id:
            task = lookup(req.instance_id)
        elif req.repo_dir and req.issue:
            if not Path(req.repo_dir).is_dir():
                raise HTTPException(400, f"no directory {req.repo_dir}")
            task = {"instance_id": f"local-{uuid.uuid4().hex[:6]}", "repo": req.repo,
                    "problem_statement": req.issue, "local_repo": req.repo_dir}
        else:
            raise HTTPException(400, "give instance_id, or repo_dir and issue")
        rid = uuid.uuid4().hex[:10]
        runs[rid] = run = _Run(rid, task, req.method)
        threading.Thread(target=work, args=(run,), daemon=True).start()
        return {"run_id": rid, "instance_id": task["instance_id"]}

    @app.get("/runs")
    def list_runs():
        return [{"run_id": r.id, "instance_id": r.task["instance_id"], "method": r.method,
                 "status": r.status} for r in runs.values()]

    @app.get("/runs/{rid}")
    def get_run(rid: str):
        r = runs.get(rid) or _404(rid)
        res = r.result or {}
        return {"run_id": r.id, "status": r.status, "error": r.error, "method": r.method,
                "instance_id": r.task["instance_id"],
                "final_verdict": res.get("final_verdict"), "stop_reason": res.get("stop_reason"),
                "model_patch": res.get("model_patch"), "report": res.get("report"),
                "attempts": [{k: a[k] for k in ("round", "verdict", "rationale", "statuses")}
                             for a in res.get("attempts", [])],
                "usage": res.get("usage") or r.tracer.summary()}

    @app.get("/runs/{rid}/events")
    def events(rid: str):
        r = runs.get(rid) or _404(rid)
        q: queue.Queue = queue.Queue()
        with r.tracer._lock:                         # replay what happened so far, then follow live
            backlog = list(r.tracer.events)
            r.tracer.subscribers.append(q.put)

        def gen():
            try:
                for ev in backlog:
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                    if ev["type"] == "end":
                        return
                while True:
                    try:
                        ev = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                    if ev["type"] == "end":
                        return
            finally:
                if q.put in r.tracer.subscribers:
                    r.tracer.subscribers.remove(q.put)
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _404(rid):
    raise HTTPException(404, f"no run {rid}")
