"""One task, end to end: pick the sandbox, build the retrieval index, run a
method, return a JSON-serialisable result. Shared by the benchmark scripts,
the CLI and the API.

A task is either
* an SWT-bench instance (the full record, gold fields included -- they are
  needed to locate the Docker image and stay inside this module), or
* a local task: ``{"instance_id", "repo", "problem_statement", "local_repo": path}``
  with an optional ``fix_patch`` used only by ``evaluate_local``.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .agent import TestAgent
from .config import Config
from .llm import BaseLLM, OpenAICompatLLM
from .oneshot import run_oneshot
from .rag import RepoIndex
from .sandbox import DockerSandbox, LocalSandbox
from .swt import PUBLIC_FIELDS, swt_dir
from .tracing import Tracer

METHODS = {
    "agent": [],
    "agent_norag": ["agent.use_rag=false"],
    "agent_noreview": ["agent.use_review=false"],
    "agent_noexec": ["agent.use_exec=false"],
    "agent_nofix": ["agent.use_fix_check=false"],
    "oneshot": [],
}


def public_view(task: dict) -> dict:
    """What the agent is allowed to see."""
    return {k: task[k] for k in PUBLIC_FIELDS if k in task}


def method_config(cfg: Config, method: str) -> Config:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    c = copy.deepcopy(cfg)
    for o in METHODS[method]:
        sec, rest = o.split(".", 1)
        key, _, val = rest.partition("=")
        setattr(getattr(c, sec), key, val.lower() == "true")
    return c


def make_sandbox(task: dict, cfg: Config, workdir: Path):
    """Sandbox plus the directory the retrieval index is built from."""
    if "local_repo" in task:
        sb = LocalSandbox(task["local_repo"], timeout=cfg.sandbox.timeout, reruns=cfg.sandbox.reruns)
        return sb, Path(task["local_repo"])
    sb = DockerSandbox(task, swt_dir(cfg.sandbox.swt_dir), timeout=cfg.sandbox.timeout, reruns=cfg.sandbox.reruns)
    repo_copy = sb.export_repo(workdir / "repos" / task["instance_id"])
    sb.set_listing(repo_copy)
    return sb, repo_copy


def run_task(task: dict, cfg: Config, method: str = "agent", workdir: str | Path = "work",
             llm: BaseLLM | None = None, tracer: Tracer | None = None) -> dict:
    workdir = Path(workdir)
    cfg = method_config(cfg, method)
    tracer = tracer or Tracer()
    llm = llm or OpenAICompatLLM(cfg.llm)
    llm.on_usage = tracer.on_llm
    sandbox, repo_dir = make_sandbox(task, cfg, workdir)
    tracer.emit("task", instance_id=task["instance_id"], method=method, sandbox=sandbox.name)
    index = RepoIndex.build(repo_dir, embed_model=cfg.agent.embed_model)
    view = public_view(task)
    if method == "oneshot":
        state = run_oneshot(view, llm, sandbox, index, cfg.agent, tracer)
    else:
        state = TestAgent(llm, sandbox, index, cfg.agent, tracer, repo_root=repo_dir).run(view)
    state = {k: v for k, v in state.items() if k != "task"}
    return {"instance_id": task["instance_id"], "method": method, **state}


def evaluate_local(task: dict, files: dict, cfg: Config | None = None) -> dict:
    """Fail-to-pass check for a local task with a known fix: the official
    criterion (fails on the buggy code, passes with the fix), run locally."""
    if not files:
        return {"resolved": False, "reason": "no tests"}
    before = LocalSandbox(task["local_repo"]).run_tests(files)
    after = LocalSandbox(task["local_repo"], pre_patch=Path(task["fix_patch"]).read_text()).run_tests(files)
    fail_before = any(v in ("FAILED", "ERROR") for v in before.statuses.values())
    pass_after = bool(after.statuses) and all(v in ("PASSED", "SKIPPED") for v in after.statuses.values())
    return {"resolved": fail_before and pass_after, "before": before.statuses, "after": after.statuses}
