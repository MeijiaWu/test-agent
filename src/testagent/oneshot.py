"""Baseline: one LLM call, no review, no execution -- the zero-shot setting
SWT-bench reports. Gets the same retrieved context and test location as the
agent, so the comparison isolates the agent loop."""

from __future__ import annotations

from . import prompts
from .agent import TestAgent
from .config import AgentConfig
from .llm import BaseLLM, LLMOutputError
from .patch import new_files_patch, safe_relpath
from .rag import RepoIndex, render
from .sandbox import Sandbox
from .schemas import TestFile
from .tracing import Tracer


def run_oneshot(task: dict, llm: BaseLLM, sandbox: Sandbox, index: RepoIndex | None,
                cfg: AgentConfig | None = None, tracer: Tracer | None = None) -> dict:
    cfg = cfg or AgentConfig()
    tracer = tracer or Tracer()
    if llm.on_usage is None:
        llm.on_usage = tracer.on_llm
    helper = TestAgent.__new__(TestAgent)              # reuse location logic without building a graph
    helper.index, helper.sandbox = index, sandbox
    ctx, target = "(no retrieval)", helper._target({}, [], [])
    if index and cfg.use_rag:
        hits = index.search(task["problem_statement"], k=cfg.search_k)
        ctx = render(hits, budget_chars=cfg.context_chars)
        target = helper._target({"task": task}, [h.path for h in hits], [])
    with tracer.span("oneshot"):
        try:
            tf = llm.structured("oneshot", prompts.ONESHOT.format(
                repo=task["repo"], issue=task["problem_statement"], context=ctx,
                target_dir=target["target_dir"], example=target["example_test"]), TestFile)
            files = {helper._fresh_path(safe_relpath(tf.path)): tf.content}
        except (LLMOutputError, ValueError) as e:
            return {"task": task, "final_files": {}, "model_patch": "", "final_verdict": "none",
                    "stop_reason": str(e), "usage": tracer.summary()}
    return {"task": task, "final_files": files, "model_patch": new_files_patch(files),
            "final_verdict": "unexecuted", "stop_reason": "one shot", "usage": tracer.summary()}
