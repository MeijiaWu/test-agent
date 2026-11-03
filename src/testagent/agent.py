"""The test agent as a LangGraph state machine.

    analyze -> retrieve -> design <-> review -> data -> write -> execute -> triage
                             ^                           ^                 |
                             |  wrong_expectation /      |  test_error     |
                             +--- not_reproduced --------+-----------------+
                             |                                             | reproduces_issue
                             +------------ refuted ---------------- verify (candidate fix)
                                                                           | confirmed / unverified / budget
                                                                           v
                                                                         report

Each node is a method that reads the state and returns the fields it changes.
The LLM is used where judgement is needed (reading the issue, designing and
reviewing cases, choosing data, writing code, interpreting a failure); file
handling, syntax checks, execution, flakiness and the obvious triage outcomes
are plain code. Ablation switches in ``AgentConfig`` remove retrieval, review
or execution feedback without changing anything else.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import AgentConfig
from .llm import BaseLLM, LLMOutputError
from .patch import new_files_patch, safe_relpath
from .rag import RepoIndex, render
from .sandbox import Sandbox
from .prune import func_name, match_id, prune
from .fixcheck import FixError, apply_edits, diff, fix_broke_code, parse_edits
from .rag import is_test_path
from .schemas import (BugReport, CaseReview, Edit, ExecResult, FailureTriage, IssueAnalysis,
                      TestData, TestFile, TestPlan, Triage)
from .tracing import Tracer
from .triage import fixture_error, pre_triage


class State(TypedDict, total=False):
    task: dict                    # public fields only: instance_id, repo, problem_statement, ...
    analysis: dict
    context: str                  # rendered code context
    locations: list[str]          # retrieved code locations (for the bug report)
    target_dir: str
    example_test: str
    target_options: list          # other candidate test directories, in order
    plan: dict
    review: dict
    design_round: int
    data: dict
    files: dict                   # path -> content of the current attempt
    exec: dict
    triage: dict
    exec_round: int
    feedback: str                 # what the next design / write step must fix
    attempts: list                # history: files, exec summary, verdict per round
    final_files: dict
    final_verdict: str
    model_patch: str
    report: dict
    stop_reason: str
    fix: dict                     # candidate fix: edits, diff, status (reused across rounds)
    fix_check: str                # confirmed / refuted / unverified for the latest reproduction
    final_fix_check: str          # fix check of the submitted attempt (None if not checked)


def _reqs(state) -> str:
    a = state.get("analysis") or {}
    return "\n".join(f"- {r['id']} [{r['kind']}] {r['text']}" for r in a.get("requirements", []))


def _issue(state, limit: int = 8000) -> str:
    text = (state.get("task") or {}).get("problem_statement", "")
    return text if len(text) <= limit else text[:limit] + "\n...(truncated)"


def _cases(state, with_data=False) -> str:
    cases = (state.get("plan") or {}).get("cases", [])
    data = {d["case_id"]: d["values"] for d in (state.get("data") or {}).get("data", [])}
    out = []
    for c in cases:
        line = (f"- {c['id']} ({c['kind']}, checks {', '.join(c['requirement_ids'])}): {c['description']}\n"
                f"  inputs: {c['inputs']}\n  expected: {c['expected']}")
        if with_data and c["id"] in data:
            line += f"\n  data: {json.dumps(data[c['id']], ensure_ascii=False)}"
        out.append(line)
    return "\n".join(out)


class TestAgent:
    __test__ = False

    def __init__(self, llm: BaseLLM, sandbox: Sandbox, index: RepoIndex | None,
                 cfg: AgentConfig | None = None, tracer: Tracer | None = None,
                 repo_root: str | Path | None = None):
        self.llm, self.sandbox, self.index = llm, sandbox, index
        # read-only copy of the repository at the base commit (candidate fixes are computed on it)
        self.repo_root = Path(repo_root) if repo_root else (index.root if index else None)
        self.cfg = cfg or AgentConfig()
        self.tracer = tracer or Tracer()
        if self.llm.on_usage is None:
            self.llm.on_usage = self.tracer.on_llm
        self.graph = self._build()

    # ------------------------------------------------------------- helpers
    def _ask(self, task, prompt, schema):
        return self.llm.structured(task, prompt, schema)

    def _tokens(self) -> int:
        return sum(u.prompt_tokens + u.completion_tokens for u in self.llm.calls)

    def _over_budget(self) -> bool:
        return self._tokens() > self.cfg.max_tokens

    def _node(self, name, fn):
        def wrapped(state):
            with self.tracer.span(name):
                return fn(state)
        return wrapped

    # --------------------------------------------------------------- nodes
    def analyze(self, s: State) -> dict:
        t = s["task"]
        a = self._ask("analyze_issue", prompts.ANALYZE.format(repo=t["repo"], issue=t["problem_statement"]),
                      IssueAnalysis)
        return {"analysis": a.model_dump(), "design_round": 0, "exec_round": 0, "attempts": [],
                "feedback": ""}

    def retrieve(self, s: State) -> dict:
        a = s["analysis"]
        if not (self.cfg.use_rag and self.index):
            return {"context": "(no retrieval)", "locations": [], **self._target(s, [], [])}
        hits = []
        for sym in a.get("symbols", []):
            hits += self.index.lookup(sym)
        query = " ".join([a["summary"], *a.get("symbols", []), *(r["text"] for r in a["requirements"])])
        hits += self.index.search(query, k=self.cfg.search_k)
        ctx = render(hits, budget_chars=self.cfg.context_chars)
        locs = list(dict.fromkeys(h.header() for h in hits))[:10]
        self.tracer.emit("retrieval", hits=locs)
        return {"context": ctx, "locations": locs,
                **self._target(s, [h.path for h in hits], a.get("symbols", []))}

    def _target(self, s, code_paths, symbols) -> dict:
        """Where the new test goes, and an existing test there to imitate; other candidate
        directories are kept in case tests cannot run in the first one."""
        issue = (s.get("task") or {}).get("problem_statement", "")
        related = self.index.related_tests(code_paths, symbols, k=8, issue_text=issue) if self.index else []
        options, seen = [], set()
        for rel in related:
            d = str(Path(rel).parent.as_posix())
            if d not in seen:
                seen.add(d)
                options.append({"dir": d, "example": rel})
        if not options:
            return {"target_dir": "tests", "example_test": "(no existing test found)", "target_options": []}
        first = options[0]
        return {"target_dir": first["dir"], "example_test": self._example(first["example"]),
                "target_options": options[1:4]}

    def _example(self, rel: str) -> str:
        return (self.index.root / rel).read_text(errors="replace")[:5000] if self.index else ""

    def _switch_target(self, s, r: ExecResult) -> dict:
        """Two runs in a row produced no test result at all in the same directory: the
        directory itself may be unusable here (e.g. it needs a special backend). Move on."""
        attempts = s.get("attempts") or []
        options = list(s.get("target_options") or [])
        if r.statuses or not attempts or attempts[-1]["statuses"] or not options:
            return {}
        if Path(next(iter(attempts[-1]["files"]))).parent.as_posix() != s.get("target_dir"):
            return {}
        nxt = options.pop(0)
        self.tracer.emit("retarget", old=s.get("target_dir"), new=nxt["dir"])
        return {"target_dir": nxt["dir"], "example_test": self._example(nxt["example"]),
                "target_options": options, "files": {},
                "retarget_note": f"Tests under {s.get('target_dir')} could not run in this environment "
                                 f"twice in a row; write the new file under {nxt['dir']} instead.\n"}

    def design(self, s: State) -> dict:
        fb = s.get("feedback") or ""
        fb = f"Feedback to address:\n{fb}\n" if fb else ""
        plan = self._ask("design_cases", prompts.DESIGN.format(
            issue=_issue(s), requirements=_reqs(s),
            ambiguities="; ".join(s["analysis"].get("ambiguities", [])) or "none",
            context=s.get("context", ""), feedback=fb), TestPlan)
        return {"plan": plan.model_dump(), "design_round": s.get("design_round", 0) + 1}

    def review(self, s: State) -> dict:
        problems = self._rule_review(s)
        if problems:                       # cheap structural checks first, no LLM call
            r = CaseReview(approved=False, problems=problems)
        else:
            r = self._ask("review_cases", prompts.REVIEW.format(requirements=_reqs(s), cases=_cases(s)),
                          CaseReview)
        fb = "\n".join(f"- {p.case_id}: {p.problem}" for p in r.problems)
        return {"review": r.model_dump(), "feedback": fb if not r.approved else ""}

    @staticmethod
    def _rule_review(s) -> list:
        from .schemas import CaseProblem
        reqs = {r["id"]: r["kind"] for r in s["analysis"]["requirements"]}
        cases = s["plan"]["cases"]
        out = []
        if not any(c["kind"] == "reproduce" for c in cases):
            out.append(CaseProblem(case_id="-", problem="no 'reproduce' case"))
        for c in cases:
            unknown = [r for r in c["requirement_ids"] if r not in reqs]
            if unknown:
                out.append(CaseProblem(case_id=c["id"], problem=f"unknown requirement ids {unknown}"))
            if c["kind"] == "reproduce" and not any(reqs.get(r) == "expected" for r in c["requirement_ids"]):
                out.append(CaseProblem(case_id=c["id"], problem="reproduce case checks no 'expected' requirement"))
            if not c["expected"].strip():
                out.append(CaseProblem(case_id=c["id"], problem="missing expected outcome"))
        return out

    def data(self, s: State) -> dict:
        d = self._ask("test_data", prompts.DATA.format(cases=_cases(s), context=s.get("context", "")), TestData)
        bad = []
        for cd in d.data:                  # every value must at least be a Python expression
            for k, v in cd.values.items():
                try:
                    ast.parse(v, mode="eval")
                except SyntaxError:
                    bad.append(f"{cd.case_id}.{k}")
        if bad:
            self.tracer.emit("data_invalid", values=bad)
            d.data = [cd.model_copy(update={"values": {k: v for k, v in cd.values.items()
                                                       if f"{cd.case_id}.{k}" not in bad}})
                      for cd in d.data]
        return {"data": d.model_dump()}

    def write(self, s: State) -> dict:
        fb = s.get("feedback") or ""
        prev = s.get("files") or {}
        if fb and prev:
            path, code = next(iter(prev.items()))
            fb = f"Previous attempt ({path}):\n```python\n{code}\n```\nProblem to fix:\n{fb}\n"
        elif fb:
            fb = f"Problem to fix:\n{fb}\n"
        prompt = prompts.WRITE.format(repo=s["task"]["repo"], issue=_issue(s), requirements=_reqs(s),
                                      cases=_cases(s, with_data=True), context=s.get("context", ""),
                                      target_dir=s.get("target_dir", "tests"),
                                      example=s.get("example_test", ""), feedback=fb)
        for attempt in range(3):           # syntax / path problems are fixed before running anything
            tf = self._ask("write_tests", prompt, TestFile)
            err = self._check_file(tf)
            if not err:
                break
            prompt += f"\n\nYour file was rejected before running: {err}. Fix it."
        else:
            raise LLMOutputError("write_tests", err)
        path = self._fresh_path(safe_relpath(tf.path))
        return {"files": {path: tf.content}, "feedback": ""}

    @staticmethod
    def _check_file(tf: TestFile) -> str | None:
        try:
            p = safe_relpath(tf.path)
        except ValueError as e:
            return str(e)
        if not Path(p).name.startswith("test"):
            return f"file name {Path(p).name!r} must start with 'test'"
        try:
            ast.parse(tf.content)
        except SyntaxError as e:
            return f"SyntaxError line {e.lineno}: {e.msg}"
        if "__doc__" in tf.content or "getsource" in tf.content:
            return "tests must check behaviour, not docstrings or source text"
        return None

    def _fresh_path(self, path: str) -> str:
        """Never overwrite an existing file of the repository."""
        if not self.sandbox.exists(path):
            return path
        stem = path[:-3]
        i = 1
        while self.sandbox.exists(f"{stem}_agent{i}.py"):
            i += 1
        return f"{stem}_agent{i}.py"

    def execute(self, s: State) -> dict:
        t = time.perf_counter()
        r = self.sandbox.run_tests(s["files"])
        self.tracer.emit("exec", command=r.command, exit_code=r.exit_code, statuses=r.statuses,
                         consistent=r.consistent, seconds=round(time.perf_counter() - t, 2))
        return {"exec": r.model_dump(), "exec_round": s.get("exec_round", 0) + 1}

    def triage(self, s: State) -> dict:
        r = ExecResult(**s["exec"])
        files = s["files"]
        v = pre_triage(r)
        per: list[dict] = []
        if v is None:
            v, per, files, r = self._judge_failures(s, r)
        attempts = list(s.get("attempts", [])) + [{
            "round": s["exec_round"], "files": files, "statuses": r.statuses,
            "exit_code": r.exit_code, "verdict": v.verdict, "rationale": v.rationale, "tests": per}]
        fb = "" if v.verdict == "reproduces_issue" else f"[{v.verdict}] {v.rationale}\n" \
             f"Test output (tail):\n{r.output[-3000:]}"
        self.tracer.emit("triage", verdict=v.verdict, rationale=v.rationale, kept=v.keep)
        out = {"triage": v.model_dump(), "attempts": attempts, "feedback": fb, "files": files,
               "exec": r.model_dump()}
        if v.verdict == "test_error":
            moved = self._switch_target(s, r)
            if moved:
                out["feedback"] = moved.pop("retarget_note") + fb
                out.update(moved)
        if v.verdict in ("wrong_expectation", "not_reproduced"):
            out["design_round"] = 0        # a redesign gets its own review budget
        return out

    def _judge_failures(self, s, r: ExecResult):
        """One LLM verdict per failing test. Tests that reproduce the issue are
        kept; the rest are cut from the file, and the pruned file is run again
        to confirm it still fails the same way."""
        files = s["files"]
        path, code = next(iter(files.items()))
        failing = [k for k, v in r.statuses.items() if v in ("FAILED", "ERROR")]
        ft = self._ask("triage_failure", prompts.TRIAGE.format(
            requirements=_reqs(s), path=path, code=code, command=r.command, exit_code=r.exit_code,
            statuses=json.dumps(r.statuses), output=r.output, failing="\n".join(f"- {f}" for f in failing)),
            FailureTriage)
        judged = {}
        for t in ft.tests:
            key = match_id(t.test, failing)
            if key:
                broken = fixture_error(r.output, key) if t.verdict == "reproduces_issue" else None
                if broken:
                    t = t.model_copy(update={"verdict": "test_error", "rationale": broken})
                elif t.verdict == "reproduces_issue" and not t.passes_after_fix:
                    # fails now, but would still fail after the fix: not fail-to-pass
                    t = t.model_copy(update={"verdict": "wrong_expectation",
                                             "rationale": f"would still fail after the fix: {t.after_fix}"})
                judged[key] = t
        per = [{"test": k, "verdict": t.verdict, "requirement_id": t.requirement_id,
                "after_fix": t.after_fix, "rationale": t.rationale} for k, t in judged.items()]
        repro = [k for k, t in judged.items() if t.verdict == "reproduces_issue"]
        if not repro:
            kinds = {t.verdict for t in judged.values()} or {"test_error"}
            verdict = "test_error" if "test_error" in kinds else "wrong_expectation"
            why = " | ".join(f"{func_name(k)}: {t.rationale}" for k, t in judged.items()) or \
                "no failing test could be attributed to the issue"
            return Triage(verdict=verdict, rationale=why), per, files, r
        keep = {func_name(k) for k in repro}
        first = judged[repro[0]]
        v = Triage(verdict="reproduces_issue", requirement_id=first.requirement_id,
                   rationale=first.rationale, keep=sorted(keep))
        pruned = {path: prune(code, keep)}
        if pruned[path] == code:
            return v, per, files, r
        r2 = self.sandbox.run_tests(pruned)
        self.tracer.emit("exec", command=r2.command, exit_code=r2.exit_code, statuses=r2.statuses,
                         consistent=r2.consistent, seconds=0, purpose="verify pruned file")
        ok = r2.consistent and r2.statuses and all(st in ("FAILED", "ERROR") for st in r2.statuses.values()) \
            and {func_name(k) for k in r2.statuses} == keep
        if ok:
            return v, per, pruned, r2
        v.rationale += " (pruned file did not fail as before; kept the unpruned file)"
        return v, per, files, r

    # ------------------------------------------------- candidate-fix check
    def _fix_enabled(self) -> bool:
        return self.cfg.use_fix_check and self.cfg.use_exec and self.repo_root is not None

    def _candidate_fix(self, s: State, files: dict):
        """Get a candidate fix (written without seeing the tests) and run ``files`` with it.
        Returns (fix_state, overlay or None, ExecResult or None)."""
        fix = dict(s.get("fix") or {})
        if fix.get("status") in ("unapplied", "broken"):       # tried before; do not spend more calls
            return fix, None, None
        fb = ""
        for attempt in range(self.cfg.max_fix_attempts):
            if fix.get("status") == "ok" and attempt == 0:      # reuse the fix from an earlier round
                edits = [Edit(**e) for e in fix["candidate"]["edits"]]
            else:
                try:
                    edits = self.llm.text("candidate_fix", prompts.FIX.format(
                        repo=s["task"]["repo"], issue=_issue(s), requirements=_reqs(s),
                        context=self._fix_context(s), feedback=fb), parse_edits)
                except LLMOutputError as e:
                    fix.update(status="unapplied", error=str(e))
                    return fix, None, None
                fix["candidate"] = {"edits": [e.model_dump() for e in edits]}
            try:
                overlay = apply_edits(self.repo_root, edits)
            except FixError as e:
                fix.update(status="unapplied", error=str(e))
                fb = f"Your previous edits could not be applied: {e}\n"
                continue
            r = self.sandbox.run_tests(files, overlay=overlay)
            self.tracer.emit("exec", command=r.command, exit_code=r.exit_code, statuses=r.statuses,
                             consistent=r.consistent, seconds=0, purpose="candidate fix")
            broke = fix_broke_code(r.output, r.statuses, overlay)
            if broke:
                fix.update(status="broken", error=broke)
                fb = f"With your previous edits the code no longer worked:\n{broke}\n"
                continue
            fix.update(status="ok", error="", diff=diff(self.repo_root, overlay))
            return fix, overlay, r
        return fix, None, None

    def _fix_context(self, s: State, file_chars: int = 20_000, budget: int = 45_000) -> str:
        """Real source for the fix author: whole files when small, else the retrieved regions
        with some margin -- search text must be copied from what is actually in the file."""
        spans: dict[str, list[tuple[int, int]]] = {}
        for loc in s.get("locations") or []:
            head = loc.split()[0]
            path, _, rng = head.rpartition(":")
            if not path or is_test_path(path):
                continue
            a, _, b = rng.partition("-")
            if a.isdigit():
                spans.setdefault(path, []).append((int(a), int(b or a)))
        if not spans or self.repo_root is None:
            return s.get("context", "")
        out, used = [], 0
        for path, rngs in list(spans.items())[:4]:
            f = self.repo_root / path
            if not f.is_file():
                continue
            text = f.read_text(errors="replace")
            if len(text) > file_chars:
                lines = text.splitlines()
                parts, last = [], 0
                for a, b in sorted(rngs):
                    lo, hi = max(0, a - 30), min(len(lines), b + 30)
                    if lo < last:
                        lo = last
                    if lo >= hi:
                        continue
                    parts.append(f"# ... lines {lo + 1}-{hi}\n" + "\n".join(lines[lo:hi]))
                    last = hi
                text = "\n".join(parts)
            block = f"### {path}\n```python\n{text}\n```"
            if used + len(block) > budget:
                break
            out.append(block)
            used += len(block)
        return "\n\n".join(out) or s.get("context", "")

    def verify(self, s: State) -> dict:
        """Run the reproducing tests again with a candidate fix applied (see fixcheck.py)."""
        files = s["files"]
        path, code = next(iter(files.items()))
        failing = {func_name(k) for k, v in s["exec"]["statuses"].items() if v in ("FAILED", "ERROR")}
        keep = set(s["triage"].get("keep") or []) or failing
        fix, overlay, r = self._candidate_fix(s, files)
        attempts = [dict(a) for a in s.get("attempts", [])]
        out: dict[str, Any] = {"fix": fix, "attempts": attempts}
        check, why = "unverified", f"no usable candidate fix ({fix.get('status')}: {fix.get('error', '')[:300]})"
        if overlay is not None:
            passed = {func_name(k) for k, v in r.statuses.items() if v == "PASSED"}
            good = keep & passed
            if good == keep:
                check, why = "confirmed", "every reproducing test passes with the candidate fix"
            elif good:
                pruned = {path: prune(code, good)}
                before = self.sandbox.run_tests(pruned)
                after = self.sandbox.run_tests(pruned, overlay=overlay)
                ok = (before.consistent and before.statuses
                      and all(v in ("FAILED", "ERROR") for v in before.statuses.values())
                      and {func_name(k) for k in before.statuses} == good
                      and after.statuses and all(v == "PASSED" for v in after.statuses.values()))
                if ok:
                    check, why = "confirmed", (f"kept {sorted(good)}; cut {sorted(keep - good)}, "
                                               "which still failed with the candidate fix")
                    files = pruned
                    out.update(files=pruned, exec=before.model_dump())
                    attempts[-1].update(files=pruned, statuses=before.statuses)
                else:
                    why = "the pruned file did not behave as expected with and without the fix"
            else:
                check = "refuted"
                why = "every reproducing test still fails with the candidate fix"
                out["feedback"] = (
                    "[not fail-to-pass] A candidate fix written from the issue alone (it never saw your test) "
                    "makes the code behave as the issue asks, yet your test still fails with it:\n"
                    f"{r.output[-2500:]}\n"
                    "So the test probably asserts something the issue does not specify, or exercises a "
                    "different code path than the one the issue is about. Redesign it.")
                out["design_round"] = 0
        attempts[-1].update(fix_check=check, fix_reason=why)
        self.tracer.emit("fix_check", check=check, reason=why, fix_status=fix.get("status"))
        out["fix_check"] = check
        return out

    def report(self, s: State) -> dict:
        files, verdict, reason = self._choose_final(s)
        out: dict[str, Any] = {"final_files": files, "final_verdict": verdict,
                               "model_patch": new_files_patch(files) if files else "",
                               "stop_reason": s.get("stop_reason") or reason}
        chosen = next((a for a in reversed(s.get("attempts") or []) if a["files"] == files), {})
        out["final_fix_check"] = chosen.get("fix_check")
        if verdict == "reproduces_issue":
            path, code = next(iter(files.items()))
            rep = self._ask("bug_report", prompts.REPORT.format(
                requirements=_reqs(s), path=path, code=code, output=s["exec"]["output"][-4000:],
                locations=", ".join(s.get("locations", [])) or "none"), BugReport)
            out["report"] = rep.model_dump()
            if chosen.get("fix_check") == "confirmed" and (s.get("fix") or {}).get("diff"):
                out["report"]["candidate_fix"] = s["fix"]["diff"]
        return out

    @staticmethod
    def _choose_final(s):
        """A reproducing attempt if any -- one confirmed by the candidate fix first, then an
        unchecked one, then a refuted one (latest first within each); else the latest attempt
        whose tests at least failed consistently on the buggy code; else the last file written."""
        attempts = s.get("attempts") or []
        rank = {"confirmed": 0, None: 1, "unverified": 1, "refuted": 2}
        repro = [a for a in attempts if a["verdict"] == "reproduces_issue"]
        if repro:
            a = min(reversed(repro), key=lambda a: rank.get(a.get("fix_check"), 1))
            tag = a.get("fix_check")
            return a["files"], "reproduces_issue", "reproduced" + (f" ({tag} by candidate fix)" if tag else "")
        for a in reversed(attempts):
            if a["verdict"] == "wrong_expectation" or (
                    a["verdict"] != "test_error" and any(v == "FAILED" for v in a["statuses"].values())):
                return a["files"], a["verdict"], "budget exhausted; best failing attempt"
        if s.get("files"):
            return s["files"], (attempts[-1]["verdict"] if attempts else "unexecuted"), \
                "budget exhausted" if attempts else "no execution feedback"
        return {}, "none", "no test produced"

    # -------------------------------------------------------------- routing
    def _after_design(self, s) -> str:
        return "review" if self.cfg.use_review else "data"

    def _after_review(self, s) -> str:
        if s["review"]["approved"] or s["design_round"] >= self.cfg.max_design_rounds or self._over_budget():
            return "data"
        return "design"

    def _after_write(self, s) -> str:
        return "execute" if self.cfg.use_exec else "report"

    def _after_triage(self, s) -> str:
        v = s["triage"]["verdict"]
        if v == "reproduces_issue":
            return "verify" if self._fix_enabled() else "report"
        if s["exec_round"] >= self.cfg.max_exec_rounds or self._over_budget():
            return "report"
        return "write" if v == "test_error" else "design"

    def _after_verify(self, s) -> str:
        if s.get("fix_check") == "refuted" and s["exec_round"] < self.cfg.max_exec_rounds \
                and not self._over_budget():
            return "design"
        return "report"

    def _build(self):
        g = StateGraph(State)
        for name in ("analyze", "retrieve", "design", "review", "data", "write", "execute", "triage",
                     "verify", "report"):
            g.add_node(name, self._node(name, getattr(self, name)))
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "retrieve")
        g.add_edge("retrieve", "design")
        g.add_conditional_edges("design", self._after_design, {"review": "review", "data": "data"})
        g.add_conditional_edges("review", self._after_review, {"design": "design", "data": "data"})
        g.add_edge("data", "write")
        g.add_conditional_edges("write", self._after_write, {"execute": "execute", "report": "report"})
        g.add_edge("execute", "triage")
        g.add_conditional_edges("triage", self._after_triage,
                                {"report": "report", "write": "write", "design": "design", "verify": "verify"})
        g.add_conditional_edges("verify", self._after_verify, {"report": "report", "design": "design"})
        g.add_edge("report", END)
        return g.compile()

    # ------------------------------------------------------------------ run
    def run(self, task: dict) -> State:
        limit = 10 + 8 * (self.cfg.max_exec_rounds + self.cfg.max_design_rounds)
        state: State = {"task": task}
        try:
            for update in self.graph.stream(state, config={"recursion_limit": limit},
                                            stream_mode="updates"):
                for node, delta in update.items():
                    state.update(delta or {})
        except LLMOutputError as e:
            self.tracer.emit("abort", error=str(e))
            state["stop_reason"] = f"invalid LLM output: {e}"
            files, verdict, _ = self._choose_final(state)
            state.update({"final_files": files, "final_verdict": verdict,
                          "model_patch": new_files_patch(files) if files else ""})
        state["usage"] = self.tracer.summary()
        return state
