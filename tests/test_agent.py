import json

from conftest import BROKEN_IMPORT, GOOD_FIX, GOOD_TEST, TOY, script, triage_by_name
from testagent.agent import TestAgent
from testagent.config import AgentConfig
from testagent.llm import ScriptedLLM
from testagent.oneshot import run_oneshot
from testagent.rag import RepoIndex
from testagent.sandbox import LocalSandbox
from testagent.tracing import Tracer


def fixed_run(files):
    """Local stand-in for the SWT-bench evaluation: run the tests after the fix."""
    return LocalSandbox(TOY / "repo", pre_patch=(TOY / "fix.patch").read_text()).run_tests(files)


def make(write_replies, cfg=None, triage=None):
    llm = ScriptedLLM(script=script(write_replies, triage))
    agent = TestAgent(llm, LocalSandbox(TOY / "repo"), RepoIndex.build(TOY / "repo"), cfg or AgentConfig(),
                      Tracer())
    return agent, llm


def test_repairs_a_broken_test_then_reproduces(toy_task):
    agent, llm = make([BROKEN_IMPORT, GOOD_TEST])
    s = agent.run(toy_task)
    verdicts = [a["verdict"] for a in s["attempts"]]
    assert verdicts == ["test_error", "reproduces_issue"]
    # the import error was caught by rules, not sent to the model for triage
    assert [t for t, _ in llm.prompts].count("triage_failure") == 1
    assert s["final_verdict"] == "reproduces_issue" and s["report"]["requirement_ids"] == ["R1"]
    assert s["target_dir"] == "tests" and "shop/paginate.py" in s["context"]
    # fail on the buggy code (checked in the loop) and pass on the fixed code
    assert all(v == "PASSED" for v in fixed_run(s["final_files"]).statuses.values())
    assert s["model_patch"].startswith("diff --git a/tests/test_default_limit.py")
    u = s["usage"]
    assert u["test_runs"] == 3 and u["llm_calls"] >= 8 and u["prompt_tokens"] > 0   # 2 rounds + candidate fix


def test_review_rejection_triggers_redesign(toy_task):
    agent, llm = make([GOOD_TEST])
    llm.script["review_cases"] = ['{"approved": false, "problems": [{"case_id": "C1", "problem": "expected copies current behaviour"}]}',
                                  '{"approved": true, "problems": []}']
    s = agent.run(toy_task)
    tasks = [t for t, _ in llm.prompts]
    assert tasks.count("design_cases") == 2 and s["final_verdict"] == "reproduces_issue"
    redesign_prompt = [p for t, p in llm.prompts if t == "design_cases"][1]
    assert "expected copies current behaviour" in redesign_prompt


def test_budget_exhaustion_keeps_best_attempt(toy_task):
    cfg = AgentConfig(max_exec_rounds=2)
    agent, _ = make([GOOD_TEST], cfg, triage=triage_by_name(repro="nothing-matches"))
    s = agent.run(toy_task)
    assert len(s["attempts"]) == 2 and s["final_verdict"] == "wrong_expectation"
    assert s["final_files"] and "budget" in s["stop_reason"]


def test_failure_that_would_survive_the_fix_is_not_a_reproduction(toy_task):
    # the model calls the failure a reproduction but admits the test would still fail after the fix
    cfg = AgentConfig(max_exec_rounds=1)
    agent, _ = make([GOOD_TEST], cfg, triage=triage_by_name(passes_after_fix=False))
    s = agent.run(toy_task)
    assert s["attempts"][0]["verdict"] == "wrong_expectation"
    assert s["attempts"][0]["tests"][0]["verdict"] == "wrong_expectation"
    assert s["final_verdict"] != "reproduces_issue"


def test_never_overwrites_an_existing_test_file(toy_task):
    agent, llm = make([GOOD_TEST])
    llm.script["write_tests"] = [json.dumps({"path": "tests/test_cart.py", "content": GOOD_TEST})]
    s = agent.run(toy_task)
    assert list(s["final_files"]) == ["tests/test_cart_agent1.py"]


def test_ablation_without_execution(toy_task):
    agent, llm = make([GOOD_TEST], AgentConfig(use_exec=False, use_review=False, use_rag=False))
    s = agent.run(toy_task)
    tasks = {t for t, _ in llm.prompts}
    assert "review_cases" not in tasks and "triage_failure" not in tasks
    assert s["final_verdict"] == "unexecuted" and s["context"] == "(no retrieval)"
    assert s["usage"]["test_runs"] == 0


def test_oneshot_baseline(toy_task):
    llm = ScriptedLLM(script=script([GOOD_TEST]))
    out = run_oneshot(toy_task, llm, LocalSandbox(TOY / "repo"), RepoIndex.build(TOY / "repo"))
    assert out["final_verdict"] == "unexecuted" and out["usage"]["llm_calls"] == 1
    assert all(v == "PASSED" for v in fixed_run(out["final_files"]).statuses.values())


MIXED_TEST = GOOD_TEST + '''

def test_page_count_wrong_expectation():
    from shop.paginate import page_count
    assert page_count(25) == 2


class TestMore:
    def test_also_wrong(self):
        assert paginate([1, 2, 3], limit=2) == [1]
'''


def test_unrelated_failing_tests_are_pruned(toy_task):
    """The demo failure mode: one test reproduces the issue, others fail for
    reasons a fix would not change. Only the reproducing test may be submitted."""
    agent, llm = make([MIXED_TEST])
    s = agent.run(toy_task)
    final = next(iter(s["final_files"].values()))
    assert s["final_verdict"] == "reproduces_issue" and s["triage"]["keep"] == ["test_default_limit_is_ten"]
    assert "page_count" not in final.split("import")[-1] and "class TestMore" not in final
    assert s["usage"]["test_runs"] == 3                       # pruned file re-run, then run with the fix
    after = fixed_run(s["final_files"]).statuses
    assert after and all(v == "PASSED" for v in after.values())


def test_docstring_assertions_are_rejected_before_running(toy_task):
    agent, llm = make([GOOD_TEST + "\n\ndef test_doc():\n    assert '10' in paginate.__doc__\n", GOOD_TEST])
    s = agent.run(toy_task)
    writes = [p for t, p in llm.prompts if t == "write_tests"]
    assert len(writes) == 2 and "not docstrings" in writes[1]
    assert "__doc__" not in next(iter(s["final_files"].values()))


# ------------------------------------------------------------ candidate-fix check
OVERSPECIFIED = GOOD_TEST.replace("from shop.paginate import paginate",
                                  "import pytest\nfrom shop.paginate import paginate") + \
    """

def test_default_limit_message():
    # fails before the fix (50 items) and after it (the message is not what the issue says)
    assert len(paginate(list(range(100)))) == 10
    with pytest.raises(ValueError, match="page must be positive"):
        paginate([], page=0)
"""


def test_candidate_fix_confirms_and_never_sees_the_test(toy_task):
    agent, llm = make([GOOD_TEST])
    s = agent.run(toy_task)
    assert s["final_verdict"] == "reproduces_issue" and s["final_fix_check"] == "confirmed"
    assert "DEFAULT_LIMIT" in s["report"]["candidate_fix"]
    fix_prompts = [p for t, p in llm.prompts if t == "candidate_fix"]
    assert len(fix_prompts) == 1 and "test_default_limit_is_ten" not in fix_prompts[0]
    assert "def page_count" in fix_prompts[0]                # the real source file, not a summary


def test_candidate_fix_cuts_tests_that_still_fail_with_it(toy_task):
    agent, _ = make([OVERSPECIFIED])                 # both tests are called reproductions by triage
    s = agent.run(toy_task)
    final = next(iter(s["final_files"].values()))
    assert s["final_fix_check"] == "confirmed"
    assert "test_default_limit_is_ten" in final and "test_default_limit_message" not in final
    after = fixed_run(s["final_files"]).statuses
    assert after and all(v == "PASSED" for v in after.values())


def test_refuted_reproduction_is_redesigned_and_fix_reused(toy_task):
    only_bad = OVERSPECIFIED.replace(GOOD_TEST.split("\n\n\n")[1], "")    # just the over-specified test
    agent, llm = make([only_bad, GOOD_TEST])
    s = agent.run(toy_task)
    checks = [a.get("fix_check") for a in s["attempts"]]
    assert checks == ["refuted", "confirmed"]
    assert [t for t, _ in llm.prompts].count("candidate_fix") == 1        # computed once, reused
    assert "not fail-to-pass" in [p for t, p in llm.prompts if t == "design_cases"][-1]
    assert s["final_fix_check"] == "confirmed" and "is_ten" in next(iter(s["final_files"].values()))


def test_unusable_candidate_fix_leaves_reproduction_unverified(toy_task):
    bad = "shop/paginate.py\n<<<<<<< SEARCH\n    if limit is None:\n        limit = 5\n=======\nx\n>>>>>>> REPLACE\n"
    llm = ScriptedLLM(script={**script([GOOD_TEST]), "candidate_fix": [bad]})
    agent = TestAgent(llm, LocalSandbox(TOY / "repo"), RepoIndex.build(TOY / "repo"), AgentConfig(), Tracer())
    s = agent.run(toy_task)
    assert s["final_verdict"] == "reproduces_issue" and s["final_fix_check"] == "unverified"
    fix_prompts = [p for t, p in llm.prompts if t == "candidate_fix"]
    assert len(fix_prompts) == 3 and "could not be applied" in fix_prompts[1]
    assert "limit = 50" in fix_prompts[1]                    # the closest real text is shown


def test_fix_check_ablation(toy_task):
    agent, llm = make([GOOD_TEST], AgentConfig(use_fix_check=False))
    s = agent.run(toy_task)
    assert "candidate_fix" not in {t for t, _ in llm.prompts} and s["final_fix_check"] is None


def test_design_and_write_see_the_issue_text(toy_task):
    agent, llm = make([GOOD_TEST])
    agent.run(toy_task)
    marker = toy_task["problem_statement"].strip().splitlines()[0]
    for task in ("design_cases", "write_tests"):
        assert marker in [p for t, p in llm.prompts if t == task][0]


def test_moves_to_another_test_directory_when_tests_cannot_run_there(toy_task):
    from testagent.schemas import ExecResult
    agent, _ = make([GOOD_TEST])
    nothing = ExecResult(command="x", exit_code=1, output="Aborting: A GIS database backend is required")
    s = {"target_dir": "tests/gis", "attempts": [{"files": {"tests/gis/test_a.py": ""}, "statuses": {}}],
         "target_options": [{"dir": "tests", "example": "tests/test_paginate.py"}]}
    moved = agent._switch_target(s, nothing)
    assert moved["target_dir"] == "tests" and "def test" in moved["example_test"]
    assert moved["target_options"] == [] and "could not run" in moved["retarget_note"]
    # one failed run is not enough, and a run with results never moves
    assert agent._switch_target({**s, "attempts": []}, nothing) == {}
    ran = nothing.model_copy(update={"statuses": {"t": "FAILED"}})
    assert agent._switch_target(s, ran) == {}
