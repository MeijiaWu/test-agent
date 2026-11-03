import json
import subprocess
import sys
import time

from fastapi.testclient import TestClient

from conftest import BROKEN_IMPORT, GOOD_TEST, ROOT, TOY, script
from testagent.cli import toy_task
from testagent.config import load
from testagent.llm import ScriptedLLM
from testagent.runner import evaluate_local, public_view, run_task
from testagent.service import create_app


def test_public_view_withholds_gold_fields():
    inst = {"instance_id": "x", "repo": "a/b", "problem_statement": "p", "base_commit": "c",
            "version": "1", "created_at": "2020", "golden_code_patch": "FIX", "golden_test_patch": "T",
            "hints_text": "the fix is ...", "FAIL_TO_PASS": "[]"}
    v = public_view(inst)
    assert set(v) == {"instance_id", "repo", "problem_statement", "base_commit", "version", "created_at"}


def test_run_task_local_and_fail_to_pass(tmp_path):
    cfg = load(overrides=["llm.cache_dir=null"])
    task = toy_task()
    res = run_task(task, cfg, "agent", tmp_path, llm=ScriptedLLM(script=script([BROKEN_IMPORT, GOOD_TEST])))
    assert res["final_verdict"] == "reproduces_issue" and "task" not in res
    assert evaluate_local(task, res["final_files"])["resolved"] is True
    # a test that passes on the buggy code is not a reproduction
    assert evaluate_local(task, {"tests/test_t.py": "def test_ok():\n    assert True\n"})["resolved"] is False
    json.dumps(res)                                  # trajectories must be serialisable


def test_api_run_and_event_stream():
    app = create_app(load(), tasks_path=None, llm_factory=lambda: ScriptedLLM(script=script([GOOD_TEST])))
    c = TestClient(app)
    rid = c.post("/runs", json={"repo_dir": str(TOY / "repo"), "issue": (TOY / "ISSUE.md").read_text(),
                                "repo": "local/toyshop"}).json()["run_id"]
    for _ in range(200):
        r = c.get(f"/runs/{rid}").json()
        if r["status"] != "running":
            break
        time.sleep(0.1)
    assert r["status"] == "done" and r["final_verdict"] == "reproduces_issue"
    assert r["model_patch"].startswith("diff --git") and r["usage"]["test_runs"] == 2   # the test, then with the candidate fix
    with c.stream("GET", f"/runs/{rid}/events") as s:
        events = [json.loads(l[6:]) for l in s.iter_lines() if l.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "task" and kinds[-1] == "end" and "exec" in kinds and "llm" in kinds
    assert c.post("/runs", json={"method": "nope", "instance_id": "x"}).status_code == 400


def test_report_script_on_synthetic_runs(tmp_path, monkeypatch):
    runs = ROOT / "runs"
    made = []
    for name, resolved in (("_t_agent", [1, 1, 0, 1]), ("_t_oneshot", [1, 0, 0, 0])):
        d = runs / name
        (d / "trajectories").mkdir(parents=True, exist_ok=True)
        made.append(d)
        ev = {}
        for i, r in enumerate(resolved):
            iid = f"repo__x-{i}"
            ev[iid] = {"resolved": bool(r), "evaluated": True}
            (d / "trajectories" / f"{iid}.json").write_text(json.dumps({
                "instance_id": iid, "final_verdict": "reproduces_issue" if r else "not_reproduced",
                "seconds": 10, "usage": {"prompt_tokens": 100, "completion_tokens": 20, "test_runs": 2}}))
        (d / "eval.json").write_text(json.dumps(ev))
    try:
        out = subprocess.run([sys.executable, str(ROOT / "bench" / "report.py"), "_t_agent", "_t_oneshot",
                              "--ref", "_t_oneshot", "--tasks", "missing.jsonl"],
                             capture_output=True, text=True, check=True).stdout
        assert "| _t_agent | 4 | 75.0% |" in out and "+50.0 pp (n=4)" in out
        assert "precision 100%, recall 100%" in out
    finally:
        import shutil
        for d in made:
            shutil.rmtree(d)
        (runs / "report.md").unlink(missing_ok=True)
