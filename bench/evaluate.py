#!/usr/bin/env python3
"""Score a run with the official SWT-bench harness.

    python bench/evaluate.py --run agent_dev --workers 4

The harness applies each predicted test patch, runs the tests before and
after the gold fix, and marks an instance resolved when the patch adds at
least one fail-to-pass test and no test that fails after the fix. Its
per-instance report.json files are collected into runs/<run>/eval.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT, read_jsonl
from testagent import swt


def collect(root: Path, run: str, model: str, ids) -> dict:
    base = root / "run_instance_swt_logs" / run / model.replace("/", "__")
    out = {}
    for iid in ids:
        f = base / iid / "report.json"
        if not f.exists():
            out[iid] = {"resolved": False, "evaluated": False}
            continue
        rep = json.loads(f.read_text())
        rep = rep.get(iid, rep)
        out[iid] = {"resolved": bool(rep.get("resolved")), "evaluated": True,
                    "added_f2p": rep.get("added_f2p"), "patch_applied": rep.get("patch_successfully_applied")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--swt-dir", default=None)
    ap.add_argument("--collect-only", action="store_true", help="skip the harness, only gather reports")
    a = ap.parse_args()
    root = swt.swt_dir(a.swt_dir)
    run_dir = ROOT / "runs" / a.run
    preds = run_dir / "predictions.jsonl"
    rows = read_jsonl(preds)
    if not rows:
        sys.exit(f"no predictions in {preds}")
    # an empty patch cannot be applied; the harness counts it as unresolved either way
    if not a.collect_only:
        msg = swt.docker_health()
        if msg:
            sys.exit("ERROR: " + msg)
        cmd = [sys.executable, "-m", "src.main", "--dataset_name", "princeton-nlp/SWE-bench_Lite",
               "--filter_swt", "--predictions_path", str(preds.resolve()),
               "--max_workers", str(a.workers), "--run_id", a.run]
        # fresh scores: drop reports of an earlier evaluation of this run (predictions may have changed)
        shutil.rmtree(root / "run_instance_swt_logs" / a.run, ignore_errors=True)
        print("$", " ".join(cmd))
        # the harness imports its modules both as src.x and as top-level x
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")])}
        subprocess.run(cmd, cwd=root, check=False, env=env)
    res = collect(root, a.run, rows[0]["model_name_or_path"], [r["instance_id"] for r in rows])
    (run_dir / "eval.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    k = sum(r["resolved"] for r in res.values())
    print(f"{a.run}: resolved {k}/{n} = {100 * k / max(n, 1):.1f}%  "
          f"(not evaluated: {sum(not r['evaluated'] for r in res.values())})")


if __name__ == "__main__":
    main()
