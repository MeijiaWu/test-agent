#!/usr/bin/env python3
"""Run one method over SWT-bench tasks and write predictions in the harness format.

    python bench/run.py --method agent   --split dev --limit 20
    python bench/run.py --method oneshot --split test
    python bench/run.py --method agent_norag --split test --set llm.model=qwen3-coder

Outputs (runs/<run>/): predictions.jsonl (instance_id, model_name_or_path,
model_patch, full_output), trajectories/<id>.json (full agent state, usage),
traces/<id>.jsonl (event log), config.json. Re-running resumes: finished
instances are skipped, instances that ended in an infrastructure error are run
again. If the LLM server stops answering, no new instance is started and the
script exits with status 2 (as it does whenever an instance ended in an error),
so a broken run is never evaluated as if the method had failed. Instances whose Docker image is not built yet are skipped (and reported),
so a missing image is never scored as a method failure.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from common import ROOT, load_tasks, read_jsonl
from testagent.config import load
from testagent.llm import llm_health
from testagent import swt
from testagent.runner import METHODS, run_task
from testagent.tracing import Tracer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="agent", choices=sorted(METHODS))
    ap.add_argument("--tasks", default="data/swt_lite.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="tasks in parallel (match llama-server -np)")
    ap.add_argument("--run", default=None, help="run name (default: <method>_<split>)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    ap.add_argument("--workdir", default=str(ROOT / "work"))
    a = ap.parse_args()

    cfg = load(a.config, a.set)
    run = a.run or f"{a.method}_{a.split}"
    out = ROOT / "runs" / run
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"method": a.method, "split": a.split, "set": a.set,
                                                 "config": asdict(cfg)}, indent=1, default=str))
    preds_path = out / "predictions.jsonl"
    # resume: keep finished instances, drop the ones that ended in an infrastructure error
    rows, retry = [], []
    for r in read_jsonl(preds_path):
        tf = out / "trajectories" / f"{r['instance_id']}.json"
        verdict = json.loads(tf.read_text()).get("final_verdict") if tf.exists() else None
        (retry if verdict == "error" else rows).append(r)
    if retry:
        preds_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    done = {r["instance_id"] for r in rows}
    tasks = [t for t in load_tasks(a.tasks, a.split, a.ids, a.limit) if t["instance_id"] not in done]

    msg = llm_health(cfg.llm)
    if msg:
        sys.exit(f"ERROR: LLM server: {msg} -- is llama-server running?")
    if any("local_repo" not in t for t in tasks):
        msg = swt.docker_health()
        if msg:
            sys.exit("ERROR: " + msg)
        root = swt.swt_dir(cfg.sandbox.swt_dir)
        missing = [t["instance_id"] for t in tasks if "local_repo" not in t and subprocess.run(
            ["docker", "image", "inspect", swt.image_name(root, t)], capture_output=True).returncode]
        if missing:
            print(f"skipping {len(missing)} instance(s) without a built image: {' '.join(missing)}\n"
                  f"  build them: python bench/build_images.py --ids {' '.join(missing)}")
            tasks = [t for t in tasks if t["instance_id"] not in set(missing)]
    print(f"{run}: {len(tasks)} to run, {len(done)} already done"
          + (f", {len(retry)} retried after an error" if retry else ""))
    lock = threading.Lock()
    llm_down = threading.Event()

    def one(task):
        iid = task["instance_id"]
        if llm_down.is_set():                       # not started: resumes on the next run
            return iid, "not started", 0, ""
        tracer = Tracer(out / "traces" / f"{iid}.jsonl")
        t0 = time.time()
        try:
            res = run_task(task, cfg, a.method, a.workdir, tracer=tracer)
        except Exception as e:                      # infrastructure failure: record and move on
            res = {"instance_id": iid, "method": a.method, "final_files": {}, "model_patch": "",
                   "final_verdict": "error", "stop_reason": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(), "usage": tracer.summary()}
            if "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
                if llm_health(cfg.llm):
                    llm_down.set()
        res["seconds"] = round(time.time() - t0, 1)
        (out / "trajectories" / f"{iid}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                                     default=str))
        with lock, preds_path.open("a") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": run,
                                "model_patch": res.get("model_patch", ""),
                                "full_output": res.get("stop_reason", "")}, ensure_ascii=False) + "\n")
        return iid, res.get("final_verdict"), res["seconds"], res.get("stop_reason", "")

    bad = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            iid, verdict, sec, why = f.result()
            if verdict in ("error", "not started"):
                bad.append(iid)
            extra = f"  <- {why.splitlines()[0][:200]}" if verdict == "error" and why else ""
            print(f"[{i}/{len(tasks)}] {iid}: {verdict} ({sec}s){extra}", flush=True)
    print(f"predictions: {preds_path}")
    if llm_down.is_set():
        print(f"ERROR: the LLM server stopped answering ({llm_health(cfg.llm) or 'recovered since'}); "
              "restart it and run the same command again to finish the remaining instances")
    if bad:
        print(f"{len(bad)} instance(s) ended in an error or were not started -- re-run to retry them; "
              "not evaluating an incomplete run")
        sys.exit(2)


if __name__ == "__main__":
    main()
