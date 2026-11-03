#!/usr/bin/env python3
"""Harness sanity check: submit each instance's own reference tests (the dataset's
``test_patch``) as the prediction.

    python bench/gold.py --split dev --limit 5 --run gold_dev5
    python bench/evaluate.py --run gold_dev5

The reference tests are fail-to-pass by construction, so every instance should be
resolved. One that is not cannot be solved by any method in this setup (image,
test command or parser problem) and is reported separately by ``report.py --gold``
instead of being counted as a method failure.
"""
import argparse
import json

from common import ROOT, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/swt_lite.jsonl")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run = a.run or f"gold_{a.split}"
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(a.tasks, a.split, a.ids, a.limit)
    with (out / "predictions.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps({"instance_id": t["instance_id"], "model_name_or_path": run,
                                "model_patch": t.get("golden_test_patch") or t["test_patch"], "full_output": "reference tests"}) + "\n")
    print(f"{run}: {len(tasks)} reference predictions -> {out / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
