#!/usr/bin/env python3
"""Fetch SWT-bench Lite through the official harness and split it.

    python bench/prepare.py                       # -> data/swt_lite.jsonl
    python bench/prepare.py --dev-frac 0.2

The dev split is the only one prompts may be tuned on; final numbers are
reported on the test split. The split is a hash of the instance id, so it is
stable and independent of dataset order.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from testagent import swt  # noqa: E402


def split_of(iid: str, dev_frac: float) -> str:
    h = int(hashlib.sha256(iid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_frac else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swt-dir", default=None)
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--out", default="data/swt_lite.jsonl")
    a = ap.parse_args()
    root = swt.swt_dir(a.swt_dir)
    insts = swt.load_instances(root, a.dataset, "test", filter_swt=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in insts:
            i["split"] = split_of(i["instance_id"], a.dev_frac)
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"{len(insts)} instances -> {out}")
    print("splits:", dict(Counter(i["split"] for i in insts)))
    print("repos:", dict(Counter(i["repo"] for i in insts).most_common()))
    years = Counter(str(i.get("created_at", ""))[:4] for i in insts)
    print("issue years:", dict(sorted(years.items())))


if __name__ == "__main__":
    main()
