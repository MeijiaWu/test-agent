#!/usr/bin/env python3
"""Compare evaluated runs.

    python bench/report.py agent_test oneshot_test agent_norag_test agent_noexec_test --ref oneshot_test

Per run: success rate (harness-resolved / instances) with a bootstrap 95% CI
over instances; the paired difference to ``--ref`` on the instances both runs
attempted (paired bootstrap); cost (tokens, test executions, wall time); and,
for agent runs, how well its own verdict predicts the harness result
(triage precision / recall). Results by issue year are listed because models
may have seen older fixes during training. With ``--gold <run>`` (see gold.py) the
instances whose own reference tests do not resolve are named, and success is also
given over the instances that are solvable in this setup. Writes runs/report.md.
"""
import argparse
import json
import random
from pathlib import Path
from statistics import median

from common import ROOT


def load_run(name):
    d = ROOT / "runs" / name
    ev = json.loads((d / "eval.json").read_text())
    traj = {}
    for f in (d / "trajectories").glob("*.json"):
        t = json.loads(f.read_text())
        traj[t["instance_id"]] = t
    # infrastructure errors are not method failures: leave them out (run.py refuses to evaluate them anyway)
    for i in [i for i, t in traj.items() if t.get("final_verdict") == "error"]:
        ev.pop(i, None)
        traj.pop(i)
    return ev, traj


def boot(xs, reps=10_000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    s = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return s[int(0.025 * reps)], s[int(0.975 * reps)]


def years(tasks_path):
    p = Path(tasks_path)
    if not p.exists():
        return {}
    return {json.loads(l)["instance_id"]: str(json.loads(l).get("created_at", ""))[:4]
            for l in p.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tasks", default="data/swt_lite.jsonl")
    ap.add_argument("--gold", default=None, help="evaluated run of the reference tests (bench/gold.py)")
    a = ap.parse_args()
    data = {r: load_run(r) for r in a.runs}
    yr = years(a.tasks)
    L = ["| run | n | success | 95% CI | Δ vs ref (paired) | 95% CI | median tokens | median test runs | median s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r, (ev, traj) in data.items():
        ids = sorted(ev)
        xs = [int(ev[i]["resolved"]) for i in ids]
        lo, hi = boot(xs)
        dcol, dci = "-", "-"
        if a.ref and a.ref != r:
            rev = data[a.ref][0]
            common = [i for i in ids if i in rev]
            diffs = [int(ev[i]["resolved"]) - int(rev[i]["resolved"]) for i in common]
            dlo, dhi = boot(diffs)
            dcol = f"{100 * sum(diffs) / max(len(diffs), 1):+.1f} pp (n={len(common)})"
            dci = f"[{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
        u = [traj[i].get("usage", {}) for i in ids if i in traj]
        tok = median([x.get("prompt_tokens", 0) + x.get("completion_tokens", 0) for x in u]) if u else 0
        runs_ = median([x.get("test_runs", 0) for x in u]) if u else 0
        secs = median([traj[i].get("seconds", 0) for i in ids if i in traj]) if traj else 0
        L.append(f"| {r} | {len(ids)} | {100 * sum(xs) / max(len(xs), 1):.1f}% | "
                 f"[{100 * lo:.1f}, {100 * hi:.1f}] | {dcol} | {dci} | {tok:.0f} | {runs_:.0f} | {secs:.0f} |")
    if a.gold:
        gev, _ = load_run(a.gold)
        ok = {i for i, e in gev.items() if e["resolved"]}
        bad = sorted(set(gev) - ok)
        L += ["", f"Harness check: the reference tests resolve {len(ok)}/{len(gev)} instances."
              + (f" Not solvable in this setup: {', '.join(bad)}." if bad else ""), ""]
        for r, (ev, _) in data.items():
            sel = [int(ev[i]["resolved"]) for i in ev if i in ok]
            L.append(f"- {r}: {sum(sel)}/{len(sel)} on solvable instances")
    L += ["", "Agent self-assessment vs harness (verdict 'reproduces_issue' as a prediction of 'resolved'):", ""]
    for r, (ev, traj) in data.items():
        claimed = {i for i, t in traj.items() if t.get("final_verdict") == "reproduces_issue"}
        if not claimed and all(t.get("final_verdict") in ("unexecuted", "none", "error") for t in traj.values()):
            continue
        resolved = {i for i, e in ev.items() if e["resolved"]}
        tp = len(claimed & resolved)
        L.append(f"- {r}: claimed {len(claimed)}, resolved {len(resolved)}, precision "
                 f"{100 * tp / max(len(claimed), 1):.0f}%, recall {100 * tp / max(len(resolved), 1):.0f}%")
        conf = {i for i in claimed if traj[i].get("final_fix_check") == "confirmed"}
        if any(t.get("final_fix_check") for t in traj.values()):
            rest = claimed - conf
            L.append(f"  - confirmed by the candidate fix: {len(conf)}, of which resolved {len(conf & resolved)}; "
                     f"not confirmed: {len(rest)}, of which resolved {len(rest & resolved)}")
    if yr:
        L += ["", "Success by issue year:", ""]
        ys = sorted({yr.get(i, "?") for ev, _ in data.values() for i in ev})
        L.append("| run | " + " | ".join(ys) + " |")
        L.append("|---|" + "---|" * len(ys))
        for r, (ev, _) in data.items():
            cells = []
            for y in ys:
                sel = [int(ev[i]["resolved"]) for i in ev if yr.get(i, "?") == y]
                cells.append(f"{sum(sel)}/{len(sel)}" if sel else "-")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    (ROOT / "runs" / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
