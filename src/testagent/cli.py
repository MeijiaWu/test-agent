"""Command line.

    testagent demo                      # the bundled toy issue, local sandbox, your LLM server
    testagent serve --port 8000         # HTTP API
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load

TOY = Path(__file__).resolve().parents[2] / "examples" / "toyshop"


def toy_task() -> dict:
    return {"instance_id": "toyshop-1", "repo": "local/toyshop",
            "problem_statement": (TOY / "ISSUE.md").read_text(),
            "local_repo": str(TOY / "repo"), "fix_patch": str(TOY / "fix.patch")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="testagent")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the agent on the bundled toy issue")
    d.add_argument("--method", default="agent")
    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--tasks", default="data/swt_lite.jsonl")
    a = ap.parse_args(argv)
    cfg = load(a.config, a.set)

    if a.cmd == "demo":
        from .runner import evaluate_local, run_task
        from .tracing import Tracer
        task = toy_task()
        tracer = Tracer()
        tracer.subscribers.append(lambda e: print(f"  [{e['t']:7.2f}s] {e['type']:<10} "
                                                  f"{e.get('node') or e.get('task') or e.get('verdict') or e.get('check') or ''}"))
        res = run_task(task, cfg, a.method, "work", tracer=tracer)
        print("\nverdict:", res["final_verdict"], "|", res["stop_reason"])
        print(res["model_patch"])
        ev = evaluate_local(task, res["final_files"])
        print("fail-to-pass against the known fix:", "RESOLVED" if ev["resolved"] else "not resolved",
              json.dumps({k: ev.get(k) for k in ("before", "after")}))
        print("usage:", json.dumps(res["usage"]))
    elif a.cmd == "serve":
        import uvicorn
        from .service import create_app
        uvicorn.run(create_app(cfg, a.tasks), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
