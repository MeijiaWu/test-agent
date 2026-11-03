#!/usr/bin/env python3
"""Build the SWT-bench instance images the agent runs its tests in.

Uses the harness's own builder, ``build_instance_image_from_exec_spec`` --
the function its evaluation calls (base -> environment -> instance image) --
so these are exactly the images the official evaluation uses. (The harness's
``src.prepare_images`` script is not used: in the current harness it imports
a function that no longer exists.)

    python bench/build_images.py --split dev --limit 5

Builds one image at a time by default: environment builds install a whole
conda environment and parallel ones can run Docker out of memory (exit code
137). Failed builds are retried (transient "IncompleteRead" stream drops).
Images that already exist are skipped, so the script can simply be re-run.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import load_tasks
from testagent import swt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/swt_lite.jsonl")
    ap.add_argument("--split", default="all")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--build-mode", default="api", choices=["api", "cli"],
                    help="try cli if the API build fails with UID/GID errors")
    ap.add_argument("--swt-dir", default=None)
    a = ap.parse_args()

    tasks = load_tasks(os.path.abspath(a.tasks), a.split, a.ids, a.limit)
    msg = swt.docker_health()
    if msg:
        sys.exit("ERROR: " + msg)
    root = swt.swt_dir(a.swt_dir)
    swt._mods(str(root))                       # puts the harness on sys.path
    from src.docker_build import build_instance_image_from_exec_spec
    from src.exec_spec import make_exec_spec
    os.chdir(root)                             # the builder writes ./locks and build logs relative to cwd

    def one(t):
        spec = make_exec_spec(t)
        build_instance_image_from_exec_spec(spec, False, a.build_mode)
        return spec.instance_image_key

    def build_all(todo):
        ok, bad = [], []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, t): t for t in todo}
            for f in as_completed(futs):
                t = futs[f]
                try:
                    image = f.result()
                    ok.append(t)
                    print(f"  ok    {t['instance_id']}  ->  {image}", flush=True)
                except Exception as e:              # keep going; report at the end
                    bad.append(t)
                    print(f"  FAIL  {t['instance_id']}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                    if "137" in str(e):
                        print("        exit 137 = killed, usually out of memory: build with --workers 1 "
                              "and give Docker more memory", flush=True)
        return ok, bad

    print(f"building {len(tasks)} instance images (base -> env -> instance) in {root}, "
          f"{a.workers} at a time", flush=True)
    ok, bad = build_all(tasks)
    for attempt in range(1, a.retries + 1):
        if not bad:
            break
        msg = swt.docker_health()
        if msg:
            print("ERROR: " + msg, flush=True)
            break
        print(f"retry {attempt}/{a.retries}: {len(bad)} image(s), one at a time", flush=True)
        time.sleep(10)
        a.workers = 1
        ok2, bad = build_all(bad)
        ok += ok2
    print(f"built {len(ok)}/{len(tasks)}; failed: {[t['instance_id'] for t in bad] or 'none'}")
    print(f"build logs: {root}/image_build_logs/")
    sys.exit(1 if bad and not ok else 0)

if __name__ == "__main__":
    main()
