"""Thin adapter over the official SWT-bench harness (github.com/logic-star-ai/swt-bench).

Everything benchmark-specific -- which instances, which Docker image holds a
given repository at its base commit, which command runs its tests, how test
paths become test-runner arguments, how logs are parsed -- is taken from the
harness itself rather than re-implemented, so the agent's in-loop execution
behaves like the official evaluation.

The harness is a plain checkout run as ``python -m src.main`` from its root;
``TESTAGENT_SWT_DIR`` (or ``swt_dir`` in the config) points at it.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Fields the agent may see. The gold code patch, the gold test patch, the
# FAIL_TO_PASS / PASS_TO_PASS lists and the issue comments (``hints_text``,
# which often discuss the fix) are withheld.
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "version", "problem_statement", "created_at")


def swt_dir(explicit: str | None = None) -> Path:
    d = Path(explicit or os.environ.get("TESTAGENT_SWT_DIR", "third_party/swt-bench")).resolve()
    if not (d / "src" / "main.py").exists():
        raise FileNotFoundError(
            f"SWT-bench harness not found at {d}. Clone it:\n"
            f"  git clone https://github.com/logic-star-ai/swt-bench {d}\n"
            f"  pip install -e {d}\n"
            f"and set TESTAGENT_SWT_DIR or swt_dir in the config.")
    return d


@lru_cache(maxsize=None)
def _mods(root: str):
    # The harness imports itself both as ``src.x`` and as top-level ``x``
    # (``pip install -e`` exposes both); put both locations on the path.
    for p in (os.path.join(root, "src"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(root)                # the harness opens dataset/filter_cases_*.txt relative to its root
    try:
        from src import constants, dataset, exec_spec, log_parsers, utils   # noqa: E402
    finally:
        os.chdir(cwd)
    return constants, dataset, exec_spec, log_parsers, utils


def load_instances(root: Path, name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test",
                   filter_swt: bool = True) -> list[dict]:
    """SWT-bench instances (needs network access to Hugging Face the first time)."""
    _, dataset, *_ = _mods(str(root))
    cwd = os.getcwd()
    os.chdir(root)
    try:
        return [dict(i) for i in dataset.load_swebench_dataset(name, split, False, filter_swt)]
    finally:
        os.chdir(cwd)


def image_name(root: Path, instance: dict) -> str:
    _, _, exec_spec, _, _ = _mods(str(root))
    return exec_spec.make_exec_spec(instance).instance_image_key


def test_command(root: Path, instance: dict, test_patch: str) -> str:
    constants, _, _, _, utils = _mods(str(root))
    # Old environments pin old test runners: astropy <= 1.3 has pytest 3.3, where -rA prints no
    # per-test lines; their install spec carries a "test_cmd" that works there (-vv, classic output).
    install = constants.MAP_VERSION_TO_INSTALL.get(instance["repo"], {}).get(instance["version"], {})
    # The verbose variant keeps tracebacks (the official one passes --tb=no),
    # which the agent needs to tell a broken test from a reproduced bug.
    fw_map = getattr(constants, "MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE", constants.MAP_REPO_TO_TEST_FRAMEWORK)
    fw = install.get("test_cmd") or fw_map.get(instance["repo"], {}).get(instance["version"]) or \
        constants.MAP_REPO_TO_TEST_FRAMEWORK[instance["repo"]][instance["version"]]
    fw = fw.replace("--tb=no", "--tb=long")
    return " ".join([fw, *utils.get_test_directives(test_patch, instance["repo"])])


def eval_setup_commands(root: Path, instance: dict) -> list[str]:
    """Per-repo commands the harness runs before the tests (locale, env vars, ...)."""
    constants, *_ = _mods(str(root))
    install = constants.MAP_VERSION_TO_INSTALL[instance["repo"]][instance["version"]]
    return list(install.get("eval_commands", []))


def log_parser(root: Path, repo: str):
    _, _, _, log_parsers, _ = _mods(str(root))
    return log_parsers.MAP_REPO_TO_PARSER[repo]


def docker_health() -> str | None:
    """None when both the docker CLI and the Python SDK (used by the harness) reach the daemon,
    else a message saying what is wrong and what to try."""
    import os
    import subprocess
    r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return ("the docker CLI cannot reach the daemon: " + (r.stderr.strip() or "?") +
                "\n  the daemon may have stopped or crashed (memory?) -- check: docker info; "
                "sudo systemctl status docker; sudo dmesg | grep -iE 'oom|killed' | tail")
    try:
        import docker
        docker.from_env().ping()
    except Exception as e:                       # noqa: BLE001
        host = subprocess.run(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                              capture_output=True, text=True).stdout.strip()
        hint = (f"\n  the CLI uses {host}; the SDK uses DOCKER_HOST={os.environ.get('DOCKER_HOST') or '(unset -> unix:///var/run/docker.sock)'}"
                f"\n  try: export DOCKER_HOST={host}") if host else ""
        return f"the docker Python SDK (used by the harness) cannot reach the daemon: {e}{hint}"
    return None
