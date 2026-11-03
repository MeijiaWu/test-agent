"""Where generated tests run.

``DockerSandbox`` runs them inside the SWT-bench instance image -- the
repository at its base commit with its environment installed, exactly what
the official evaluation uses -- with the network off and a time limit. The
test files are streamed in as a tar archive, so nothing on the host is
mounted or modified.

``LocalSandbox`` runs pytest on a temporary copy of a local repository; it is
used for the bundled example and the tests, and needs no Docker.

Both return an ``ExecResult`` with per-test statuses. Failing runs are
repeated (``reruns``) and marked inconsistent if the outcome changes, so a
flaky test is never taken as evidence of a bug.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from .patch import new_files_patch, safe_relpath
from .schemas import ExecResult

TAIL_CHARS = 12_000
STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")


def parse_pytest(log: str) -> dict[str, str]:
    """``pytest -rA`` short summary lines: ``PASSED path::test`` / ``FAILED path::test - msg``."""
    out = {}
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        for st in STATUSES:
            if line.startswith(st + " "):
                name = line[len(st) + 1:].split(" - ")[0].strip()
                if name:
                    out[name] = st
    return out


VERBOSE_LINE = re.compile(r"^(\S+::\S+(?:\[.*?\])?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


def parse_any(log: str) -> dict[str, str]:
    """Fallback when the harness parser finds nothing: ``-rA`` summary lines, else
    ``pytest -v`` lines (``path::test STATUS``), as printed by old pytest versions."""
    out = parse_pytest(log)
    if out:
        return out
    for line in log.splitlines():
        m = VERBOSE_LINE.match(re.sub(r"\x1b\[[0-9;]*m", "", line))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def tail(s: str, n: int = TAIL_CHARS) -> str:
    return s if len(s) <= n else "...(truncated)...\n" + s[-n:]


class Sandbox:
    name = "base"

    def __init__(self, timeout: int = 300, reruns: int = 1):
        self.timeout, self.reruns = timeout, reruns

    def _run_once(self, files: dict[str, str], overlay: dict[str, str] | None = None) -> ExecResult:
        raise NotImplementedError

    def run_tests(self, files: dict[str, str], overlay: dict[str, str] | None = None) -> ExecResult:
        """Run the test ``files``. ``overlay`` replaces repository files for this run only
        (a candidate fix); only ``files`` are passed to the test command."""
        files = {safe_relpath(p): c for p, c in files.items()}
        overlay = {safe_relpath(p): c for p, c in (overlay or {}).items()}
        first = self._run_once(files, overlay)
        bad = first.timed_out or any(v in ("FAILED", "ERROR") for v in first.statuses.values()) \
            or (first.exit_code != 0 and not first.statuses)
        if bad:
            for _ in range(self.reruns):
                again = self._run_once(files, overlay)
                if again.statuses != first.statuses or again.timed_out != first.timed_out:
                    first.consistent = False
                    break
        return first

    def export_repo(self, dest: Path) -> Path:
        raise NotImplementedError

    def exists(self, relpath: str) -> bool:
        raise NotImplementedError


class LocalSandbox(Sandbox):
    """pytest on a temporary copy of ``repo_dir``; ``pre_patch`` (a git diff)
    is applied to the copy first -- the local evaluator uses it to apply the fix."""

    name = "local"

    def __init__(self, repo_dir: str | Path, python: str = sys.executable, timeout: int = 120,
                 reruns: int = 1, pre_patch: str | None = None):
        super().__init__(timeout, reruns)
        self.repo_dir = Path(repo_dir).resolve()
        self.python = python
        self.pre_patch = pre_patch

    def exists(self, relpath: str) -> bool:
        return (self.repo_dir / relpath).exists()

    def export_repo(self, dest: Path) -> Path:
        return self.repo_dir

    def _run_once(self, files, overlay=None):
        with tempfile.TemporaryDirectory(prefix="testagent_") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(self.repo_dir, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            if self.pre_patch:
                apply_patch(work, self.pre_patch)
            for rel, content in {**(overlay or {}), **files}.items():
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                (work / rel).write_text(content)
            cmd = [self.python, "-m", "pytest", "-rA", "--tb=long", "-p", "no:cacheprovider", *files]
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(work)}
            try:
                p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                                   timeout=self.timeout, env=env)
                log, code, to = p.stdout + p.stderr, p.returncode, False
            except subprocess.TimeoutExpired as e:
                log = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                code, to = -1, True
        return ExecResult(command=" ".join(cmd[1:]), exit_code=code, timed_out=to,
                          output=tail(log), statuses=parse_pytest(log))


def apply_patch(work: Path, patch: str):
    p = subprocess.run(["git", "apply", "--unsafe-paths", "-"], cwd=work, input=patch,
                       capture_output=True, text=True)
    if p.returncode != 0:
        p2 = subprocess.run(["patch", "-p1", "--forward"], cwd=work, input=patch,
                            capture_output=True, text=True)
        if p2.returncode != 0:
            raise RuntimeError(f"patch failed: {p.stderr}\n{p2.stdout}{p2.stderr}")


class DockerSandbox(Sandbox):
    """Tests inside the SWT-bench instance image (repo at ``/testbed``, conda env ``testbed``)."""

    name = "docker"
    CONDA = "source /opt/miniconda3/bin/activate && conda activate testbed"

    def __init__(self, instance: dict, swt_root: Path, timeout: int = 600, reruns: int = 1,
                 docker: str = "docker"):
        super().__init__(timeout, reruns)
        from . import swt
        self.instance, self.swt_root, self.docker = instance, swt_root, docker
        self.image = swt.image_name(swt_root, instance)
        self._swt = swt
        self._parser = swt.log_parser(swt_root, instance["repo"])
        self._listing: set[str] | None = None

    def ensure_image(self):
        r = subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True, text=True)
        if r.returncode == 0:
            return
        err = (r.stderr or "").strip()
        if "No such image" not in err and "No such object" not in err:
            raise RuntimeError(f"Docker daemon not reachable ({err or 'docker image inspect failed'}) "
                               f"-- check: docker info")
        raise RuntimeError(f"image {self.image} missing -- build it: "
                           f"python bench/build_images.py --ids {self.instance['instance_id']}")

    def export_repo(self, dest: Path) -> Path:
        """Copy ``/testbed`` out of the image once (read-only use: retrieval)."""
        dest = Path(dest)
        if (dest / ".exported").exists():
            return dest
        self.ensure_image()
        dest.mkdir(parents=True, exist_ok=True)
        cid = subprocess.run([self.docker, "create", self.image], capture_output=True, text=True,
                             check=True).stdout.strip()
        try:
            subprocess.run([self.docker, "cp", f"{cid}:/testbed/.", str(dest)], check=True,
                           capture_output=True)
        finally:
            subprocess.run([self.docker, "rm", "-f", cid], capture_output=True)
        (dest / ".exported").write_text(self.image)
        return dest

    def exists(self, relpath: str) -> bool:
        if self._listing is None:
            return False
        return relpath in self._listing

    def set_listing(self, root: Path):
        self._listing = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

    def _run_once(self, files, overlay=None):
        patch = new_files_patch(files)
        test_cmd = self._swt.test_command(self.swt_root, self.instance, patch)
        # the harness's per-repo setup (locale, env vars) must run in this shell, not a subshell
        setup = "; ".join(self._swt.eval_setup_commands(self.swt_root, self.instance)) or "true"
        script = (f"tar -x -C /testbed && {self.CONDA} && cd /testbed && {{ {setup}; }} >/dev/null 2>&1; "
                  f"timeout {self.timeout} {test_cmd}; echo \"__EXIT_CODE__=$?\"")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel, content in {**(overlay or {}), **files}.items():
                data = content.encode()
                info = tarfile.TarInfo(rel)
                info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())   # fresh mtime: no stale .pyc
                tar.addfile(info, io.BytesIO(data))
        cmd = [self.docker, "run", "--rm", "-i", "--network", "none", "--entrypoint", "/bin/bash",
               self.image, "-c", script]
        try:
            p = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, timeout=self.timeout + 120)
            log = p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            return ExecResult(command=test_cmd, exit_code=-1, timed_out=True, output="docker run timed out")
        m = re.search(r"__EXIT_CODE__=(\d+)", log)
        code = int(m.group(1)) if m else p.returncode
        return ExecResult(command=test_cmd, exit_code=code, timed_out=(code == 124),
                          output=tail(log.replace(m.group(0), "") if m else log),
                          statuses=dict(self._parser(log)) or parse_any(log))
