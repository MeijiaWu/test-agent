"""DockerSandbox's script assembly, tar streaming and exit-code parsing, checked
with a fake ``docker`` executable that runs the container script on the host
against a copy of the toy repository (no Docker daemon needed)."""
import stat
import sys
import textwrap

from conftest import GOOD_TEST, TOY
from testagent import swt
from testagent.sandbox import DockerSandbox, parse_pytest

FAKE = textwrap.dedent(f"""\
    #!{sys.executable}
    import os, shutil, subprocess, sys, tempfile
    a = sys.argv[1:]
    if a[:2] == ["image", "inspect"]:
        sys.exit(0)
    assert a[0] == "run" and "--network" in a and a[a.index("--network") + 1] == "none"
    script = a[a.index("-c") + 1]
    tb = tempfile.mkdtemp()
    shutil.copytree(os.environ["FAKE_TESTBED"], tb, dirs_exist_ok=True)
    script = script.replace("source /opt/miniconda3/bin/activate && conda activate testbed", "true")
    script = script.replace("/testbed", tb)
    sys.exit(subprocess.run(["bash", "-c", script], stdin=sys.stdin).returncode)
""")


def test_docker_sandbox_runs_streamed_tests(tmp_path, monkeypatch):
    fake = tmp_path / "docker"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_TESTBED", str(TOY / "repo"))
    monkeypatch.setattr(swt, "image_name", lambda root, inst: "exec.eval.fake:latest")
    monkeypatch.setattr(swt, "log_parser", lambda root, repo: parse_pytest)
    monkeypatch.setattr(swt, "eval_setup_commands", lambda root, inst: ["export TOY_FLAG=1"])
    # the setup must run in the same shell as the tests: the test checks the variable
    monkeypatch.setattr(swt, "test_command",
                        lambda root, inst, patch: f"{sys.executable} -m pytest -rA -p no:cacheprovider tests/test_new.py")
    sb = DockerSandbox({"instance_id": "toy", "repo": "local/toy"}, swt_root=tmp_path, docker=str(fake), timeout=60)
    env_check = GOOD_TEST + "\n\ndef test_setup_ran():\n    import os\n    assert os.environ.get('TOY_FLAG') == '1'\n"
    r = sb.run_tests({"tests/test_new.py": env_check})
    assert r.statuses == {"tests/test_new.py::test_default_limit_is_ten": "FAILED",
                          "tests/test_new.py::test_setup_ran": "PASSED"}
    assert r.exit_code == 1 and "__EXIT_CODE__" not in r.output and r.consistent


def test_ensure_image_tells_missing_image_from_dead_daemon(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(swt, "image_name", lambda root, inst: "exec.eval.fake:latest")
    monkeypatch.setattr(swt, "log_parser", lambda root, repo: parse_pytest)
    cases = {"Error: No such image: exec.eval.fake:latest": "missing -- build it",
             "Cannot connect to the Docker daemon at unix:///var/run/docker.sock": "daemon not reachable"}
    for i, (stderr, expect) in enumerate(cases.items()):
        fake = tmp_path / f"docker{i}"
        fake.write_text(f"#!/bin/sh\necho '{stderr}' >&2\nexit 1\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        sb = DockerSandbox({"instance_id": "toy", "repo": "local/toy"}, swt_root=tmp_path, docker=str(fake))
        with pytest.raises(RuntimeError, match=expect):
            sb.ensure_image()


def test_overlay_replaces_source_files_but_only_tests_are_run(tmp_path, monkeypatch):
    fake = tmp_path / "docker"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_TESTBED", str(TOY / "repo"))
    monkeypatch.setattr(swt, "image_name", lambda root, inst: "exec.eval.fake:latest")
    monkeypatch.setattr(swt, "log_parser", lambda root, repo: parse_pytest)
    monkeypatch.setattr(swt, "eval_setup_commands", lambda root, inst: [])
    seen = []

    def cmd(root, inst, patch):
        seen.append(patch)
        return f"{sys.executable} -m pytest -rA -p no:cacheprovider tests/test_new.py"
    monkeypatch.setattr(swt, "test_command", cmd)
    sb = DockerSandbox({"instance_id": "toy", "repo": "local/toy"}, swt_root=tmp_path, docker=str(fake), timeout=60)
    src = (TOY / "repo" / "shop" / "paginate.py").read_text().replace("limit = 50", "limit = DEFAULT_LIMIT")
    r = sb.run_tests({"tests/test_new.py": GOOD_TEST}, overlay={"shop/paginate.py": src})
    assert r.statuses == {"tests/test_new.py::test_default_limit_is_ten": "PASSED"}
    assert "shop/paginate.py" not in seen[0]                 # the fix is overlaid, not a test directive
