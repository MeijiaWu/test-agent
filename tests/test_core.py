import subprocess

import pytest

from conftest import GOOD_TEST, TOY
from testagent.config import load
from testagent.llm import ScriptedLLM, extract_json
from testagent.patch import new_files_patch, safe_relpath
from testagent.rag import RepoIndex, tokenize
from testagent.sandbox import LocalSandbox, parse_pytest
from testagent.schemas import ExecResult, TestFile
from testagent.triage import pre_triage


def test_patch_applies_with_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    patch = new_files_patch({"tests/test_x.py": "a = 1\nb = 2"})
    subprocess.run(["git", "apply", "-"], cwd=tmp_path, input=patch, text=True, check=True)
    assert (tmp_path / "tests/test_x.py").read_text() == "a = 1\nb = 2\n"


@pytest.mark.parametrize("bad", ["../evil.py", "/etc/x.py", "tests/../../x.py", "tests/x.txt"])
def test_paths_outside_repo_are_rejected(bad):
    with pytest.raises(ValueError):
        safe_relpath(bad)


def test_extract_json_tolerates_fences_and_thinking():
    assert extract_json('<think>{"no": 1}</think>sure ```json\n{"a": {"b": "}"}}\n```') == '{"a": {"b": "}"}}'
    assert extract_json('prefix {"a": "x\\"}"} suffix') == '{"a": "x\\"}"}'


def test_structured_retries_after_invalid_reply():
    llm = ScriptedLLM(script={"t": ["not json", '{"path": "tests/test_a.py"}',
                                    '{"path": "tests/test_a.py", "content": "x=1"}']})
    tf = llm.structured("t", "write", TestFile)
    assert tf.content == "x=1" and llm.calls[0].attempts == 3


def test_tokenize_splits_identifiers():
    t = tokenize("HttpRequest parse_qsl")
    assert {"http", "request", "parse", "qsl", "httprequest"} <= set(t)


def test_rag_lookup_search_and_related_tests():
    idx = RepoIndex.build(TOY / "repo")
    assert [c.path for c in idx.lookup("paginate")] == ["shop/paginate.py"]
    assert idx.lookup("shop.paginate.page_count")[0].name == "page_count"
    assert idx.search("default page size limit", k=3)[0].path == "shop/paginate.py"
    assert idx.related_tests(["shop/paginate.py"], ["paginate"])[0] == "tests/test_paginate.py"


def test_local_sandbox_fail_pass_and_broken():
    sb = LocalSandbox(TOY / "repo")
    r = sb.run_tests({"tests/test_new.py": GOOD_TEST})
    assert list(r.statuses.values()) == ["FAILED"] and "assert 50 == 10" in r.output and r.consistent
    fixed = LocalSandbox(TOY / "repo", pre_patch=(TOY / "fix.patch").read_text())
    assert list(fixed.run_tests({"tests/test_new.py": GOOD_TEST}).statuses.values()) == ["PASSED"]
    broken = sb.run_tests({"tests/test_new.py": "import nope\n\ndef test_x():\n    pass\n"})
    assert pre_triage(broken).verdict == "test_error"


def test_pre_triage_rules():
    ok = ExecResult(command="", exit_code=0, output="", statuses={"t": "PASSED"})
    assert pre_triage(ok).verdict == "not_reproduced"
    flaky = ExecResult(command="", exit_code=1, output="", statuses={"t": "FAILED"}, consistent=False)
    assert pre_triage(flaky).verdict == "test_error"
    real = ExecResult(command="", exit_code=1, output="AssertionError", statuses={"t": "FAILED"})
    assert pre_triage(real) is None            # needs judgement


def test_parse_pytest_summary():
    log = "FAILED tests/a.py::test_x - assert 1 == 2\nPASSED tests/a.py::test_y\n"
    assert parse_pytest(log) == {"tests/a.py::test_x": "FAILED", "tests/a.py::test_y": "PASSED"}


def test_config_rejects_unknown_keys():
    cfg = load(overrides=["agent.use_rag=false", "sandbox.timeout=30"])
    assert cfg.agent.use_rag is False and cfg.sandbox.timeout == 30
    with pytest.raises(KeyError):
        load(overrides=["agent.use_ragg=false"])


def test_prune_keeps_helpers_and_drops_unrelated_tests():
    from testagent.prune import func_name, match_id, prune
    code = (
        "import pytest\n\n\n@pytest.fixture\ndef items():\n    return [1]\n\n\n"
        "def test_keep(items):\n    assert items\n\n\n@pytest.mark.slow\ndef test_drop():\n    assert 0\n\n\n"
        "class TestOnlyDrop:\n    def test_x(self):\n        assert 0\n\n\n"
        "class TestMixed:\n    def helper(self):\n        return 1\n\n    def test_y(self):\n        assert 0\n")
    out = prune(code, {"test_keep"})
    import ast
    ast.parse(out)
    assert "def items" in out and "def test_keep" in out and "helper" in out
    assert "test_drop" not in out and "mark.slow" not in out and "TestOnlyDrop" not in out and "test_y" not in out
    assert func_name("tests/a.py::TestX::test_y[1-2]") == "test_y"
    assert func_name("test_y (auth_tests.test_x.TestX)") == "test_y"
    assert match_id("test_y", ["tests/a.py::TestX::test_y"]) == "tests/a.py::TestX::test_y"


def test_related_tests_prefers_the_package_the_issue_names(tmp_path):
    from testagent.rag import RepoIndex
    files = {
        "django/contrib/syndication/views.py": "class Feed:\n    def item_extra_kwargs(self, item):\n        return {}\n",
        "django/contrib/gis/feeds.py": "class Feed:\n    def item_extra_kwargs(self, item):\n        return {}\n",
        "django/contrib/auth/admin.py": "class UserAdmin:\n    pass\n",
        "tests/auth_tests/test_views.py": "def test_login():\n    assert True\n",
        "tests/gis_tests/test_feeds.py": "from django.contrib.gis import feeds\n\ndef test_geo_feed():\n    assert feeds\n",
        "tests/syndication_tests/tests.py": "from django.contrib.syndication import views\n\n"
                                            "def test_rss():\n    assert views.Feed\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    idx = RepoIndex.build(tmp_path)
    code = ["django/contrib/gis/feeds.py", "django/contrib/syndication/views.py", "django/contrib/auth/admin.py"]
    got = idx.related_tests(code, ["item_extra_kwargs", "syndication.views"],
                            issue_text="Add support for item_comments to syndication framework")
    assert got[0] == "tests/syndication_tests/tests.py"


def test_fixture_error_detects_broken_setup_or_cleanup():
    from testagent.triage import fixture_error
    tid = "test_watch (utils_tests.test_x.T)"
    both = "FAIL: test_watch (utils_tests.test_x.T)\nAssertionError\nERROR: test_watch (utils_tests.test_x.T)\nOSError: Directory not empty\n"
    assert fixture_error(both, tid)
    assert fixture_error("FAIL: test_watch (utils_tests.test_x.T)\nAssertionError\n", tid) is None
    assert fixture_error("ERROR at teardown of test_watch\n", "pkg/tests/test_x.py::test_watch")
    assert fixture_error("FAILED pkg/tests/test_x.py::test_watch - assert 1 == 2\n", "pkg/tests/test_x.py::test_watch") is None


def test_apply_edits_rules():
    import pytest
    from conftest import TOY
    from testagent.fixcheck import FixError, apply_edits
    from testagent.schemas import Edit
    root = TOY / "repo"
    out = apply_edits(root, [Edit(path="shop/paginate.py", search="limit = 50", replace="limit = DEFAULT_LIMIT")])
    assert "limit = DEFAULT_LIMIT" in out["shop/paginate.py"]
    assert "limit = 50" in (root / "shop/paginate.py").read_text()          # the repo itself is untouched
    for e, msg in [(Edit(path="tests/test_paginate.py", search="x", replace="y"), "test file"),
                   (Edit(path="shop/nope.py", search="x", replace="y"), "does not exist"),
                   (Edit(path="shop/paginate.py", search="nothing like this", replace="y"), "not found"),
                   (Edit(path="shop/paginate.py", search="limit", replace="y"), "times"),
                   (Edit(path="../etc/passwd.py", search="x", replace="y"), "invalid")]:
        with pytest.raises(FixError, match=msg):
            apply_edits(root, [e])


def test_parse_edits_and_tolerant_matching():
    import pytest
    from conftest import TOY
    from testagent.fixcheck import FixError, apply_edits, parse_edits
    reply = ("Here is the fix.\n\n```\nshop/paginate.py\n<<<<<<< SEARCH\n"
             "        limit = 50   \n=======\n        limit = DEFAULT_LIMIT\n>>>>>>> REPLACE\n```\n")
    edits = parse_edits(reply)
    assert [e.path for e in edits] == ["shop/paginate.py"]
    out = apply_edits(TOY / "repo", edits)            # trailing spaces in the search text are tolerated
    assert "        limit = DEFAULT_LIMIT\n" in out["shop/paginate.py"]
    with pytest.raises(ValueError):
        parse_edits("no blocks here")
    wrong = parse_edits("shop/paginate.py\n<<<<<<< SEARCH\n    if limit is None:\n        limit = 51\n"
                        "=======\nx\n>>>>>>> REPLACE")
    with pytest.raises(FixError, match="closest text"):
        apply_edits(TOY / "repo", wrong)
