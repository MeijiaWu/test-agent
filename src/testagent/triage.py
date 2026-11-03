"""Rule-based first pass of failure triage.

Outcomes that need no judgement are decided here, so the LLM is only asked
about genuine assertion / exception failures:

* nothing failed                          -> not_reproduced
* timed out or flaky                      -> test_error
* the file never imported (collection / syntax / import / fixture errors)
                                          -> test_error
Everything else returns ``None``: an LLM decides between reproduces_issue,
wrong_expectation and test_error.
"""

from __future__ import annotations

import re

from .schemas import ExecResult, Triage

COLLECTION = re.compile(
    r"(ERROR collecting|errors? during collection|ImportError: Failed to import test module|"
    r"ImportError while importing test module|SyntaxError|IndentationError|"
    r"fixture '[^']+' not found|no tests ran|collected 0 items|Ran 0 tests)", re.I)


def pre_triage(r: ExecResult) -> Triage | None:
    if r.timed_out:
        return Triage(verdict="test_error", rationale="the test run timed out")
    if not r.consistent:
        return Triage(verdict="test_error", rationale="outcome changed between runs (flaky)")
    failing = {k: v for k, v in r.statuses.items() if v in ("FAILED", "ERROR")}
    if r.statuses and not failing:
        return Triage(verdict="not_reproduced", rationale="all tests passed on the current code")
    if not r.statuses:
        if r.exit_code == 0:
            return Triage(verdict="test_error", rationale="no tests were collected")
        m = COLLECTION.search(r.output)
        why = m.group(0) if m else f"exit code {r.exit_code} and no per-test result could be read " \
            "from the output -- read the output: the file may have crashed, or a test failed"
        return Triage(verdict="test_error", rationale=f"the test file did not run: {why}")
    if all(v == "ERROR" for v in failing.values()) and COLLECTION.search(r.output):
        return Triage(verdict="test_error", rationale="errors during collection / setup")
    return None


def fixture_error(output: str, test_id: str) -> str | None:
    """The test's own setup / cleanup failed besides its assertion -- it would fail after any fix.
    unittest reports such a test twice (``FAIL: name`` and ``ERROR: name``); pytest reports
    ``ERROR at setup of`` / ``ERROR at teardown of``."""
    name = test_id.split("::")[-1].split(" (")[0].split("[")[0]
    n = re.escape(name)
    if re.search(rf"^FAIL: {n}\b", output, re.M) and re.search(rf"^ERROR: {n}\b", output, re.M):
        return "the test also errors in its own setup or cleanup (reported as both FAIL and ERROR)"
    m = re.search(rf"ERROR at (setup|teardown) of \S*{n}\b", output)
    if m:
        return f"the test errors at {m.group(1)}"
    return None
