import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toyshop"


@pytest.fixture
def toy_task():
    return {"instance_id": "toyshop-1", "repo": "local/toyshop",
            "problem_statement": (TOY / "ISSUE.md").read_text(), "created_at": None}


GOOD_TEST = '''from shop.paginate import paginate


def test_default_limit_is_ten():
    assert len(paginate(list(range(100)))) == 10
'''
# the toy repository's real fix, as the model would propose it (edit block)
GOOD_FIX = """The default ignores DEFAULT_LIMIT.

shop/paginate.py
<<<<<<< SEARCH
        limit = 50
=======
        limit = DEFAULT_LIMIT
>>>>>>> REPLACE
"""
BROKEN_IMPORT = GOOD_TEST.replace("shop.paginate", "shop.paginat")


def triage_by_name(repro="default_limit", other="wrong_expectation", passes_after_fix=True):
    """Triage reply as a function of the prompt: tests whose name contains ``repro``
    reproduce the issue, every other failing test gets ``other``."""
    def reply(prompt):
        ids = [l[2:] for l in prompt.split("Failing tests (copy these ids exactly):\n", 1)[1]
               .split("\n\n", 1)[0].splitlines() if l.startswith("- ")]
        hit = lambda i: repro in i.split("::")[-1]               # match the function name, not the path
        return json.dumps({"tests": [
            {"test": i, "after_fix": "the default becomes 10, so the assertion holds" if hit(i) else "unchanged",
             "passes_after_fix": passes_after_fix if hit(i) else False,
             "verdict": "reproduces_issue" if hit(i) else other,
             "requirement_id": "R1" if hit(i) else None,
             "rationale": "assert 50 == 10 violates R1" if hit(i) else "not supported by R1"}
            for i in ids]})
    return reply


def script(write_replies, triage_reply=None):
    """Scripted model replies for the toy issue."""
    j = lambda o: json.dumps(o)
    return {
        "analyze_issue": [j({"summary": "paginate ignores the default limit",
                             "requirements": [
                                 {"id": "R1", "kind": "expected", "text": "paginate(items) without limit returns at most DEFAULT_LIMIT (10) items"},
                                 {"id": "R2", "kind": "observed", "text": "it returns 50 items"}],
                             "ambiguities": [], "symbols": ["paginate", "DEFAULT_LIMIT"]})],
        "design_cases": [j({"cases": [{"id": "C1", "requirement_ids": ["R1"], "kind": "reproduce",
                                       "description": "100 items, no limit", "inputs": "paginate(list(range(100)))",
                                       "expected": "10 items"}]})],
        "review_cases": [j({"approved": True, "problems": []})],
        "test_data": [j({"data": [{"case_id": "C1", "values": {"items": "list(range(100))"}}]})],
        "write_tests": [j({"path": "tests/test_default_limit.py", "content": c}) for c in write_replies],
        "triage_failure": [triage_reply or triage_by_name()],
        "bug_report": [j({"title": "paginate default page size is 50, documented as 10",
                          "requirement_ids": ["R1"], "reproduction": "tests/test_default_limit.py",
                          "expected": "10 items", "actual": "50 items",
                          "suspected_locations": ["shop/paginate.py"]})],
        "oneshot": [j({"path": "tests/test_default_limit.py", "content": GOOD_TEST})],
        "candidate_fix": [GOOD_FIX],
    }
