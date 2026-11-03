"""Structured outputs of every agent step.

Each LLM call is asked for JSON matching one of these models and the reply is
validated; a reply that does not validate is sent back with the validation
error (llm.py). Fields carry short descriptions because they are rendered
into the prompt as the expected schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------ 1. issue review
class Requirement(BaseModel):
    id: str = Field(description="R1, R2, ...")
    text: str = Field(description="one checkable statement")
    kind: Literal["expected", "observed", "constraint"] = Field(
        description="expected = what the software should do; observed = the reported "
                    "wrong behaviour; constraint = context (versions, inputs, environment)")


class IssueAnalysis(BaseModel):
    summary: str = Field(description="one sentence")
    requirements: list[Requirement]
    ambiguities: list[str] = Field(default_factory=list,
                                   description="what the issue leaves unspecified")
    symbols: list[str] = Field(default_factory=list,
                               description="functions / classes / modules the issue mentions")


# --------------------------------------------------------- 3. test design
class TestCase(BaseModel):
    __test__ = False          # not a pytest test class
    id: str = Field(description="C1, C2, ...")
    requirement_ids: list[str] = Field(description="requirements this case checks")
    kind: Literal["reproduce", "boundary", "regression"] = Field(
        description="reproduce = triggers the reported bug")
    description: str
    inputs: str = Field(description="how the code under test is called / set up")
    expected: str = Field(description="expected outcome, derived from the requirements")


class TestPlan(BaseModel):
    __test__ = False          # not a pytest test class
    cases: list[TestCase]


# --------------------------------------------------------- 4. test review
class CaseProblem(BaseModel):
    case_id: str
    problem: str


class CaseReview(BaseModel):
    approved: bool
    problems: list[CaseProblem] = Field(default_factory=list)


# ------------------------------------------------------------ 5. test data
class CaseData(BaseModel):
    case_id: str
    values: dict[str, str] = Field(
        description="name -> Python expression for each concrete input value")


class TestData(BaseModel):
    __test__ = False          # not a pytest test class
    data: list[CaseData]


# ------------------------------------------------------------ 6. test code
class TestFile(BaseModel):
    __test__ = False          # not a pytest test class
    path: str = Field(description="repository-relative path of a NEW test file")
    content: str = Field(description="complete file content")


# ------------------------------------------------------------- 7. execution
class ExecResult(BaseModel):
    command: str
    exit_code: int
    timed_out: bool = False
    output: str = Field(description="combined stdout/stderr, tail")
    statuses: dict[str, str] = Field(default_factory=dict,
                                     description="test id -> PASSED / FAILED / ERROR / SKIPPED")
    consistent: bool = True


# --------------------------------------------------------------- 8. triage
Verdict = Literal["reproduces_issue", "wrong_expectation", "test_error", "not_reproduced"]


class Triage(BaseModel):
    """Overall outcome of one test run (decided by rules or aggregated from TestVerdicts)."""
    verdict: Verdict
    requirement_id: str | None = None
    rationale: str
    keep: list[str] = Field(default_factory=list, description="test functions that reproduce the issue")


class TestVerdict(BaseModel):
    __test__ = False          # not a pytest test class
    test: str = Field(description="failing test id, copied exactly from the list")
    after_fix: str = Field(description="what this test would do once the issue is fixed exactly as the "
                                       "issue describes, and why (which assertion would then hold or still fail)")
    passes_after_fix: bool = Field(description="true only if every assertion in this test would hold after that fix")
    verdict: Literal["reproduces_issue", "wrong_expectation", "test_error"]
    requirement_id: str | None = Field(default=None,
                                       description="requirement the failure demonstrates")
    rationale: str


class FailureTriage(BaseModel):
    tests: list[TestVerdict] = Field(description="one verdict per failing test")


# --------------------------------------------------------------- 9. report
class BugReport(BaseModel):
    title: str
    requirement_ids: list[str]
    reproduction: str = Field(description="test file and test names that reproduce it")
    expected: str
    actual: str
    suspected_locations: list[str] = Field(default_factory=list)


# ------------------------------------------------------ 9. candidate fix
class Edit(BaseModel):
    """One search/replace edit of a candidate fix (parsed from edit blocks, see fixcheck.py)."""
    path: str = Field(description="library source file to change (never a test file)")
    search: str = Field(description="text copied exactly from the current file, unique in it")
    replace: str = Field(description="what that text becomes")
