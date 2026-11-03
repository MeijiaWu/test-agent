"""Prompt templates, one per agent step. Kept in one place so prompt changes
are reviewable as a diff, and tuned only on the dev split."""

ANALYZE = """You are reviewing a bug report before writing tests for it.

Repository: {repo}

Issue:
<<<
{issue}
>>>

Break the issue into atomic, checkable requirements:
- "expected": what the software SHOULD do (from the reporter's description, documentation
  quoted in the issue, or behaviour the issue treats as correct);
- "observed": the wrong behaviour that is reported;
- "constraint": inputs, versions, settings needed to trigger it.
Do not invent requirements the issue does not support; put open questions in "ambiguities".
List the functions / classes / modules the issue mentions in "symbols", spelled as in code."""

DESIGN = """Design test cases for this issue.

Issue (verbatim):
<<<
{issue}
>>>

Requirements:
{requirements}

Ambiguities noted in review: {ambiguities}

Relevant code:
{context}

{feedback}
Rules:
- At least one case of kind "reproduce": it must exercise the reported scenario and FAIL on the
  current code because the expected behaviour is not met.
- If the issue shows code or steps that trigger the problem, the reproduce case follows them: the
  same calls in the same order with the same kind of inputs (including steps such as drawing,
  saving or calling the method that actually hits the bug), adapted to the test framework.
- Test observable behaviour only: return values, raised exceptions, state changes, output.
  Never assert on docstrings, comments or source text -- a fix does not change those.
- Every expected outcome must follow from an "expected" requirement. Never derive it from what the
  current code does -- the current code is the buggy one.
- Exercise the code path the issue is about, through the entry point it names (the class,
  function, option or command in the issue). Do not mock or patch that code, and do not switch to a
  different API just to obtain a failure.
- Assert only what the issue specifies. Details it leaves open -- error ids or codes, exact message
  wording, output formatting, which exception subclass -- must not be asserted exactly; check them
  loosely (e.g. that an error is reported and mentions the field), or not at all.
- Build inputs that a fix would reject (an invalid model, a bad setting) inside the test and isolate
  them with the framework's facilities (e.g. Django's @isolate_apps) -- never at module level, where
  after the fix they would break the whole test run.
- 1 to 4 cases; prefer one precise reproduction over many vague ones."""

REVIEW = """Review these test cases against the requirements.

Requirements:
{requirements}

Test cases:
{cases}

Check each case:
1. Is its expected outcome justified by an "expected" requirement (not by current behaviour)?
2. Does at least one "reproduce" case trigger the reported scenario, through the code path the
   issue names (not a neighbouring API, not with that code mocked out)?
3. Does any expectation pin a detail the issue does not specify (an error id, exact message text,
   formatting)? Then it would fail even after a correct fix -- flag it.
4. Are inputs concrete enough to implement? Any duplicates?
Approve only if all checks pass; otherwise list the problems per case."""

DATA = """Choose concrete input values for each test case.

Test cases:
{cases}

Relevant code:
{context}

For every case give each input as a Python expression (e.g. "[1, 2, 3]", "'2021-01-01'",
"None"). Use boundary values where the case is about a boundary."""

WRITE = """Write ONE new test file that implements these test cases.

Repository: {repo}
Issue (verbatim):
<<<
{issue}
>>>

Requirements:
{requirements}

Test cases and their data:
{cases}

Relevant code:
{context}

An existing test file in the target directory ({target_dir}) -- follow its framework, base
classes, imports and helpers exactly:
{example}

{feedback}
Rules:
- "path" must be a NEW file under {target_dir}, named test_<something>.py.
- The tests must FAIL on the current code and PASS once the issue is fixed: assert the expected
  behaviour directly. Do not catch the failure, do not mark tests xfail/skip.
- Assert behaviour (results, exceptions, state), never docstrings, comments or source text.
- Exercise the code path the issue is about, through the entry point it names (the class,
  function, option or command in the issue). Do not mock or patch that code, and do not switch to a
  different API just to obtain a failure.
- Assert only what the issue specifies. Details it leaves open -- error ids or codes, exact message
  wording, output formatting, which exception subclass -- must not be asserted exactly; check them
  loosely (e.g. that an error is reported and mentions the field), or not at all.
- Build inputs that a fix would reject (an invalid model, a bad setting) inside the test and isolate
  them with the framework's facilities (e.g. Django's @isolate_apps) -- never at module level, where
  after the fix they would break the whole test run.
- No network access; temporary files only via the framework's tmp facilities.
- Import from the package under test the same way the example does.
- Where the issue shows reproducing code, keep its decisive steps; do not replace them with
  shortcuts that skip the code path the issue is about."""

TRIAGE = """Decide, for EACH failing test, what its failure means.

Requirements:
{requirements}

Test file {path}:
```python
{code}
```

Test run (command: {command}, exit code {exit_code}):
statuses: {statuses}
output (tail):
<<<
{output}
>>>

Failing tests (copy these ids exactly):
{failing}

For each failing test, first imagine the issue fixed exactly as described -- no more. Would every
assertion in the test then hold? A test that fails now but would also fail after that fix is NOT a
reproduction: typical causes are asserting a detail the issue does not specify (an error id, exact
message wording, formatting), or testing a neighbouring API that the fix does not touch.

Verdicts:
- "reproduces_issue": the test fails BECAUSE the software violates an "expected" requirement in
  the way the issue reports, and it would pass once the issue is fixed. Cite the requirement.
- "wrong_expectation": the test asserts something the requirements do not support, or something a
  fix would not change (documentation text, an unspecified detail, another API) -- it would still
  fail after the fix.
- "test_error": the test itself is broken (bad import, wrong API usage, typo, setup error).
Give one entry per failing test, quoting the decisive line of output."""

FIX = """Fix this issue in {repo} with the smallest change to the library code. You will not see
any tests; make the software behave as the issue expects, nothing more.

Issue:
<<<
{issue}
>>>

Requirements extracted from it:
{requirements}

Source of the most relevant files (copy search text from here):
{context}

{feedback}
Answer with one or more edit blocks, nothing else is needed. Each block is the file path on its own
line, then the exact current text and its replacement:

path/to/module.py
<<<<<<< SEARCH
the current lines, copied exactly (indentation included), enough of them to be unique
=======
the new lines
>>>>>>> REPLACE

Only library files, never tests. To add code, SEARCH an existing line next to the insertion point
and REPLACE it with that line plus the new code. Keep the code style."""

REPORT = """Write a concise bug report from this reproduction.

Requirements:
{requirements}

Reproducing test file {path}:
```python
{code}
```

Failure output (tail):
<<<
{output}
>>>

Code locations retrieved during analysis: {locations}"""

ONESHOT = """Write ONE new test file that reproduces this issue: the tests must FAIL on the
current code and PASS once the issue is fixed.

Repository: {repo}
Issue:
<<<
{issue}
>>>

Relevant code:
{context}

An existing test file in {target_dir} -- follow its framework and imports:
{example}

"path" must be a NEW file under {target_dir}, named test_<something>.py.
Test observable behaviour only, never docstrings or source text.
- Exercise the code path the issue is about, through the entry point it names (the class,
  function, option or command in the issue). Do not mock or patch that code, and do not switch to a
  different API just to obtain a failure.
- Assert only what the issue specifies. Details it leaves open -- error ids or codes, exact message
  wording, output formatting, which exception subclass -- must not be asserted exactly; check them
  loosely (e.g. that an error is reported and mentions the field), or not at all.
- Build inputs that a fix would reject (an invalid model, a bad setting) inside the test and isolate
  them with the framework's facilities (e.g. Django's @isolate_apps) -- never at module level, where
  after the fix they would break the whole test run."""
