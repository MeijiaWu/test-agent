# testagent — an LLM agent for the testing lifecycle

Given a bug report and a repository, the agent reviews the report, retrieves
the relevant code, designs and reviews test cases, chooses test data, writes
an executable test, runs it in a sandbox, triages the outcome (broken test /
wrong expectation / real bug), repairs or redesigns, and finally files a bug
report linked to the requirement it violates.

It is evaluated on **SWT-bench Lite** (Mündler et al., NeurIPS 2024) with the
**official harness**: a generated test counts only if it fails on the original
code and passes once the real fix is applied. The numbers are therefore
comparable to published methods, and nothing about success is self-reported.

| Testing-lifecycle step | Node | LLM or code |
|---|---|---|
| Requirement review | `analyze` | LLM → numbered requirements (expected / observed / constraint), ambiguities, symbols |
| Context | `retrieve` | code: AST lookup of named symbols + BM25 over the repo; picks the test directory (package-aware: the package the issue names wins) and an example test to imitate |
| Test design | `design` | LLM → cases, each traced to requirement ids |
| Test review | `review` | code rules first (a reproduce case exists, ids valid, expectations present), then an LLM reviewer: is every expectation justified by a requirement rather than by the current (buggy) behaviour? |
| Test data | `data` | LLM proposes values; code checks each is a valid Python expression |
| Automated test | `write` | LLM writes the file; code rejects bad paths, syntax errors and overwriting existing files before anything runs |
| Execution | `execute` | code: Docker sandbox, network off, time limit; failing runs repeated to catch flakiness |
| Result triage | `triage` | code decides the obvious (all passed, collection/import errors, timeouts, flaky); the LLM judges each real failure separately; tests that do not reproduce the issue are cut from the file, which is re-run to confirm |
| Fix check | `verify` | LLM writes a candidate fix (search/replace edit blocks over the real source files) from the issue and code **without seeing the tests**; the tests are re-run with it applied: all pass → confirmed, some → the rest are cut, none → refuted and redesigned, fix unusable → unverified |
| Defect report | `report` | LLM bug report: reproduction, expected vs actual, requirement, suspected code locations |

```
analyze → retrieve → design ⇄ review → data → write → execute → triage ─┬─ budget → report
                       ▲                         ▲                        │
                       ├── wrong_expectation / ──┴──── test_error ────────┤
                       │   not_reproduced                                 │ reproduces_issue
                       └──────────── refuted ──────────── verify (candidate fix) → report
```

## Layout

```
src/testagent/
  agent.py      LangGraph state machine: nodes, routing, budgets, final-attempt choice
  schemas.py    Pydantic schema of every step's output (validated; invalid replies are re-asked)
  llm.py        OpenAI-compatible client (llama-server / vLLM / hosted), JSON extraction, retry, disk cache
  rag.py        AST chunking, identifier-aware BM25, symbol lookup, related-test discovery, optional embedding fusion
  sandbox.py    DockerSandbox (SWT-bench instance images) and LocalSandbox (pytest on a temp copy)
  swt.py        adapter over the official harness: instances, image names, test commands, log parsers
  triage.py     rule-based first pass of triage
  oneshot.py    zero-shot baseline (same context, one call)
  runner.py     one task end to end; withholds gold fields from the agent
  fixcheck.py   candidate fix: search/replace edits, overlay run, confirmed / refuted / unverified
  tracing.py    JSONL event log: nodes, LLM calls (tokens, latency, retries), test runs
  service.py    FastAPI: start runs, fetch results, stream events (SSE)
  cli.py        `testagent demo`, `testagent serve`
bench/          prepare (dataset + dev/test split), build_images, run, evaluate (official harness), report
examples/toyshop/  a small repository with a seeded bug, its issue text and the fix
tests/          42 tests (see "What is tested")
```

## Quick start (toy issue, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# an OpenAI-compatible server, e.g. llama.cpp:
#   llama-server -m <coder-model>.gguf -c 65536 -np 2 --port 8080
testagent demo                                   # or: --set llm.base_url=... --set llm.model=...
```

The demo prints every step, the generated test patch, and whether the test
is fail-to-pass against the toy repository's known fix.

## SWT-bench Lite

Requirements of the harness: Docker, x86_64, ~120 GB disk, 16 GB RAM, 8 cores.

```bash
make swt                       # clone + install the official harness into third_party/swt-bench
make prepare                   # data/swt_lite.jsonl with a fixed dev/test split (20 % dev)
make images SPLIT=dev          # build instance images with the harness's own builder
make run  METHOD=agent   SPLIT=dev        # -> runs/agent_dev/predictions.jsonl, trajectories, traces
make eval RUN=agent_dev                   # official harness -> runs/agent_dev/eval.json
```

`bash scripts/pilot.sh` runs the whole chain on a few dev instances. It also
evaluates each instance's own reference tests (`bench/gold.py`): they are
fail-to-pass by construction, so an instance where they do not resolve is
unsolvable in this setup and is reported separately, not as a method failure.

Methods (`--method`): `agent`, `oneshot`, and the ablations `agent_norag`
(no code context; the test location and one example test are still given),
`agent_noreview`, `agent_noexec` (no execution feedback), `agent_nofix` (no candidate-fix check). Any config value can
be overridden: `--set llm.model=... --set agent.max_exec_rounds=3`.

**Protocol.** Prompts and budgets are tuned on the dev split only. Final
numbers come from one run per method on the test split, then

```bash
make report SPLIT=test         # success rate + bootstrap CI, paired Δ vs one-shot, cost, triage precision/recall
```

`report.py` also lists success by issue year: SWE-bench issues are public and
older than current models, so a model may have seen some fixes. The one-shot
baseline uses the same model and context, so the agent-vs-baseline difference
is not explained by memorisation alone.

## API

```bash
testagent serve --port 8000
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"instance_id": "sympy__sympy-20590"}'                 # or {"repo_dir": "...", "issue": "..."}
curl localhost:8000/runs/<run_id>                               # verdict, patch, bug report, attempts, usage
curl -N localhost:8000/runs/<run_id>/events                     # live event stream
```

## Design notes

- **No leakage.** The agent sees `instance_id, repo, base_commit, version,
  problem_statement, created_at`. Gold patches, the FAIL_TO_PASS lists and the
  issue comments (`hints_text`, which often discuss the fix) are withheld; the
  instance image has no git remote, so newer commits are not reachable.
- **The oracle problem.** A test written by reading buggy code tends to assert
  the bug. Requirements are extracted from the issue first, every case is
  traced to them, and the reviewer rejects expectations that are not justified
  by an "expected" requirement.
- **Judgement only where needed.** Collection errors, import errors, timeouts,
  flakiness and "everything passed" are decided by rules; the LLM is asked only
  about genuine failures. Each call's output is schema-validated.
- **Fail now, pass after the fix.** A failing test is not yet a reproduction. Triage asks,
  per test, what it would do once the issue is fixed exactly as described; a test that would
  still fail (it pins an error id or message the issue never states, or tests a neighbouring
  API) is classed as a wrong expectation, and design/review rules forbid such assertions.
- **An independent oracle for "passes after the fix".** The real fix is withheld, so the agent
  writes its own candidate fix -- from the issue and code only, never shown the tests, so it
  cannot be bent to fit them -- and re-runs the tests with it (`fixcheck.py`). A reproduction
  the fix confirms is preferred; tests that still fail with it are cut; a fully refuted one is
  redesigned. A fix that does not apply or breaks the code says nothing about the test and
  leaves it unverified. The fix never enters the submission; it is attached to the bug report.
- **The issue text reaches the test writer.** Design and writing see the issue verbatim, not only
  the extracted requirements: a reporter's reproduction snippet (e.g. a final `draw()` call that
  actually hits the bug) is easily lost in a paraphrase.
- **A directory where tests cannot run is abandoned.** If two runs in a row in the chosen test
  directory produce no result at all (e.g. it needs a GIS backend), the next candidate directory
  is used.
- **Submit only reproducing tests.** A generated file counts only if every test in it
  passes after the fix, so one unrelated failing test sinks it. Triage judges each
  failing test; the others are removed (`prune.py`) and the pruned file is run again.
  Assertions on docstrings or source text are rejected before anything runs.
- **Same execution as the evaluation.** In-loop test runs use the harness's own
  image, test command, setup commands and log parser (the verbose command
  variant, which keeps tracebacks).
- **Reproducible.** Temperature 0 and an on-disk response cache keyed by the
  full request; every run writes its full trajectory and event log.

## What is tested

`python -m pytest -q` (42 tests, no Docker or model needed):

- the full graph on the toy repository with a scripted model: repair after a
  broken import (caught by rules, no LLM triage), redesign after a review
  rejection, budget exhaustion keeping the best attempt, never overwriting an
  existing file, the no-execution ablation, the one-shot baseline, pruning of
  unrelated failing tests, rejection of docstring assertions;
- the final test is fail-to-pass against the toy fix;
- `DockerSandbox` script assembly, tar streaming, same-shell setup and exit
  codes, using a fake `docker` executable;
- the OpenAI-compatible client against a mock HTTP server, including caching;
- the API (run, poll, event stream), the report script, patches (`git apply`),
  path safety, retrieval, triage rules, config validation.

The SWT-bench adapter was checked against the harness source (test commands,
setup commands, parsers, image keys). **Not exercised here:** real Docker runs
of SWT-bench images and real model quality — that is what `make run` / `make
eval` measure.
