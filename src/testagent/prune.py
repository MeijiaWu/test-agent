"""Keep only the tests that reproduce the issue.

The benchmark (and any sensible CI) counts a generated test file as a
reproduction only if *every* test in it passes once the bug is fixed; one
extra test that fails for an unrelated reason sinks the whole file. After
triage, test functions not judged ``reproduces_issue`` are removed from the
file by line range (decorators included), leaving imports, fixtures and
helpers untouched.
"""

from __future__ import annotations

import ast
import re


def func_name(test_id: str) -> str:
    """Test function name from a runner id:
    ``path::Class::test_x[p]`` (pytest), ``test_x (mod.Class)`` (unittest/Django)."""
    tail = test_id.split("::")[-1].split("[")[0]
    m = re.match(r"\s*(\w+)", tail)
    return m.group(1) if m else tail.strip()


def match_id(given: str, ids: list[str]) -> str | None:
    """The status key a model-reported test id refers to (exact, else by function name)."""
    if given in ids:
        return given
    name = func_name(given)
    hits = [i for i in ids if func_name(i) == name]
    return hits[0] if len(hits) == 1 else None


def find_test_ranges(code: str) -> dict[str, list[tuple[int, int]]]:
    """Line ranges (1-based, inclusive) of every test function / method, by name."""
    out: dict[str, list[tuple[int, int]]] = {}
    tree = ast.parse(code)
    nodes = list(tree.body)
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            nodes.extend(n.body)
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            start = min([d.lineno for d in n.decorator_list] + [n.lineno])
            out.setdefault(n.name, []).append((start, n.end_lineno))
    return out


def prune(code: str, keep: set[str]) -> str:
    """``code`` without the test functions whose names are not in ``keep``."""
    drop = [r for name, rs in find_test_ranges(code).items() if name not in keep for r in rs]
    if not drop:
        return code
    # a class left with no statements would not parse: drop the whole class instead
    for n in ast.parse(code).body:
        if isinstance(n, ast.ClassDef):
            body = [(min([d.lineno for d in getattr(c, "decorator_list", [])] + [c.lineno]), c.end_lineno)
                    for c in n.body]
            if body and all(b in drop for b in body):
                start = min([d.lineno for d in n.decorator_list] + [n.lineno])
                drop.append((start, n.end_lineno))
    lines = code.splitlines(keepends=True)
    gone = {i for a, b in drop for i in range(a - 1, b)}
    out = "".join(l for i, l in enumerate(lines) if i not in gone)
    return re.sub(r"\n{4,}", "\n\n\n", out)
