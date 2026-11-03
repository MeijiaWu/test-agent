"""Candidate-fix check: is a failing test fail-to-pass?

A test that fails on the buggy code is a reproduction only if it passes once
the bug is fixed -- which the agent cannot observe, since the real fix is
withheld. Instead the model writes a candidate fix from the issue and the code
alone, *without seeing the tests* (so the fix cannot be tailored to them), and
the tests are run again with that fix applied:

* every kept test passes      -> confirmed
* some pass                   -> the others are cut, and the pruned file is re-checked
* none pass                   -> refuted: the test likely asserts something the issue does
                                 not ask for, or a different code path; it is redesigned
* the fix cannot be applied or breaks the code -> unverified (says nothing about the test)

The fix is a set of search/replace edit blocks (plain text, not JSON: code in
JSON strings is easy to break), applied to the exported repository copy; the
edited files are overlaid on the sandbox for one run. A search text that is
not found exactly is matched ignoring trailing whitespace; if that fails too,
the error shows the closest text in the file so the next try can copy it.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from .patch import safe_relpath
from .rag import is_test_path


class FixError(ValueError):
    pass


BLOCK = re.compile(r"^[ \t]*(?P<path>[^\n]*?\.py)[`*:\s]*\n(?:```[a-z]*\n)?<<<<<<< SEARCH\n(?P<search>.*?)\n?"
                   r"=======\n(?P<replace>.*?)\n?>>>>>>> REPLACE", re.M | re.S)


def parse_edits(text: str):
    """Edit blocks -> [Edit]. Raises ValueError (the model is asked again) when there are none."""
    from .schemas import Edit
    edits = []
    for m in BLOCK.finditer(text):
        path = m.group("path").strip().strip("`*").split()[-1]
        edits.append(Edit(path=path, search=m.group("search"), replace=m.group("replace")))
    if not edits:
        raise ValueError("no edit block found; use: path line, <<<<<<< SEARCH, =======, >>>>>>> REPLACE")
    return edits


def _aligned(text: str, search: str) -> list[int]:
    """Occurrences of ``search`` that do not cut through a word at either end
    (``limit = 5`` must not match inside ``limit = 50``)."""
    word = lambda c: c.isalnum() or c == "_"
    out, i = [], text.find(search)
    while i >= 0:
        end = i + len(search)
        ok_start = i == 0 or not (word(text[i - 1]) and word(search[0]))
        ok_end = end == len(text) or not (word(text[end]) and word(search[-1]))
        if ok_start and ok_end:
            out.append(i)
        i = text.find(search, i + 1)
    return out


def _norm(lines):
    return [l.rstrip() for l in lines]


def _replace_lines(text: str, search: str, replace: str) -> str | None:
    """Replace the unique window of lines equal to ``search`` ignoring trailing whitespace."""
    lines = text.splitlines(keepends=True)
    want = _norm(search.strip("\n").split("\n"))
    have = _norm(lines)
    k = len(want)
    hits = [i for i in range(len(have) - k + 1) if have[i:i + k] == want]
    if len(hits) != 1:
        return None
    i = hits[0]
    rep = replace.strip("\n")
    rep = rep + "\n" if rep else ""
    return "".join(lines[:i]) + rep + "".join(lines[i + k:])


def closest(text: str, search: str, limit: int = 1500) -> str:
    """The window of the file most similar to ``search`` (same number of lines)."""
    lines = text.splitlines()
    k = max(1, len(search.strip("\n").split("\n")))
    target = "\n".join(_norm(search.strip("\n").split("\n")))
    best, best_i = 0.0, 0
    for i in range(0, max(1, len(lines) - k + 1)):
        r = difflib.SequenceMatcher(None, target, "\n".join(_norm(lines[i:i + k])), autojunk=False).quick_ratio()
        if r > best:
            best, best_i = r, i
    if best < 0.5:
        return ""
    lo, hi = max(0, best_i - 2), min(len(lines), best_i + k + 2)
    return f"(lines {lo + 1}-{hi})\n" + "\n".join(lines[lo:hi])[:limit]


def apply_edits(repo_root: Path, edits) -> dict[str, str]:
    """Edited file contents (path -> new text). Raises FixError with a message the model can act on."""
    out: dict[str, str] = {}
    for e in edits:
        try:
            path = safe_relpath(e.path)
        except ValueError as err:
            raise FixError(str(err)) from None
        if is_test_path(path):
            raise FixError(f"{path} is a test file; change library code only")
        f = Path(repo_root) / path
        if path not in out and not f.is_file():
            raise FixError(f"{path} does not exist")
        text = out.get(path) if path in out else f.read_text(errors="replace")
        if not e.search:
            raise FixError(f"empty search text for {path}")
        hits = _aligned(text, e.search)
        n = len(hits)
        if n == 1:
            i = hits[0]
            out[path] = text[:i] + e.replace + text[i + len(e.search):]
            continue
        if n > 1:
            raise FixError(f"search text found {n} times in {path}; include more lines to make it unique")
        fuzzy = _replace_lines(text, e.search, e.replace)
        if fuzzy is not None:
            out[path] = fuzzy
            continue
        near = closest(text, e.search)
        raise FixError(f"search text not found in {path}"
                       + (f"; the closest text in the file is {near}\nCopy the search text exactly from the file."
                          if near else "; it does not exist in the file -- copy existing lines exactly"))
    return out


def diff(repo_root: Path, new: dict[str, str]) -> str:
    chunks = []
    for path, text in new.items():
        old = (Path(repo_root) / path).read_text(errors="replace")
        chunks += difflib.unified_diff(old.splitlines(keepends=True), text.splitlines(keepends=True),
                                       f"a/{path}", f"b/{path}")
    return "".join(chunks)


_BROKEN = re.compile(r"(SyntaxError|IndentationError|NameError|ImportError|ModuleNotFoundError|TypeError)")


def fix_broke_code(output: str, statuses: dict, edited: dict) -> str | None:
    """Evidence that the candidate fix itself is broken -- only lines about the edited files,
    so the fix author never sees the tests' assertions."""
    if not statuses:
        lines = [l for l in output.splitlines() if any(p in l for p in edited) or _BROKEN.search(l)]
        return "\n".join(lines[-20:]) or "nothing ran with the fix applied"
    lines = output.splitlines()
    hits = []
    for i, l in enumerate(lines):
        if any(p in l for p in edited) and i + 1 < len(lines):
            nxt = " ".join(lines[i + 1:i + 4])
            if _BROKEN.search(nxt):
                hits += [l, nxt]
    return "\n".join(hits[-20:]) or None
