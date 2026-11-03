"""Retrieval over the repository under test.

Two kinds of lookup, because they answer different questions:

* structural -- the issue names a function, class or module; find its
  definition (and its existing tests) exactly, through the AST;
* lexical (BM25) -- free-text search for code the issue describes without
  naming it. Identifiers are split (``parse_qsl`` -> parse, qsl;
  ``HttpRequest`` -> http, request) so prose and code meet in one vocabulary.

Optional semantic re-ranking with a sentence-transformers model is fused by
reciprocal rank when ``embed_model`` is given; BM25 alone is the default and
what the tests exercise.

Search is exhaustive (every chunk is scored). For one repository -- tens of
thousands of chunks -- that is milliseconds; an approximate index (FAISS /
HNSW) only pays off at millions of chunks.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".tox", ".venv", "venv", "node_modules", "build", "dist",
             "__pycache__", ".eggs", ".mypy_cache", ".pytest_cache"}
MAX_FILE_BYTES = 400_000


@dataclass
class Chunk:
    path: str            # repo-relative, posix
    name: str            # qualified: Class.method, function, <module>
    start: int           # 1-based line numbers
    end: int
    text: str
    is_test: bool

    def header(self) -> str:
        return f"{self.path}:{self.start}-{self.end}  {self.name}"


def is_test_path(p: str) -> bool:
    parts = p.split("/")
    fname = parts[-1]
    return (fname.startswith("test_") or fname.endswith("_test.py") or fname.endswith("_tests.py")
            or any(d in ("tests", "test", "testing") for d in parts[:-1]))


_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        low = word.lower()
        out.append(low)
        parts = [p.lower() for piece in word.split("_") for p in _SPLIT.findall(piece)]
        if len(parts) > 1:
            out.extend(parts)
    return out


def _chunks_of_file(rel: str, src: str) -> list[Chunk]:
    lines = src.splitlines()
    test = is_test_path(rel)
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return [Chunk(rel, f"<lines {i + 1}>", i + 1, min(i + 80, len(lines)),
                      "\n".join(lines[i:i + 80]), test) for i in range(0, max(len(lines), 1), 80)]
    out = []
    doc = ast.get_docstring(tree)
    head_end = min(len(lines), 40)
    out.append(Chunk(rel, "<module>", 1, head_end, (doc or "") + "\n" + "\n".join(lines[:head_end]), test))

    def visit(node, prefix):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                end = getattr(child, "end_lineno", child.lineno)
                out.append(Chunk(rel, name, start, end, "\n".join(lines[start - 1:end]), test))
                if isinstance(child, ast.ClassDef):
                    visit(child, name + ".")
    visit(tree, "")
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.post = defaultdict(list)                    # term -> doc ids (inverted index)
        for i, d in enumerate(self.tf):
            for t in d:
                self.post[t].append(i)

    def scores(self, query: list[str]) -> dict[int, float]:
        s: dict[int, float] = defaultdict(float)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass
class RepoIndex:
    root: Path
    chunks: list[Chunk] = field(default_factory=list)
    embed_model: str | None = None

    @classmethod
    def build(cls, root: str | Path, embed_model: str | None = None) -> "RepoIndex":
        root = Path(root)
        idx = cls(root=root, embed_model=embed_model)
        for p in sorted(root.rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                src = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            idx.chunks.extend(_chunks_of_file(rel, src))
        idx._bm25 = BM25([tokenize(c.name + " " + c.path + " " + c.text) for c in idx.chunks])
        idx._emb = None
        return idx

    # ----------------------------------------------------------- structural
    def lookup(self, symbol: str) -> list[Chunk]:
        """Definitions matching ``symbol``: a bare name, ``Class.method``, or a dotted module path."""
        sym = symbol.strip().strip("`()")
        if not sym:
            return []
        code = [c for c in self.chunks if not c.is_test and c.name != "<module>"]
        hits = [c for c in code if c.name == sym or c.name.endswith("." + sym)]
        if not hits and "." in sym:                       # pkg.module.func / module.Class.method
            head, last = sym.rsplit(".", 1)
            path_part = head.replace(".", "/")
            hits = [c for c in code if c.name.split(".")[-1] == last and
                    (path_part in c.path or c.name.startswith(head.split(".")[-1] + "."))]
        if not hits and "." not in sym:
            hits = [c for c in code if c.name.split(".")[-1] == sym]
        if not hits and "." in sym:                       # module path, e.g. django.utils.http
            mod = sym.replace(".", "/")
            hits = [c for c in self.chunks if c.name == "<module>" and
                    (c.path.endswith(mod + ".py") or c.path.endswith(mod + "/__init__.py"))]
        return hits[:8]

    # directory and module names too common to say where a test belongs
    GENERIC = {"src", "lib", "core", "utils", "util", "contrib", "common", "base", "internal", "tests",
               "views", "models", "admin", "forms", "fields", "api", "main", "helpers", "__init__",
               "management", "commands", "backends", "tests"}

    def related_tests(self, code_paths: list[str], symbols: list[str], k: int = 3,
                      issue_text: str = "") -> list[str]:
        """Existing test files most likely to cover ``code_paths`` / ``symbols``.

        ``code_paths`` are in retrieval order (most relevant first). A test file scores for
        sitting in a directory named after the code's package (``syndication`` ->
        ``tests/syndication_tests``), for being named after the module (``test_<module>.py``;
        generic module names like ``views`` only count together with the package), and for
        mentioning the issue's symbols and the package. Packages the issue itself names
        (``syndication``) weigh three times as much as packages that only came up in retrieval."""
        named = set(tokenize(" ".join(symbols) + " " + issue_text))
        paths = list(dict.fromkeys(code_paths))
        pkgs, mods = {}, {}
        for r, p in enumerate(paths):
            w = 1.0 if r < 3 else 0.4
            parts = Path(p).with_suffix("").parts
            top = parts[0] if len(parts) > 1 else ""
            pkg = next((d for d in reversed(parts[:-1]) if d not in self.GENERIC and d != top), None)
            if pkg and pkg.lower() in named:
                w *= 3
            if pkg:
                pkgs[pkg.lower()] = max(pkgs.get(pkg.lower(), 0), w)
            mods[(parts[-1].lower(), (pkg or "").lower())] = w
        names = {s.split(".")[-1].strip("()").lower() for s in symbols if s}
        files: dict[str, set] = {}
        for c in self.chunks:                      # score files, not chunks: long files get no extra weight
            if c.is_test:
                files.setdefault(c.path, set()).update(tokenize(c.text))
        score: Counter = Counter()
        for path, toks in files.items():
            tp = Path(path)
            dirs = [d.lower() for d in tp.parent.parts]
            fname = tp.stem.lower()
            for pkg, w in pkgs.items():
                if any(d == pkg or d in (f"{pkg}_tests", f"test_{pkg}", f"tests_{pkg}", f"{pkg}tests")
                       for d in dirs):
                    score[path] += 4 * w
                if pkg in toks:
                    score[path] += 2 * w
            for (m, pkg), w in mods.items():
                if fname in (f"test_{m}", f"{m}_test", f"{m}_tests", f"tests_{m}"):
                    generic = m in self.GENERIC
                    in_pkg = bool(pkg) and any(pkg in d for d in dirs)
                    score[path] += (5 if not generic or in_pkg else 1) * w
            score[path] += sum(1 for n in names if n in toks)
            score[path] += sum(0.5 for m, _ in mods if m not in self.GENERIC and m in toks)
        return [p for p, s in score.most_common(k) if s > 0]

    # -------------------------------------------------------------- lexical
    def search(self, query: str, k: int = 8, tests: bool = False) -> list[Chunk]:
        s = self._bm25.scores(tokenize(query))
        ranked = [i for i, _ in sorted(s.items(), key=lambda x: -x[1]) if self.chunks[i].is_test == tests]
        if self.embed_model:
            ranked = self._fuse(query, ranked[:200])
        return [self.chunks[i] for i in ranked[:k]]

    def _fuse(self, query: str, cand: list[int]) -> list[int]:
        """Reciprocal-rank fusion of BM25 order with embedding similarity."""
        from sentence_transformers import SentenceTransformer       # optional dependency
        import numpy as np
        if self._emb is None:
            self._emb = SentenceTransformer(self.embed_model)
        vecs = self._emb.encode([self.chunks[i].text[:2000] for i in cand] + [query],
                                normalize_embeddings=True)
        sim = vecs[:-1] @ vecs[-1]
        emb_rank = {cand[j]: r for r, j in enumerate(np.argsort(-sim))}
        rrf = {i: 1 / (60 + r) + 1 / (60 + emb_rank[i]) for r, i in enumerate(cand)}
        return sorted(cand, key=lambda i: -rrf[i])


def render(chunks: list[Chunk], budget_chars: int = 24_000, per_chunk: int = 3_000) -> str:
    out, used, seen = [], 0, set()
    for c in chunks:
        key = (c.path, c.start)
        if key in seen:
            continue
        seen.add(key)
        body = c.text if len(c.text) <= per_chunk else c.text[:per_chunk] + "\n# ... (truncated)"
        block = f"### {c.header()}\n```python\n{body}\n```"
        if used + len(block) > budget_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)
