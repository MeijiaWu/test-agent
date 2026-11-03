"""Unified diffs for the generated test files.

SWT-bench takes predictions as a git patch (``model_patch``) and applies it
with ``git apply``; new files use the ``--- /dev/null`` form.
"""

from __future__ import annotations

import posixpath


def safe_relpath(path: str) -> str:
    """Normalise a model-proposed path and refuse anything outside the repo."""
    p = posixpath.normpath(path.strip().replace("\\", "/"))
    if p.startswith("../") or p == ".." or p.startswith("/") or not p.endswith(".py"):
        raise ValueError(f"invalid test file path: {path!r}")
    return p


def new_files_patch(files: dict[str, str]) -> str:
    out = []
    for path, content in files.items():
        path = safe_relpath(path)
        if not content.endswith("\n"):
            content += "\n"
        lines = content.splitlines(keepends=True)
        out.append(f"diff --git a/{path} b/{path}\n"
                   f"new file mode 100644\n"
                   f"--- /dev/null\n"
                   f"+++ b/{path}\n"
                   f"@@ -0,0 +1,{len(lines)} @@\n" + "".join("+" + ln for ln in lines))
    return "".join(out)
