"""Bounds that keep a commit from stalling on a huge diff.

These stay offline: no model, no network. They only check that the harness
returns a line for inputs that used to spend their time in SequenceMatcher.
"""

from __future__ import annotations

import time

import harness


def _udiff(removed: list[str], added: list[str], path: str = "app.py") -> str:
    rows = [
        f"diff --git a/{path} b/{path}",
        "index 1111111..2222222 100644",
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,{max(len(removed), 1)} +1,{max(len(added), 1)} @@",
    ]
    rows.extend("-" + line for line in removed)
    rows.extend("+" + line for line in added)
    return "\n".join(rows)


def test_long_code_lines_finish():
    # Distinct tokens, so difflib cannot skip the lines as junk. Each line is
    # longer than the ratio cap; the subject still has to come back quickly.
    def row(i: int, tail: str) -> str:
        parts = [f"tok{i}_{j}" for j in range(60)]
        return f"value_{i} = helper({', '.join(parts)}, {tail})"

    removed = [row(i, "plain") for i in range(36)]
    added = [row(i, "strict") for i in range(36)]
    started = time.monotonic()
    line = harness.subject("", _udiff(removed, added))
    assert time.monotonic() - started < 3
    assert isinstance(line, str) and line


def test_string_wall_does_not_scan_every_pair():
    """A code line after a wall of strings must not compare every string."""
    removed = [f'"alpha {i} ' + ("y" * 500) + '"' for i in range(80)]
    added = [f'"alpha {i} ' + ("z" * 500) + '"' for i in range(80)]
    added.append("counter += 1")
    started = time.monotonic()
    line = harness.subject("", _udiff(removed, added))
    assert time.monotonic() - started < 3
    assert isinstance(line, str) and line
