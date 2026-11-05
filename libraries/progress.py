"""progress.py - a one-line "current/total" progress bar on stderr.

Rendered on stderr so it never pollutes stdout. Carriage-return in place, then
one newline when done, so a later step cannot restart the same line.
"""

from __future__ import annotations

import sys

_WIDTH = 24


def bar(current: int, total: int, *, prefix: str = "") -> None:
    """Draw/redraw the bar in place: `prefix [####----] cur/total`. Clamped so a
    model that overshoots max_tokens still shows a full, in-range bar."""
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = (current * _WIDTH) // total
    line = f"\r{prefix}[{'#' * filled}{'-' * (_WIDTH - filled)}] {current}/{total}"
    sys.stderr.write(line)
    sys.stderr.flush()


def done() -> None:
    """Finish the line. One bar, then a newline, so the next step cannot redraw it."""
    sys.stderr.write("\n")
    sys.stderr.flush()
