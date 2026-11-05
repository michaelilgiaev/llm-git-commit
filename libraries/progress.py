"""progress.py - a one-line "current/total" progress bar on stderr.

Rendered on stderr so it never pollutes stdout (the commit subject path) and never
lands in a pipe. Carriage-return in place, cleared when done. Zero dependencies.
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
    """Clear the bar line so it leaves no trace once generation finishes."""
    sys.stderr.write("\r\033[K")
    sys.stderr.flush()
