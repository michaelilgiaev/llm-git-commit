"""checks.py - the die() helper and precondition checks for `repo`.

Mirrors qvm's checks.py: each check raises RepoError (caught in
command_line_interface.py, printed as 'repo: <msg>' and exit 1) instead of
calling exit directly, so the checks stay composable and testable.
"""

from __future__ import annotations

import os
import shutil
import subprocess


class RepoError(Exception):
    """A user-facing error. command_line_interface.py prints it and exits 1."""


def die(msg: str) -> "typing.NoReturn":  # noqa: F821
    raise RepoError(msg)


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def require_git() -> None:
    if not _have("git"):
        die("git missing -- sudo pacman -S git")


def require_inside_work_tree() -> None:
    """Fail cleanly when not inside a git repository (a drop-in for git commit must)."""
    require_git()
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, env=env, timeout=30,
        )
    except subprocess.TimeoutExpired:
        die("git rev-parse timed out")
    if res.returncode != 0 or res.stdout.strip() != "true":
        die("not inside a git repository.")
