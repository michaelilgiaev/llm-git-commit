"""git_io.py - thin, testable wrappers over the git plumbing we need.

Every git shell-out lives here so the command/shim modules stay pure logic and the
tests can drive git through ONE surface. Each wrapper returns data (or raises
RepoError via die); none print.
"""

from __future__ import annotations

import subprocess

from checks import die

# How much of the diff we feed the model. A small model needs no more than this to
# write one subject line, and a huge body only invites a rambling preamble.
_DIFF_BYTES = 6000


def _git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run `git <args>` capturing text output. With check=True a non-zero exit dies."""
    res = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        die(f"git {' '.join(args)} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res


def has_staged_changes() -> bool:
    """True when something is staged. Mirrors `git diff --cached --quiet`: exit 1
    means there IS a staged change."""
    return _git("diff", "--cached", "--quiet").returncode != 0


def anything_to_commit(*, include_tracked: bool) -> bool:
    """True when a `git commit` would actually commit something: staged changes, or
    (with include_tracked, the `-a` case) any tracked-file working-tree change. The
    git-native counterpart to commit_preview_diff's emptiness."""
    if has_staged_changes():
        return True
    return include_tracked and _git("diff", "--quiet").returncode != 0


def staged_diff() -> str:
    """The staged diff: a `--stat` summary, a separator, then the first _DIFF_BYTES
    of the full cached diff."""
    return _shape(_git("--no-pager", "diff", "--cached", "--stat").stdout,
                  _git("--no-pager", "diff", "--cached").stdout)


def commit_preview_diff(*, include_tracked: bool) -> str:
    """The diff the model should summarise for a `git commit`. Always the staged
    (--cached) changes; with include_tracked=True (the `-a`/`--all` case, where git
    also commits tracked-but-unstaged edits) the working-tree diff of tracked files
    is appended so the subject reflects everything that will land."""
    stat = _git("--no-pager", "diff", "--cached", "--stat").stdout
    body = _git("--no-pager", "diff", "--cached").stdout
    if include_tracked:
        stat += _git("--no-pager", "diff", "--stat").stdout
        body += _git("--no-pager", "diff").stdout
    return _shape(stat, body)


def _shape(stat: str, body: str) -> str:
    return f"{stat}\n---\n{body[:_DIFF_BYTES]}"


def commit(message: str) -> None:
    """Commit the ALREADY-staged changes with `message`. Dies on a git failure."""
    _git("commit", "-m", message, check=True)


def head_has_commits() -> bool:
    """True once the repo has at least one commit (a fresh repo has no HEAD)."""
    return _git("rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
