"""git_io.py - thin, testable wrappers over the git plumbing we need.

Every git shell-out lives here so the command/shim modules stay pure logic and the
tests can drive git through ONE surface. Each wrapper returns data (or raises
RepoError via die); none print.
"""

from __future__ import annotations

import os
import subprocess

import harness
from checks import die

# A diff that never returns (a smudge filter, a pager, a credential prompt)
# used to freeze `git commit`. Reads fail instead. `git commit` itself waits,
# so a hook the user configured can still finish.
_GIT_TIMEOUT_S = 120

# Docs and images stay out of the patch even when it is short. Their prose is
# exactly what the model copies back as the subject. The stat line still names them.
_DOC_EXCLUDE = (
    ":(exclude)*.md",
    ":(exclude)*.png",
    ":(exclude)*.jpg",
    ":(exclude)*.jpeg",
    ":(exclude)*.gif",
    ":(exclude)*.webp",
    ":(exclude)*.svg",
    ":(exclude)*.ico",
)


def _git(*args: str, check: bool = False, timeout: float | None = _GIT_TIMEOUT_S,
        ) -> subprocess.CompletedProcess:
    """Run `git <args>` capturing text output. With check=True a non-zero exit dies.

    `timeout` bounds a read. None waits, which is what `commit` uses so a
    pre-commit hook is not killed.
    """
    env = os.environ.copy()
    # Do not sit on a username prompt. Pass-through git (push, fetch) does not
    # come through here, so those commands can still ask.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        res = subprocess.run(
            ["git", *args], capture_output=True, text=True, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        die(f"git {' '.join(args)} timed out after {timeout}s")
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


def _patch(*, cached: bool, tracked: bool) -> str:
    """Code patch only. Markdown and images are excluded (see _DOC_EXCLUDE)."""
    spec = [".", *_DOC_EXCLUDE]
    chunks = []
    if cached:
        chunks.append(_git("--no-pager", "diff", "--cached", "--", *spec).stdout)
    if tracked:
        chunks.append(_git("--no-pager", "diff", "--", *spec).stdout)
    return "".join(chunks)


def staged_diff() -> str:
    """What the model reads for a staged commit: the stat, plus a short code patch."""
    return _shape(_git("--no-pager", "diff", "--cached", "--stat").stdout,
                  _patch(cached=True, tracked=False))


def commit_preview_diff(*, include_tracked: bool) -> str:
    """What the model reads for `git commit`. Staged changes always; with
    include_tracked (the `-a` case) the unstaged edits of tracked files too."""
    stat = _git("--no-pager", "diff", "--cached", "--stat").stdout
    if include_tracked:
        stat += _git("--no-pager", "diff", "--stat").stdout
    return _shape(stat, _patch(cached=True, tracked=include_tracked))


def _shape(stat: str, body: str) -> str:
    """Fact card for Qwen. The draft line inside it is the commit fallback.

    Raw unified diffs are not included. The 1.5B copies those back as the subject.
    """
    return harness.brief(stat.strip(), body.strip())


def commit(message: str) -> None:
    """Commit the ALREADY-staged changes with `message`. Dies on a git failure."""
    _git("commit", "-m", message, check=True, timeout=None)


def head_has_commits() -> bool:
    """True once the repo has at least one commit (a fresh repo has no HEAD)."""
    return _git("rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
