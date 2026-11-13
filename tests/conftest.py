"""Shared pytest fixtures + import-path setup for the llm-git-commit test suite.

`bash tests.sh` already puts libraries/ on PYTHONPATH, and pyproject.toml's pythonpath
does the same for a bare `pytest` run. This conftest belt-and-suspenders it so the flat
modules resolve no matter how the tests are launched, and provides a `git_repo` fixture:
a throwaway git repository under tmp_path with a deterministic identity, so tests that
exercise git_io/commit never touch a real repo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_LIB = str(REPO / "libraries")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )


@pytest.fixture()
def git_repo(tmp_path, monkeypatch):
    """An initialised, empty git repo with a pinned user identity. cd'd into, so the
    CWD-relative git calls in git_io act on it. Returns its Path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture()
def staged_repo(git_repo):
    """A git_repo with one file staged, ready for a commit. Returns its Path."""
    (git_repo / "app.py").write_text("def hi():\n    return 'hi'\n")
    _git(git_repo, "add", "app.py")
    return git_repo
