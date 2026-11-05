"""Tests for git_io.py -- the git plumbing wrappers, against throwaway repos.

These run REAL git, but only inside the `git_repo`/`staged_repo` fixtures (fresh
repos under tmp_path). Nothing in any real repository is read or modified.
"""

from __future__ import annotations

import subprocess

import git_io


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True)


def test_no_staged_changes_on_fresh_repo(git_repo):
    assert git_io.has_staged_changes() is False


def test_staged_changes_detected(staged_repo):
    assert git_io.has_staged_changes() is True


def test_staged_diff_contains_stat_and_body(staged_repo):
    diff = git_io.staged_diff()
    assert "app.py" in diff          # the --stat names the file
    assert "---" in diff             # the separator between stat and body
    assert "def hi" in diff          # a short patch carries the actual change


def test_long_patch_is_dropped():
    # The failure mode: a big patch (this repo's README) makes the model copy
    # file text and the cleaner then rejects it.
    text = git_io._shape("README.md | 64 +-\n", "Write " + ("x" * 2000))
    assert "README.md" in text
    assert "xxxx" not in text
    assert "---" not in text


def test_readme_prose_is_not_fed_to_the_model(git_repo):
    prose = "Write git commit subjects with a small language model.\n"
    (git_repo / "README.md").write_text(prose)
    _git(git_repo, "add", "README.md")
    diff = git_io.staged_diff()
    assert "README.md" in diff
    assert "small language model" not in diff


def test_commit_preview_cached_only_ignores_unstaged(staged_repo):
    # A tracked file changed but NOT staged: without include_tracked it must not
    # appear; with it (the `-a` case) it must.
    (staged_repo / "app.py").write_text("def hi():\n    return 'HELLO'\n")
    _git(staged_repo, "add", "app.py")
    _git(staged_repo, "commit", "-q", "-m", "base")
    (staged_repo / "app.py").write_text("def hi():\n    return 'CHANGED'\n")  # unstaged

    assert "CHANGED" not in git_io.commit_preview_diff(include_tracked=False)
    assert "CHANGED" in git_io.commit_preview_diff(include_tracked=True)


def test_anything_to_commit(staged_repo):
    assert git_io.anything_to_commit(include_tracked=False) is True   # file is staged
    git_io.commit("base")
    assert git_io.anything_to_commit(include_tracked=False) is False  # clean tree now
    (staged_repo / "app.py").write_text("def hi():\n    return 'X'\n")  # unstaged edit
    assert git_io.anything_to_commit(include_tracked=False) is False  # not staged
    assert git_io.anything_to_commit(include_tracked=True) is True    # -a would catch it


def test_commit_records_message(staged_repo):
    git_io.commit("Add greeting helper")
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Add greeting helper" in log
    assert git_io.has_staged_changes() is False


def test_head_has_commits_tracks_state(staged_repo):
    assert git_io.head_has_commits() is False
    git_io.commit("first")
    assert git_io.head_has_commits() is True
