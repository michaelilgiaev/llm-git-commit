"""Tests for commit_command.run -- the full flow, OFFLINE.

llm.generate_subject (the only model call) is monkeypatched, so these exercise the
real checks -> git_io -> message -> git_io pipeline against a throwaway repo without
loading any model.
"""

from __future__ import annotations

import subprocess

import commit_command
import llm


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True)


def test_commits_with_generated_subject(staged_repo, monkeypatch):
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "Add greeting helper")
    assert commit_command.run() == 0
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Add greeting helper" in log


def test_nothing_staged_is_noop(git_repo, monkeypatch, capsys):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return "should not be used"

    monkeypatch.setattr(llm, "generate_subject", _boom)
    assert commit_command.run() == 0
    assert called["n"] == 0
    assert "Nothing staged" in capsys.readouterr().out


def test_retries_once_then_succeeds(staged_repo, monkeypatch):
    # First reply is pure chatter (cleans to ''), second is a real subject.
    replies = iter(["Here is the subject:", "Fix the thing"])
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: next(replies))
    assert commit_command.run() == 0
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Fix the thing" in log


def test_all_chatter_fails_without_committing(staged_repo, monkeypatch, capsys):
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "Done. Ready above.")
    assert commit_command.run() == 1
    assert "could not generate" in capsys.readouterr().err
    import git_io
    assert git_io.head_has_commits() is False
