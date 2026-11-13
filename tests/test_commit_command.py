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


def test_one_generation_then_commit(staged_repo, monkeypatch):
    calls = {"n": 0}

    def _once(*a, **k):
        calls["n"] += 1
        return "Fix the thing"

    monkeypatch.setattr(llm, "generate_subject", _once)
    assert commit_command.run() == 0
    assert calls["n"] == 1
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Fix the thing" in log


def test_chatter_commits_the_harness_draft(staged_repo, monkeypatch, capsys):
    # The model answered with an acknowledgement. The staged file still commits
    # with the line the harness already wrote.
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "Done. Ready above.")
    assert commit_command.run() == 0
    captured = capsys.readouterr()
    assert "could not write a commit subject" not in captured.err
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Add hi in app.py" in log


def test_imperative_ignore_subject_is_committed(staged_repo, monkeypatch):
    subject = "Ignore FORGOTTEN_NODE for the active sender"
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: subject)
    assert commit_command.run() == 0
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert subject in log


def test_model_crash_commits_the_harness_draft(staged_repo, monkeypatch):
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "")
    assert commit_command.run() == 0
    log = _git(staged_repo, "log", "--oneline", "-1").stdout
    assert "Add hi in app.py" in log
