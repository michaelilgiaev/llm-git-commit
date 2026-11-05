"""Tests for git_shim -- the `git` interception logic.

The classifier (_is_plain_commit) is the heart: it decides whether a `git ...`
invocation gets hijacked or passed straight through. We also cover real-git
resolution (never resolving back to the shim itself) and the full hijack flow with
the model and the delegated git call both faked -- no model, no network.
"""

from __future__ import annotations

import os

import pytest

import git_shim


@pytest.mark.parametrize("args", [
    ["commit"],
    ["commit", "-a"],
    ["commit", "--all"],
    ["commit", "-s"],
    ["commit", "-v"],
    ["commit", "-a", "-s"],
    ["commit", "--", "file.py"],
    ["commit", "-a", "--", "file.py"],
])
def test_plain_commit_is_hijacked(args):
    assert git_shim._is_plain_commit(args) is True


@pytest.mark.parametrize("args", [
    [],                                   # bare `git`
    ["status"],
    ["push"],
    ["commit", "-m", "msg"],
    ["commit", "--message", "msg"],
    ["commit", "--message=msg"],
    ["commit", "-m", "msg", "-a"],
    ["commit", "-am", "msg"],             # BUNDLED -a + -m: message IS given
    ["commit", "-sm", "msg"],             # bundled -s + -m
    ["commit", "-avm", "msg"],            # bundled -a + -v + -m
    ["commit", "-F", "file"],
    ["commit", "--file=file"],
    ["commit", "-C", "HEAD"],
    ["commit", "--amend"],
    ["commit", "--amend", "--no-edit"],
    ["commit", "--fixup", "HEAD"],
    ["commit", "--fixup=HEAD"],
    ["commit", "--squash", "HEAD"],
    ["log", "commit"],                    # `commit` only matters as the subcommand
])
def test_other_invocations_pass_through(args):
    assert git_shim._is_plain_commit(args) is False


@pytest.mark.parametrize("args,expected", [
    (["-a"], True),
    (["--all"], True),
    (["-as"], True),                      # bundled -a + -s
    (["-sa"], True),
    (["-s"], False),
    (["-v"], False),
    (["--", "-a"], False),                # after --, `-a` is a pathspec
])
def test_stages_all_detection(args, expected):
    assert git_shim._stages_all(args) is expected


def test_message_flag_after_ddash_is_a_pathspec_not_a_flag():
    # After `--`, `-m` is a pathspec named "-m", NOT the message flag -> still hijack.
    assert git_shim._is_plain_commit(["commit", "--", "-m"]) is True


def test_real_git_uses_shim_env_when_argv0_is_dash_c(monkeypatch, tmp_path):
    # Installed launcher is `python -c`, so argv[0] is "-c". The directory to skip
    # comes from LLM_GIT_COMMIT_SHIM, which compile.sh writes into the stub.
    shim_dir = tmp_path / "shim"
    real_dir = tmp_path / "real"
    shim_dir.mkdir()
    real_dir.mkdir()
    (shim_dir / "git").write_text("#!/bin/sh\n")
    (real_dir / "git").write_text("#!/bin/sh\n")
    for p in (shim_dir / "git", real_dir / "git"):
        p.chmod(0o755)

    monkeypatch.setattr(git_shim.sys, "argv", ["-c", "commit"])
    monkeypatch.setenv("LLM_GIT_COMMIT_SHIM", str(shim_dir / "git"))
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")
    assert git_shim._real_git() == str(real_dir / "git")


def test_exec_git_hides_shim_dir_from_the_child(monkeypatch, tmp_path):
    shim_dir = tmp_path / "shim"
    real_dir = tmp_path / "real"
    shim_dir.mkdir()
    real_dir.mkdir()
    seen = tmp_path / "path.txt"
    (shim_dir / "git").write_text("#!/bin/sh\n")
    (shim_dir / "git").chmod(0o755)
    (real_dir / "git").write_text(f"#!/bin/sh\nprintf '%s' \"$PATH\" > '{seen}'\n")
    (real_dir / "git").chmod(0o755)

    monkeypatch.setenv("LLM_GIT_COMMIT_SHIM", str(shim_dir / "git"))
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")
    assert git_shim._exec_git(str(real_dir / "git"), ["status"]) == 0
    assert str(shim_dir) not in seen.read_text().split(os.pathsep)
    # The parent PATH is unchanged -- a leaked mutation would hide git from later tests.
    assert str(shim_dir) in os.environ["PATH"].split(os.pathsep)


def test_real_git_excludes_the_shim_dir(monkeypatch, tmp_path):
    # A fake `git` placed in a dir that is on PATH AND is the shim's own dir must be
    # skipped, so the shim never resolves to itself.
    shim_dir = tmp_path / "shim"
    real_dir = tmp_path / "real"
    shim_dir.mkdir()
    real_dir.mkdir()
    (shim_dir / "git").write_text("#!/bin/sh\n")
    (real_dir / "git").write_text("#!/bin/sh\n")
    for p in (shim_dir / "git", real_dir / "git"):
        p.chmod(0o755)

    monkeypatch.setattr(git_shim.sys, "argv", [str(shim_dir / "git"), "commit"])
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")
    assert git_shim._real_git() == str(real_dir / "git")


def test_hijack_generates_subject_and_delegates_to_real_git(monkeypatch):
    calls = []
    monkeypatch.setattr(git_shim, "_real_git", lambda: "/usr/bin/git")
    monkeypatch.setattr(git_shim, "_exec_git",
                        lambda git, args: calls.append((git, args)) or 0)

    # Stub the deferred imports the hijack path pulls in.
    import checks
    import git_io
    import llm
    monkeypatch.setattr(checks, "require_inside_work_tree", lambda: None)
    monkeypatch.setattr(git_io, "anything_to_commit", lambda *, include_tracked: True)
    monkeypatch.setattr(git_io, "commit_preview_diff", lambda *, include_tracked: "a diff")
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "Add feature")

    assert git_shim.main(["commit", "-s"]) == 0
    assert calls == [("/usr/bin/git", ["commit", "-m", "Add feature", "-s"])]


def test_hijack_does_not_open_the_editor_when_model_gives_nothing(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(git_shim, "_real_git", lambda: "/usr/bin/git")
    monkeypatch.setattr(git_shim, "_exec_git",
                        lambda git, args: calls.append(args) or 0)
    import checks
    import git_io
    import llm
    monkeypatch.setattr(checks, "require_inside_work_tree", lambda: None)
    monkeypatch.setattr(git_io, "anything_to_commit", lambda *, include_tracked: True)
    monkeypatch.setattr(git_io, "commit_preview_diff", lambda *, include_tracked: "a diff")
    monkeypatch.setattr(llm, "generate_subject", lambda *a, **k: "Done.")  # cleans to ''

    assert git_shim.main(["commit"]) == 1
    assert calls == []                  # no git commit, so no editor
    assert "could not write a commit subject" in capsys.readouterr().err


def test_hijack_noop_when_nothing_to_commit(monkeypatch):
    calls = []
    monkeypatch.setattr(git_shim, "_real_git", lambda: "/usr/bin/git")
    monkeypatch.setattr(git_shim, "_exec_git",
                        lambda git, args: calls.append(args) or 0)
    import checks
    import git_io
    monkeypatch.setattr(checks, "require_inside_work_tree", lambda: None)
    monkeypatch.setattr(git_io, "anything_to_commit", lambda *, include_tracked: False)

    assert git_shim.main(["commit"]) == 0
    assert calls == [["commit"]]        # nothing to commit -> let real git say so


def test_passthrough_execs_real_git(monkeypatch):
    calls = []
    monkeypatch.setattr(git_shim, "_real_git", lambda: "/usr/bin/git")
    monkeypatch.setattr(git_shim, "_exec_git",
                        lambda git, args: calls.append((git, args)) or 7)
    assert git_shim.main(["status", "-s"]) == 7
    assert calls == [("/usr/bin/git", ["status", "-s"])]


def test_no_real_git_is_clean_error(monkeypatch, capsys):
    monkeypatch.setattr(git_shim, "_real_git", lambda: None)
    assert git_shim.main(["status"]) == 127
    assert "no real `git`" in capsys.readouterr().err
