"""Tests for command_line_interface.main -- dispatch and usage.

commit_command.run is stubbed so these only check argv routing and exit codes; no
model is loaded.
"""

from __future__ import annotations

import command_line_interface as cli


def test_help_prints_usage_and_exits_zero(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert "llm-git-commit" in out
    assert "commit" in out


def test_no_args_defaults_to_help(capsys):
    assert cli.main([]) == 0
    assert "USAGE" in capsys.readouterr().out


def test_unknown_subcommand_is_error(capsys):
    assert cli.main(["frobnicate"]) == 2
    assert "unknown subcommand" in capsys.readouterr().err


def test_commit_dispatches(monkeypatch):
    monkeypatch.setattr(cli.commit_command, "run", lambda: 0)
    assert cli.main(["commit"]) == 0


def test_version_subcommand_is_gone(capsys):
    # `version` was removed; it must be an unknown subcommand, not a crash.
    assert cli.main(["version"]) == 2
    assert "unknown subcommand" in capsys.readouterr().err


def test_repoerror_becomes_exit_1(monkeypatch, capsys):
    from checks import RepoError

    def _raise():
        raise RepoError("not inside a git repository.")

    monkeypatch.setattr(cli.commit_command, "run", _raise)
    assert cli.main(["commit"]) == 1
    assert "not inside a git repository" in capsys.readouterr().err
