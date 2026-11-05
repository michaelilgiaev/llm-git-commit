"""command_line_interface.py - argument parsing and dispatch.

llm-git-commit does ONE thing: write a commit subject for the staged diff with a
small LOCAL model and commit. The primary way users reach it is the `git` shim
(git_shim.py) -- a bare `git commit` with no message is routed here. This CLI is the
direct entry (`llm-git-commit commit`) that the shim and the tests call.
"""

from __future__ import annotations

import os
import sys

# Flat sibling imports: the modules live directly in libraries/ (no package), so the
# bare imports resolve against this dir once it is on sys.path. The caller's CWD is
# preserved so git acts on the repository the user is standing in.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commit_command  # noqa: E402  (after the sys.path bootstrap above)
import configuration  # noqa: E402  (for the default-model name in usage)
import uninstall  # noqa: E402
from checks import RepoError  # noqa: E402

PROG = "llm-git-commit"


def usage() -> str:
    return f"""\
{PROG} - write a git commit subject with a small LOCAL model (no API, runs on CPU).

Installed as a `git` shim, a bare `git commit` (no -m/-F/-c/--amend/...) generates
the subject for the ALREADY-STAGED diff and commits it. `git commit -m "..."` and
every other git command pass straight through to real git, untouched.

USAGE:
  {PROG} commit      Commit the ALREADY-STAGED diff with a model-written subject.
                     Stages nothing -- `git add` first, like `git commit -m`. A clean
                     no-op with nothing staged. (This is what the `git` shim calls.)
  {PROG} --uninstall Remove the git shim, this command, the completion, the
                     downloaded model, and this checkout's venv. The source
                     checkout stays. A real git is never removed.
  {PROG} help        This text.

MODEL (local GGUF, fetched once by download_llm.sh, then fully offline):
  default {configuration.DEFAULT_REPO}
          {configuration.DEFAULT_FILE}

ENV OVERRIDES:
  LLM_GIT_COMMIT_REPO   Hugging Face repo id of the GGUF to use.
  LLM_GIT_COMMIT_FILE   GGUF filename within that repo.

EXAMPLE:
  git add -p
  git commit          # the local model writes the subject, then commits"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"

    try:
        if cmd == "commit":
            return commit_command.run()
        if cmd in ("uninstall", "--uninstall"):
            return uninstall.run()
        if cmd in ("help", "-h", "--help"):
            print(usage())
            return 0
        print(f"{PROG}: unknown subcommand: {cmd}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    except RepoError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
