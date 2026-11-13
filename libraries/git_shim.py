"""git_shim.py - the `git` shim: bare `git commit` -> local-model subject.

Installed as an executable named `git` EARLIER on PATH than the real git. Every
invocation is inspected:

  * `git commit` with NO message source (no -m/-F/-c/-C/--amend/-t/--squash/--fixup)
    -> we generate the subject for the staged diff with the local model and hand it
    to the real git as `-m <subject>`, keeping the user's other commit flags
    (-a, -s, -v, pathspecs, ...). Real git still does the staging and the commit.
  * ANYTHING else (git commit -m, git push, git log, ...) -> exec the real git
    verbatim. The shim is transparent; when in doubt it delegates.

Finding the real git: we drop the shim's OWN directory from PATH and resolve `git`
again, so the shim never calls itself. If the model cannot write a subject we
stop and say so. We do not open the editor.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Long flags on `git commit` that SUPPLY or REUSE a message, or otherwise mean "do
# not let us write the subject". Presence of any one -> pass straight through.
_MESSAGE_LONG = {
    "--message", "--file", "--reedit-message", "--reuse-message", "--template",
    "--amend",          # reuses/edits the existing message
    "--squash", "--fixup",
    "--no-edit",
}
_MESSAGE_LONG_EQ = ("--message=", "--file=", "--reuse-message=", "--reedit-message=",
                    "--template=", "--squash=", "--fixup=")

# Short message-carrying flags. These TAKE an argument, so in a bundled cluster
# (e.g. `-am "msg"`) the letter can sit anywhere and swallows the rest as its value
# -- so ANY of these letters appearing in a `-xyz` cluster means a message is given.
_MESSAGE_SHORT = set("mFcCt")


def _shim_dir() -> str | None:
    """Directory of this shim, so we never resolve `git` back to ourselves.

    The installed launcher is a shell stub that runs `python -c`, and that makes
    sys.argv[0] literally `-c` -- useless as a path. compile.sh exports
    LLM_GIT_COMMIT_SHIM with the stub's own path. A direct `python git_shim.py`
    still has a real argv[0], which the unit tests use.
    """
    marked = os.environ.get("LLM_GIT_COMMIT_SHIM", "").strip()
    if marked:
        return os.path.dirname(os.path.abspath(marked))
    argv0 = sys.argv[0]
    if argv0 and argv0 != "-c" and os.path.exists(argv0):
        return os.path.dirname(os.path.abspath(argv0))
    return None


def _path_without_shim() -> str:
    """PATH with the shim's directory removed. Empty pieces are dropped too."""
    self_dir = _shim_dir()
    return os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and (not self_dir or os.path.abspath(p) != self_dir)
    )


def _real_git() -> str | None:
    """The real git binary on PATH, never this shim. None if there is no other git."""
    return shutil.which("git", path=_path_without_shim())


def _carries_message(tok: str) -> bool:
    """True when a single `git commit` option token supplies/reuses a message. Covers
    long (`--message`, `--amend`), `=`-form (`--message=x`), and BUNDLED short forms:
    `-am`/`-sm`/`-avm` all embed `-m`, so any message letter in a `-xyz` cluster
    counts (the short flag takes an argument and eats the rest of the cluster)."""
    if tok in _MESSAGE_LONG or tok.startswith(_MESSAGE_LONG_EQ):
        return True
    if tok.startswith("-") and not tok.startswith("--"):      # a short-flag cluster
        return bool(set(tok[1:]) & _MESSAGE_SHORT)
    return False


def _is_plain_commit(args: list[str]) -> bool:
    """True when `args` (everything after `git`) is a `commit` with no message source
    -- the only shape we hijack. A `--` ends option scanning (pathspecs follow)."""
    # shortcut: `commit` must be the FIRST token, so global flags before it
    # (`git -C dir commit`, `git -c k=v commit`) fall through to plain git rather
    # than the model. Safe (no corruption), just a missed hijack. Upgrade to a
    # git-global-flag parser only if that invocation form turns out to be common.
    if not args or args[0] != "commit":
        return False
    for tok in args[1:]:
        if tok == "--":
            break
        if _carries_message(tok):
            return False
    return True


def _exec_git(git: str, args: list[str]) -> int:
    """Run the real git. The child PATH has the shim directory removed so a hook
    that calls `git` cannot re-enter this process."""
    env = os.environ.copy()
    env["PATH"] = _path_without_shim()
    return subprocess.run([git, *args], env=env).returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    git = _real_git()
    if git is None:
        print("llm-git-commit: no real `git` found on PATH.", file=sys.stderr)
        return 127

    if not _is_plain_commit(args):
        return _exec_git(git, args)          # transparent pass-through

    # Hijack: write the subject with the local model, then let real git commit it
    # with the user's other flags (-a, -s, pathspecs, ...). git does the staging.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import checks  # noqa: E402
    import git_io  # noqa: E402
    import llm  # noqa: E402
    import message  # noqa: E402

    # git_io and checks shell out to `git` by name. Drop the shim directory for
    # that window only, then put PATH back so this process does not leak the change
    # into the caller (tests, or a later command in the same interpreter).
    saved_path = os.environ.get("PATH", "")
    os.environ["PATH"] = _path_without_shim()
    try:
        checks.require_inside_work_tree()

        staged = args[1:]                        # the commit flags, minus "commit"
        stages_tracked = _stages_all(staged)
        if not git_io.anything_to_commit(include_tracked=stages_tracked):
            return _exec_git(git, args)          # nothing to do -> git's own no-op message

        diff = git_io.commit_preview_diff(include_tracked=stages_tracked)
        subject = message.extract_subject(llm.generate_subject(diff))
        if not subject:
            print("llm-git-commit: could not write a commit subject.", file=sys.stderr)
            return 1

        return _exec_git(git, ["commit", "-m", subject, *staged])
    except checks.RepoError as exc:
        print(f"llm-git-commit: {exc}", file=sys.stderr)
        return 1
    finally:
        os.environ["PATH"] = saved_path


def _stages_all(args: list[str]) -> bool:
    """True when the commit flags include `-a`/`--all` (so git will also commit
    tracked-but-unstaged edits). Scans only up to `--` (pathspecs follow) and sees
    `-a` inside bundled short clusters (`-as`, `-sa`) too."""
    for tok in args:
        if tok == "--":
            break
        if tok == "--all":
            return True
        if tok.startswith("-") and not tok.startswith("--") and "a" in tok[1:]:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
