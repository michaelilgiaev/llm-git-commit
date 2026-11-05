"""commit_command.py - the AI-written commit: a drop-in for `git commit` (no -m).

Asks the LOCAL model (llm.py) for ONE commit subject line for the ALREADY-STAGED
diff and commits with it. It does NOT stage anything -- `git add` yourself first,
exactly like a plain `git commit -m`. With nothing staged it is a clean no-op.

Orchestration only: checks -> git_io (read diff) -> llm (local model) -> message
(clean) -> git_io (commit). One pass. A rejected subject is an error, not an editor.
"""

from __future__ import annotations

import sys

import checks
import git_io
import llm
import message

def run() -> int:
    """Commit staged changes with a model-written subject. Returns an exit code."""
    checks.require_inside_work_tree()

    # Do NOT stage. Commit only what is already staged; a clean no-op otherwise,
    # exactly like a bare `git commit`.
    if not git_io.has_staged_changes():
        print("Nothing staged to commit.")
        return 0

    subject = message.extract_subject(llm.generate_subject(git_io.staged_diff()))
    if not subject:
        print("llm-git-commit: could not write a commit subject.", file=sys.stderr)
        return 1

    git_io.commit(subject)
    print(f"llm-git-commit: {subject}")
    return 0
