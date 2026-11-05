#!/usr/bin/env bash
#
# llm-git-commit -- test entry point. `bash tests.sh` is the ONE command.
#
# Self-bootstrapping: creates venv/ if missing, installs tests/requirements.txt into it,
# runs pytest. Nothing global is touched; venv/ is gitignored at the project root (the same
# venv compile.sh and clear.sh use). pip is skipped when requirements.txt is unchanged.
#
# The tests are PURE and OFFLINE. They never load a GGUF and never reach the network (the
# local model is faked). Git IS exercised, but only against throwaway repos created under
# pytest's tmp_path -- nothing in your real repos is ever touched or committed.
#
# Any arguments are passed straight through to pytest, e.g.:
#   bash tests.sh -k shim           run only tests matching "shim"
#   bash tests.sh -q                quiet
#   bash tests.sh tests/test_message.py
#
set -o errexit
set -o nounset
set -o pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPODIR"

VENV="$REPODIR/venv"
PY="$VENV/bin/python"
REQ="$REPODIR/tests/requirements.txt"
STAMP="$VENV/.requirements.installed"

if [ ! -x "$PY" ]; then
    echo "[tests] creating venv at $VENV"
    python3 -m venv "$VENV"
fi

REQ_HASH=""
[ -f "$REQ" ] && REQ_HASH="$(sha256sum "$REQ" | cut -d' ' -f1)"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$REQ_HASH" ]; then
    echo "[tests] installing requirements"
    "$PY" -m pip install --quiet --upgrade pip
    [ -f "$REQ" ] && "$PY" -m pip install --quiet -r "$REQ"
    echo "$REQ_HASH" > "$STAMP"
fi

# The tests import the flat modules in libraries/ directly; expose that import root.
export PYTHONPATH="$REPODIR/libraries${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

echo "[tests] running pytest"
exec "$PY" -m pytest "$@"
