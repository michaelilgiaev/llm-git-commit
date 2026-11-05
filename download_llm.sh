#!/usr/bin/env bash
# Download the GGUF once, before compile.sh. System python3 is enough: the venv
# does not exist yet, and this does not import llama. A file already on disk is
# left where it is. The byte bar is current/total in megabytes.
set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "download_llm.sh: no python3 -- pacman -S python" >&2; exit 1; }

exec "$PY" -P -c 'import sys; sys.path.insert(0, "libraries"); import llm; print(llm.ensure_model())'
