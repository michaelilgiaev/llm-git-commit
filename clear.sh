#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage: clear.sh [--venv] [--output] [--cache] [--logs] [-h]

Clears the build tree. With NO flags it clears everything (the default): venv/,
output/, cache/, logs/, every __pycache__ in the project, and .pytest_cache/.

Pass flags to clear only PART of the tree (flags combine), each its own directory:
  --venv     clear only venv/    (the runtime/test venv -> forces a fresh venv)
  --output   clear only output/  (the delivered launchers)
  --cache    clear only cache/   (also sweeps __pycache__ and .pytest_cache/)
  --logs     clear only logs/
  -h, --help show this help and exit

Examples:
  clear.sh                  clear venv/, output/, cache/, logs/, __pycache__, .pytest_cache (all)
  clear.sh --output --logs  clear output/ and logs/ only
  clear.sh --venv           clear only venv/ (next compile.sh/tests.sh rebuilds it)
EOF
}

do_venv=0
do_output=0
do_cache=0
do_logs=0
any_flag=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)      do_venv=1;   any_flag=1 ;;
    --output)    do_output=1; any_flag=1 ;;
    --cache)     do_cache=1;  any_flag=1 ;;
    --logs)      do_logs=1;   any_flag=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)
      echo "clear.sh: unknown option '$1'" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done
if [ "$any_flag" -eq 0 ]; then       # no flags -> the everything-clear
  do_venv=1
  do_output=1
  do_cache=1
  do_logs=1
fi
# __pycache__ / .pytest_cache ride with cache (or the no-flag default that selects cache).
sweep_pyc="$do_cache"

targets=()
[ "$do_venv" -eq 1 ]   && targets+=(venv)
[ "$do_logs" -eq 1 ]   && targets+=(logs)
[ "$do_cache" -eq 1 ]  && targets+=(cache)
[ "$do_output" -eq 1 ] && targets+=(output)
deleted=0
for d in "${targets[@]}"; do
  if [ ! -e "$d" ]; then
    echo "  [ skip    ] $d/ -- not present, nothing to delete"
    continue
  fi
  size="$(du -sh "$d" 2>/dev/null | cut -f1)"
  if rm -rf "$d"; then
    echo "  [ deleted ] $d/ (was ${size:-?})"
    deleted=$((deleted + 1))
  else
    echo "  [ FAILED  ] $d/ still present -- could not remove (permissions?)"
  fi
done

pollution_deleted=0
if [ "$sweep_pyc" -eq 1 ]; then
  echo
  pyc_found=0
  while IFS= read -r -d '' pc; do
    pyc_found=$((pyc_found + 1))
    if rm -rf "$pc"; then
      pollution_deleted=$((pollution_deleted + 1))
    else
      echo "  [ FAILED  ] $pc -- could not remove (permissions?)"
    fi
  done < <(find . -type d -name __pycache__ -print0)

  if [ "$pyc_found" -eq 0 ]; then
    echo "  [ skip    ] __pycache__ -- none found in the tree"
  else
    echo "  [ deleted ] $pyc_found __pycache__ director$( [ "$pyc_found" -eq 1 ] && echo y || echo ies )"
  fi

  if [ -e ".pytest_cache" ]; then
    if rm -rf ".pytest_cache"; then
      echo "  [ deleted ] .pytest_cache/"
      pollution_deleted=$((pollution_deleted + 1))
    else
      echo "  [ FAILED  ] .pytest_cache/ -- could not remove (permissions?)"
    fi
  else
    echo "  [ skip    ] .pytest_cache/ -- not present"
  fi
fi

echo
total=$((deleted + pollution_deleted))
if [ "$total" -eq 0 ]; then
  echo "Nothing was deleted -- the tree was already clean."
elif [ "$sweep_pyc" -eq 1 ]; then
  echo "Removed $deleted build director$( [ "$deleted" -eq 1 ] && echo y || echo ies )" \
       "and $pollution_deleted Python artifact$( [ "$pollution_deleted" -eq 1 ] && echo '' || echo s )."
else
  echo "Removed $deleted build director$( [ "$deleted" -eq 1 ] && echo y || echo ies )."
fi
