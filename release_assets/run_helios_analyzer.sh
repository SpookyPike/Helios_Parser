#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_INSTALL=0
NO_LAUNCH=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --no-launch) NO_LAUNCH=1 ;;
    *) ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python 3.10+ was not found in PATH." >&2
  exit 1
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi

VENV_PYTHON="$ROOT/.venv/bin/python"

if [ "$SKIP_INSTALL" -eq 0 ]; then
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -e ".[desktop]"
fi

if [ "$NO_LAUNCH" -eq 1 ]; then
  echo "Environment ready at $VENV_PYTHON"
  exit 0
fi

"$VENV_PYTHON" -m helios_app
