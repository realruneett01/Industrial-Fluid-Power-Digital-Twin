#!/usr/bin/env sh
# Plug-and-play launcher (Linux, macOS, Raspberry Pi / ARM). Run from the drive:
#   ./start-hydro.sh                 preflight + calibrate + self-check, then start the service
#   ./start-hydro.sh --check-only    preflight and gate only, start nothing
#   ./start-hydro.sh --repair        rebuild the Python environment first
# Extra arguments go to `python -m hydro deploy`.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
export HYDRO_ROOT="$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1   # keep the drive free of .pyc files (they would break the manifest)

CHECK_ONLY=0
REPAIR=0
ARGS=""
for a in "$@"; do
  case "$a" in
    --check-only) CHECK_ONLY=1 ;;
    --repair) REPAIR=1 ;;
    *) ARGS="$ARGS $a" ;;
  esac
done

find_python() {
  for c in python3.13 python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PY=$(find_python) || {
  echo "ERROR: Python 3.12 or newer was not found." >&2
  echo "Install it (https://www.python.org/downloads/) or use the Docker image instead." >&2
  exit 3
}

# The environment lives in the user's cache, not on the drive: USB sticks are often FAT/exFAT,
# which cannot hold virtualenv symlinks and are slow. One environment per drive path.
KEY=$(printf '%s' "$HERE" | cksum | cut -d' ' -f1)
VENV="${HYDRO_VENV:-${XDG_CACHE_HOME:-$HOME/.cache}/hydro/venv-$KEY}"
[ "$REPAIR" = 1 ] && rm -rf "$VENV"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating Python environment in $VENV ..."
  "$PY" -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c 'import numpy, scipy, pandas, yaml, onnxruntime, asyncua' >/dev/null 2>&1; then
  if [ -d "$HERE/wheelhouse" ]; then
    echo "Installing from the offline wheelhouse ..."
    "$VENV/bin/python" -m pip install --quiet --no-index --find-links "$HERE/wheelhouse" \
      -r "$HERE/wheelhouse/requirements.txt"
  else
    echo "No wheelhouse on the drive; installing from the package index (needs internet) ..."
    "$VENV/bin/python" -m pip install --quiet numpy scipy pandas pyyaml onnxruntime asyncua
  fi
fi

if [ "$CHECK_ONLY" = 1 ]; then
  # shellcheck disable=SC2086
  exec "$VENV/bin/python" -m hydro deploy $ARGS
fi
# shellcheck disable=SC2086
exec "$VENV/bin/python" -m hydro deploy --start $ARGS
