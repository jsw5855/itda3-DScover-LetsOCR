#!/usr/bin/env bash
set -euo pipefail
SCRIPT="${BASH_SOURCE[0]}"
case "$SCRIPT" in */*) ROOT="${SCRIPT%/*}" ;; *) ROOT="." ;; esac
ROOT="$(cd -- "$ROOT" && pwd)"
exec "${PYTHON:-python}" "$ROOT/scripts/download_weights.py" "$@"
