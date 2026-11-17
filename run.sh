#!/usr/bin/env bash
# Run ACR from its private .venv. Install first with install.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -x .venv/bin/python ]; then
  echo "ACR is not installed yet."
  echo "Double-click Install-ACR, or run: bash install.sh"
  exit 1
fi

# v0.11: launch through acr_launch.py, which checks the
# single-instance lock after importing QtCore only -- a second
# launch exits in about a second instead of paying for the full
# import first. The launcher execs acr_app.py in this process.
exec .venv/bin/python acr_launch.py "$@"
