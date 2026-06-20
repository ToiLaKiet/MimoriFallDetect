#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PY:-$HOME/.pyenv/versions/3.10.16/bin/python}" app.py "$@"
