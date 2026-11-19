#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
"$PY" -m pytest -q
"$PY" -m chora expert
"$PY" -m chora generate
"$PY" -m chora curate
"$PY" -m chora download
"$PY" -m chora train
"$PY" -m chora evaluate --policy expert
"$PY" -m chora evaluate --policy student
echo "Done. Start the lab with: .venv/bin/python -m chora serve"
