#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Chora by Coherence"
echo "Open http://127.0.0.1:8765 in your browser."
echo "Keep this window open; press Ctrl-C to stop."
exec .venv/bin/python -m chora serve
