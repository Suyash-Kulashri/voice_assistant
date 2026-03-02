#!/usr/bin/env bash
# Run the voice assistant server. Only watches app/ and static/ (no .venv spam).
cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-dir app --reload-dir static
