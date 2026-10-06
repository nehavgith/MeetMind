#!/usr/bin/env bash
# MeetMind launcher for macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up (takes 2-3 minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/python -m spacy download en_core_web_sm
fi
echo "MeetMind is running at http://localhost:8000  (Ctrl+C to stop)"
( sleep 2; (open http://localhost:8000 || xdg-open http://localhost:8000) >/dev/null 2>&1 ) &
.venv/bin/python -m uvicorn app.main:app --port 8000
