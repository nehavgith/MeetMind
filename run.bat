@echo off
REM MeetMind launcher for Windows
cd /d "%~dp0"
if not exist .venv (
  echo First run: setting up, this takes 2-3 minutes...
  python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q -r requirements.txt
  .venv\Scripts\python -m spacy download en_core_web_sm
)
echo MeetMind is running at http://localhost:8000  (Ctrl+C to stop)
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --port 8000
