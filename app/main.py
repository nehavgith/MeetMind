"""MeetMind API server.  Run:  python -m uvicorn app.main:app --reload"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, nlp
from .parser import parse_transcript

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "data" / "audio"
UPLOADS.mkdir(parents=True, exist_ok=True)
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".flac"}

app = FastAPI(title="MeetMind")
db.init()
_whisper = None


def transcribe(path):
    """Speech-to-text with faster-whisper (optional dependency)."""
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(400, "Audio needs faster-whisper. Run: pip install faster-whisper  (or paste a transcript instead)")
    if _whisper is None:
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(str(path), vad_filter=True)
    out = [{"start": int(s.start), "speaker": "Speaker", "text": s.text.strip()} for s in segs]
    for i, s in enumerate(out):
        s["id"] = i
    return out


def process(title, segments, audio=None):
    if not segments:
        raise HTTPException(400, "Could not find any speech/text in the input.")
    analysis = nlp.analyze(segments)
    mid = db.save_meeting(title, segments, analysis, audio)
    # cross-meeting memory: check earlier pending tasks against this meeting
    matches = nlp.follow_up_matches(db.pending_before(mid), segments)
    db.apply_followups(mid, matches)
    return {"id": mid}


@app.post("/api/meetings")
async def create_meeting(title: str = Form("Untitled meeting"), text: str = Form(""), file: UploadFile | None = File(None)):
    audio_name = None
    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext in AUDIO_EXT:
            audio_name = f"{uuid.uuid4().hex}{ext}"
            with open(UPLOADS / audio_name, "wb") as f:
                shutil.copyfileobj(file.file, f)
            segments = transcribe(UPLOADS / audio_name)
        else:
            raw = (await file.read()).decode("utf-8", errors="ignore")
            segments = parse_transcript(raw, file.filename)
    else:
        segments = parse_transcript(text)
    return process(title.strip() or "Untitled meeting", segments, audio_name)


@app.get("/api/meetings")
def meetings():
    return db.list_meetings()


@app.get("/api/meetings/{mid}")
def meeting(mid: int):
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "Meeting not found")
    return m


@app.delete("/api/meetings/{mid}")
def remove(mid: int):
    db.delete_meeting(mid)
    return {"ok": True}


@app.get("/api/tasks")
def tasks():
    return db.all_tasks()


class StatusIn(BaseModel):
    status: str


@app.patch("/api/tasks/{tid}")
def update_task(tid: int, body: StatusIn):
    if body.status not in ("pending", "done"):
        raise HTTPException(400, "status must be pending or done")
    db.set_status(tid, body.status)
    return {"ok": True}


@app.get("/api/samples")
def samples():
    return [{"name": p.stem.replace("_", " ").title(), "file": p.name, "text": p.read_text()}
            for p in sorted((ROOT / "samples").glob("*.txt"))]


@app.get("/audio/{name}")
def audio(name: str):
    p = UPLOADS / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
