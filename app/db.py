"""SQLite storage: this is MeetMind's cross-meeting memory."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "meetmind.db"
DB_PATH.parent.mkdir(exist_ok=True)


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS meetings(
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, created TEXT,
            audio TEXT, segments TEXT, analysis TEXT);
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER, task TEXT, owner TEXT,
            deadline TEXT, priority TEXT, status TEXT DEFAULT 'pending', start INTEGER, seg INTEGER,
            quote TEXT, carried INTEGER DEFAULT 0, history TEXT DEFAULT '[]');
        """)


def save_meeting(title, segments, analysis, audio=None):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO meetings(title, created, audio, segments, analysis) VALUES (?,?,?,?,?)",
            (title, datetime.now().isoformat(timespec="minutes"), audio, json.dumps(segments), json.dumps(analysis)))
        mid = cur.lastrowid
        for a in analysis["actions"]:
            c.execute(
                "INSERT INTO tasks(meeting_id, task, owner, deadline, priority, start, seg, quote) VALUES (?,?,?,?,?,?,?,?)",
                (mid, a["task"], a["owner"], a["deadline"], a["priority"], a["start"], a["seg"], a["quote"]))
        return mid


def pending_before(meeting_id=None):
    with conn() as c:
        q = "SELECT * FROM tasks WHERE status='pending'"
        args = ()
        if meeting_id:
            q += " AND meeting_id < ?"
            args = (meeting_id,)
        return [dict(r) for r in c.execute(q, args)]


def apply_followups(meeting_id, matches):
    with conn() as c:
        for tid, m in matches.items():
            row = c.execute("SELECT history, carried FROM tasks WHERE id=?", (tid,)).fetchone()
            hist = json.loads(row["history"])
            hist.append({"meeting_id": meeting_id, **m})
            if m["status"] == "completed":
                c.execute("UPDATE tasks SET status='done', history=? WHERE id=?", (json.dumps(hist), tid))
            else:
                c.execute("UPDATE tasks SET carried=carried+1, history=? WHERE id=?", (json.dumps(hist), tid))


def list_meetings():
    with conn() as c:
        rows = c.execute("SELECT id, title, created, analysis FROM meetings ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            a = json.loads(r["analysis"])
            counts = c.execute(
                "SELECT SUM(status='pending') p, SUM(status='done') d FROM tasks WHERE meeting_id=?", (r["id"],)).fetchone()
            out.append({"id": r["id"], "title": r["title"], "created": r["created"], "stats": a["stats"],
                        "speakers": [s["speaker"] for s in a["speakers"]], "tldr": a["tldr"][0]["text"] if a["tldr"] else "",
                        "pending": counts["p"] or 0, "done": counts["d"] or 0})
        return out


def get_meeting(mid):
    with conn() as c:
        r = c.execute("SELECT * FROM meetings WHERE id=?", (mid,)).fetchone()
        if not r:
            return None
        tasks = [dict(t) for t in c.execute("SELECT * FROM tasks WHERE meeting_id=? ORDER BY id", (mid,))]
        followed = []
        for t in c.execute("SELECT t.*, m.title mtitle FROM tasks t JOIN meetings m ON m.id=t.meeting_id WHERE t.meeting_id < ?", (mid,)):
            hist = [h for h in json.loads(t["history"]) if h["meeting_id"] == mid]
            if hist:
                followed.append({**dict(t), "update": hist[-1]})
        return {"id": r["id"], "title": r["title"], "created": r["created"], "audio": r["audio"],
                "segments": json.loads(r["segments"]), "analysis": json.loads(r["analysis"]),
                "tasks": tasks, "followups": followed}


def all_tasks():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT t.*, m.title mtitle, m.created mcreated, "
            "(SELECT COUNT(*) FROM meetings m2 WHERE m2.id > t.meeting_id) meetings_since "
            "FROM tasks t JOIN meetings m ON m.id=t.meeting_id ORDER BY t.status DESC, t.id DESC")]


def set_status(tid, status):
    with conn() as c:
        c.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))


def delete_meeting(mid):
    with conn() as c:
        c.execute("DELETE FROM tasks WHERE meeting_id=?", (mid,))
        c.execute("DELETE FROM meetings WHERE id=?", (mid,))
