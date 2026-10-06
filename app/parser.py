"""Turns raw transcript input into a list of timed, speaker-labelled segments.

Supported inputs
  * Zoom / Teams / Google Meet .vtt exports (speaker names + timestamps)
  * Plain text with lines like  "[00:01:23] Rahul: I will send the report"
  * Plain text with lines like  "Rahul: I will send the report"   (time estimated)
  * Any plain paragraph text                                       (single speaker)
"""
import re

TS_LINE = re.compile(r"^\s*\[?(\d{1,2}:)?(\d{1,2}):(\d{2})(?:\.\d+)?\]?\s*[-–]?\s*(.*)$")
SPEAKER = re.compile(r"^([A-Z][\w .'-]{0,30}?)\s*:\s+(.+)$")
VTT_TIME = re.compile(r"(\d{2}:)?(\d{2}):(\d{2})\.\d{3}\s+-->\s+(\d{2}:)?(\d{2}):(\d{2})\.\d{3}")
VTT_VOICE = re.compile(r"<v\s+([^>]+)>(.*?)(</v>)?$")

WORDS_PER_SEC = 2.5  # ~150 words per minute, used when no timestamps exist


def _secs(h, m, s):
    return int(h.rstrip(":") if h else 0) * 3600 + int(m) * 60 + int(s)


def parse_vtt(text):
    segs, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        t = VTT_TIME.search(line)
        if t:
            cur = {"start": _secs(t.group(1), t.group(2), t.group(3))}
            continue
        if cur is not None and line:
            v = VTT_VOICE.match(line)
            speaker, body = (v.group(1).strip(), v.group(2)) if v else (None, line)
            if not v:
                m = SPEAKER.match(line)
                if m:
                    speaker, body = m.group(1), m.group(2)
            body = re.sub(r"<[^>]+>", "", body).strip()
            if body:
                segs.append({"start": cur["start"], "speaker": speaker or "Speaker", "text": body})
            cur = None
    return segs


def parse_text(text):
    segs, clock = [], 0.0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        start = None
        t = TS_LINE.match(line)
        if t and t.group(4):
            start = _secs(t.group(1), t.group(2), t.group(3))
            line = t.group(4)
        m = SPEAKER.match(line)
        speaker, body = (m.group(1).strip(), m.group(2).strip()) if m else (None, line)
        if speaker is None and segs and not t:
            # continuation line of the previous speaker
            segs[-1]["text"] += " " + body
            clock += len(body.split()) / WORDS_PER_SEC
            continue
        if start is None:
            start = clock
        segs.append({"start": int(start), "speaker": speaker or "Speaker", "text": body})
        clock = start + len(body.split()) / WORDS_PER_SEC
    return segs


def parse_transcript(text, filename=""):
    if filename.lower().endswith(".vtt") or text.lstrip().startswith("WEBVTT"):
        segs = parse_vtt(text)
    else:
        segs = parse_text(text)
    # merge consecutive lines from the same speaker so the transcript reads cleanly
    merged = []
    for s in segs:
        if merged and merged[-1]["speaker"] == s["speaker"] and s["start"] - merged[-1]["start"] < 20:
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(dict(s))
    for i, s in enumerate(merged):
        s["id"] = i
    return merged
