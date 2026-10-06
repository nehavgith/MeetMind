_# MeetMind: AI Meeting & Lecture Intelligence System

NLP project: BE-Aiml
Neha Vengurlekar -- 224A9069
Srushti Dhotare -- 224A9065
Shravani Bhosale -- 123A9050
Om Revankar -- 123A9038

MeetMind turns a meeting or lecture (audio or transcript) into a structured record:
**who promised what, what was decided, and what is still open**, with every item linked
back to the exact moment in the transcript where it was said.

## Run it

You need **Python 3.10 to 3.12** installed.

- **Windows:** double-click `run.bat`
- **macOS / Linux:** open a terminal in this folder and run `./run.sh`

The first run sets up a virtual environment and downloads the spaCy English model
(2 to 3 minutes, needs internet). After that it starts in seconds.
The app opens at **http://localhost:8000**.

Manual way, if you prefer:
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m uvicorn app.main:app --reload
```

### Audio support (optional)
Pasting or uploading a transcript works out of the box. To upload audio files
(.mp3, .wav, .m4a) as well:
```bash
pip install -r requirements-audio.txt
```
This adds faster-whisper (speech-to-text). The first audio upload downloads the
Whisper "base" model (~150 MB).


To reset everything, stop the app and delete `data/meetmind.db`.

## Features

| Feature | How it works |
|---|---|
| Transcript input | Paste text, upload `.txt`, or upload Zoom / Teams / Meet `.vtt` exports (speaker names + timestamps) |
| Speech to text | faster-whisper (optional) |
| Key-point summary | Extractive summarisation with **TextRank** over a TF-IDF sentence graph |
| Action items | Rule-based intent detection + **spaCy NER** (PERSON, DATE, TIME) + **dependency parsing** to find the owner |
| Decisions / open questions | Lexical cue patterns + unanswered-question check |
| Chapters | **TextTiling-style topic segmentation** using cosine similarity between sentence blocks |
| Key terms | Noun-chunk extraction with lemmatisation, ranked by frequency |
| Speaker insights | Talk-time share per speaker |

### The two features a general chatbot doesn't provide
1. **Clickable evidence.** Every action item, decision and question carries the speaker
   and timestamp it came from, and is highlighted inside the transcript. One click jumps
   to that moment, so every claim can be verified.
2. **Cross-meeting memory.** Tasks are stored in SQLite. When a new meeting is analysed,
   MeetMind compares every earlier pending task with the new transcript (TF-IDF cosine
   similarity) and detects completion vs. "still open" language, including negation
   ("not finished yet"). A chatbot forgets everything when the chat ends.

## Project structure
```
meetmind/
├── app/
│   ├── main.py      FastAPI server and REST API
│   ├── parser.py    Transcript parsing (.vtt, timestamped text, plain text)
│   ├── nlp.py       The NLP pipeline (all techniques listed above)
│   └── db.py        SQLite storage, the cross-meeting memory
├── static/
│   └── index.html   Single-page UI (no build step)
├── samples/         Demo transcripts (two linked meetings + one lecture)
├── data/            Created at runtime: database and uploaded audio
├── run.bat / run.sh One-click launchers
└── requirements.txt
```

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/meetings` | Analyse a meeting (form fields: `title`, `text` or `file`) |
| GET | `/api/meetings` | List meetings |
| GET | `/api/meetings/{id}` | Full analysis, transcript and follow-ups |
| DELETE | `/api/meetings/{id}` | Delete a meeting |
| GET | `/api/tasks` | All tasks across meetings |
| PATCH | `/api/tasks/{id}` | Mark a task `done` or `pending` |

Interactive API docs: http://localhost:8000/docs

## Future scope
Speaker diarization (pyannote), Hindi/Marathi transcripts (IndicTrans2), abstractive
summaries with a transformer model, live transcription during a call, calendar reminders.
_
