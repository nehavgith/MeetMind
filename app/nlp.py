"""MeetMind NLP pipeline.

Techniques used (map these to your syllabus):
  * Sentence segmentation, tokenisation, lemmatisation      -> spaCy
  * Named Entity Recognition (PERSON, DATE, TIME)            -> spaCy NER
  * Dependency parsing (subject / root verb of a task)       -> spaCy parser
  * Extractive summarisation (TextRank over TF-IDF graph)    -> scikit-learn + numpy
  * Keyword / keyphrase extraction (noun chunks + TF-IDF)    -> spaCy + scikit-learn
  * Topic segmentation into chapters (TextTiling-style)      -> TF-IDF cosine similarity
  * Rule-based intent detection (action / decision / question)
"""
import re
from collections import Counter

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# ---------- lexicons for intent detection ----------
ACTION_CUES = re.compile(
    r"\b(i('| wi)ll|i will|i am going to|i'm going to|we need to|will (send|share|prepare|update|fix|"
    r"check|review|draft|finish|complete|schedule|follow|look|set|create|write|test|deploy|book|call|email|"
    r"submit|upload|organi[sz]e|arrange|handle|take)|need(s)? to|has to|have to|must|can you|could you|"
    r"please|action item|take care of|follow up|follow-up|make sure|assign(ed)? to|responsible for|own(s)? this)\b",
    re.I,
)
DECISION_CUES = re.compile(
    r"\b(we (have )?(decided|agreed|finali[sz]ed|chose|will go with|are going with)|let'?s go with|decision is|"
    r"agreed|final(ly)? (call|decision)|we('| a)re going with|it'?s settled|confirmed that|"
    r"we will use|we'll use|go ahead with|sign(ed)? off)\b",
    re.I,
)
OPEN_CUES = re.compile(
    r"\b(not sure|still need to (figure|decide)|open question|tbd|to be decided|unclear|no idea|"
    r"haven'?t decided|need more (info|data|clarity)|let'?s revisit|park(ing)? (this|that)|come back to)\b",
    re.I,
)
URGENT = re.compile(r"\b(urgent|asap|immediately|critical|today|tonight|right away|blocker|high priority|eod)\b", re.I)
DONE_CUES = re.compile(r"\b(done|finished|completed|sent|shared|submitted|fixed|closed|resolved|wrapped up|delivered|merged|created|approved|met)\b", re.I)
NOT_DONE = re.compile(r"\b(not|n't|yet|still|pending|working on|in progress|delayed|stuck)\b", re.I)
FIRST_PERSON = re.compile(r"^\s*(ok(ay)?,?\s*|sure,?\s*|yes,?\s*|yeah,?\s*|so,?\s*)?(i|i'll|i will|i am|i'm)\b", re.I)
SECOND_PERSON = re.compile(r"\b(can you|could you|would you|you (will|should|need|must|take)|you'll)\b", re.I)
FILLER_STOP = {"thing", "things", "lot", "way", "bit", "today", "everyone", "guys", "something", "anything", "time", "one"}


def _sentences(segments):
    sents = []
    for seg in segments:
        doc = nlp(seg["text"])
        n = max(1, len(list(doc.sents)))
        for i, s in enumerate(doc.sents):
            txt = s.text.strip()
            if len(txt) < 3:
                continue
            sents.append({
                "text": txt, "doc": s, "seg": seg["id"], "speaker": seg["speaker"],
                "start": seg["start"] + int(i * 4 / n),
            })
    return sents


# ---------- action items ----------
def _owner(sent, speakers, next_speaker):
    span, text = sent["doc"], sent["text"]
    people = [e.text for e in span.ents if e.label_ == "PERSON"]
    # explicit name as grammatical subject: "Rahul will send ..."
    for tok in span:
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ == "PROPN":
            return tok.text
    if FIRST_PERSON.search(text) or any(t.dep_ == "nsubj" and t.lower_ == "i" for t in span):
        return sent["speaker"]
    # vocative / addressee: "Priya, can you ..." or "... by Friday, Arjun?"
    for p in people:
        first = p.split()[0]
        if first.lower() not in sent["speaker"].lower():
            return first
    for sp in speakers:
        if sp != sent["speaker"] and re.search(rf"\b{re.escape(sp.split()[0])}\b", text, re.I):
            return sp
    if (SECOND_PERSON.search(text) or re.match(r"\s*please\b", text, re.I)) and next_speaker:
        return next_speaker
    if re.search(r"\bwe\b", text, re.I):
        return "Team"
    return "Unassigned"


def _clean_task(text):
    t = re.sub(r"^(ok(ay)?|so|yeah|yes|sure|alright|and|also|then|great)[,\s]+", "", text, flags=re.I)
    t = re.sub(r"^\w+,\s*(can|could|would) you\s+", "", t, flags=re.I)
    t = re.sub(r"^(can|could|would) you( please)?\s+", "", t, flags=re.I)
    t = re.sub(r"^(i|we|you|he|she|they|\w+) ('ll|will|need to|needs to|has to|have to|should|must|am going to|is going to|are going to)\s+", "", t, flags=re.I)
    t = re.sub(r"^i'll\s+|^we'll\s+|^please\s+", "", t, flags=re.I)
    t = t.rstrip(" ?.!")
    return t[:1].upper() + t[1:] if t else text


def extract_actions(sents, speakers):
    out = []
    for i, s in enumerate(sents):
        if not ACTION_CUES.search(s["text"]) or len(s["text"].split()) < 4:
            continue
        if s["text"].endswith("?") and not SECOND_PERSON.search(s["text"]):
            continue
        if OPEN_CUES.search(s["text"]) or DECISION_CUES.search(s["text"]) or re.search(r"\b(depends on|whether)\b", s["text"], re.I):
            continue
        nxt = next((x["speaker"] for x in sents[i + 1:] if x["speaker"] != s["speaker"]), None)
        dates = [e.text for e in s["doc"].ents if e.label_ in ("DATE", "TIME")]
        deadline = dates[0] if dates else None
        priority = "High" if URGENT.search(s["text"]) else ("Medium" if deadline else "Low")
        owner = _owner(s, speakers, nxt)
        # skip "Sure, I will do X" replies that just confirm the request made right before
        if out and out[-1]["owner"] == owner and out[-1]["deadline"] == deadline and s["seg"] - out[-1]["seg"] <= 1:
            continue
        out.append({
            "task": _clean_task(s["text"]), "owner": owner, "deadline": deadline,
            "priority": priority, "speaker": s["speaker"], "start": s["start"], "seg": s["seg"], "quote": s["text"],
        })
    return out


def extract_by_cue(sents, cue, min_words=4):
    return [{"text": s["text"], "speaker": s["speaker"], "start": s["start"], "seg": s["seg"]}
            for s in sents if cue.search(s["text"]) and len(s["text"].split()) >= min_words]


def extract_questions(sents):
    qs = []
    for i, s in enumerate(sents):
        t = s["text"]
        if OPEN_CUES.search(t):
            qs.append(s)
        elif t.endswith("?") and len(t.split()) >= 6 and not SECOND_PERSON.search(t):
            # unanswered if the next two sentences contain no decision
            follow = " ".join(x["text"] for x in sents[i + 1:i + 3])
            if not DECISION_CUES.search(follow):
                qs.append(s)
    return [{"text": s["text"], "speaker": s["speaker"], "start": s["start"], "seg": s["seg"]} for s in qs]


# ---------- summarisation (TextRank) ----------
def textrank(sents, k=5):
    texts = [s["text"] for s in sents]
    if len(texts) <= k:
        return sents
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    rows = sim.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    m = sim / rows
    score = np.ones(len(texts)) / len(texts)
    for _ in range(50):
        score = 0.15 / len(texts) + 0.85 * m.T @ score
    # reward informative sentences, penalise chit-chat
    lengths = np.array([len(t.split()) for t in texts])
    score = score * np.clip(lengths / 12, 0.3, 1.3)
    top = sorted(np.argsort(-score)[:k])
    return [sents[i] for i in top]


# ---------- keywords ----------
GREET = re.compile(r"\b(morning|afternoon|evening|hello|hi|thanks|class|sir|question)\b", re.I)


def keywords(full_text, n=12):
    doc = nlp(full_text)
    chunks = []
    skip = {i for e in doc.ents if e.label_ in ("PERSON", "DATE", "TIME", "CARDINAL", "ORDINAL") for i in range(e.start, e.end)}
    for nc in doc.noun_chunks:
        if any(t.i in skip for t in nc) or GREET.search(nc.text) or "." in nc.text:
            continue
        toks = [t for t in nc if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "ADJ")]
        if not toks or toks[-1].pos_ == "ADJ":
            continue
        toks = toks[-3:]
        phrase = " ".join(t.lemma_.lower() if t.pos_ != "PROPN" or not t.text.isupper() else t.text for t in toks)
        if len(phrase) > 2 and phrase.lower() not in FILLER_STOP:
            chunks.append(phrase)
    counts = Counter(chunks)
    ranked = sorted(counts.items(), key=lambda kv: (-(kv[1] * (1 + 0.5 * (len(kv[0].split()) - 1))), kv[0]))
    return [{"term": t, "count": c} for t, c in ranked[:n]]


# ---------- chapters (TextTiling-style) ----------
def chapters(sents, window=4):
    if len(sents) < window * 3:
        return [{"title": "Full discussion", "start": sents[0]["start"] if sents else 0, "seg": sents[0]["seg"] if sents else 0, "sentences": len(sents)}]
    texts = [s["text"] for s in sents]
    vec = TfidfVectorizer(stop_words="english").fit(texts)
    gaps = []
    for i in range(window, len(texts) - window + 1):
        a = vec.transform([" ".join(texts[i - window:i])])
        b = vec.transform([" ".join(texts[i:i + window])])
        gaps.append((i, cosine_similarity(a, b)[0, 0]))
    sims = np.array([g[1] for g in gaps])
    depth = []
    for j in range(len(sims)):
        left, right = sims[:j + 1].max(), sims[j:].max()
        depth.append((left - sims[j]) + (right - sims[j]))
    depth = np.array(depth)
    cutoff = depth.mean() + 0.5 * depth.std()
    bounds, last = [0], 0
    for j in np.argsort(-depth):
        i = gaps[j][0]
        if depth[j] >= cutoff and all(abs(i - b) >= window * 2 for b in bounds) and len(bounds) < 7:
            bounds.append(i)
    bounds = sorted(bounds) + [len(sents)]
    result = []
    for a, b in zip(bounds, bounds[1:]):
        block = " ".join(texts[a:b])
        kws = keywords(block, n=2)
        title = " & ".join(k["term"][:1].upper() + k["term"][1:] for k in kws) or f"Part {len(result) + 1}"
        result.append({"title": title, "start": sents[a]["start"], "seg": sents[a]["seg"], "sentences": b - a})
    return result


# ---------- speaker analytics ----------
def speaker_stats(segments):
    words = Counter()
    for s in segments:
        words[s["speaker"]] += len(s["text"].split())
    total = sum(words.values()) or 1
    return [{"speaker": sp, "words": w, "share": round(100 * w / total, 1)} for sp, w in words.most_common()]


def analyze(segments):
    sents = _sentences(segments)
    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    full = " ".join(s["text"] for s in segments)
    tldr = textrank(sents, k=5)
    duration = (segments[-1]["start"] + 10) if segments else 0
    return {
        "tldr": [{"text": s["text"], "speaker": s["speaker"], "start": s["start"], "seg": s["seg"]} for s in tldr],
        "actions": extract_actions(sents, speakers),
        "decisions": extract_by_cue(sents, DECISION_CUES),
        "questions": extract_questions(sents),
        "keywords": keywords(full),
        "chapters": chapters(sents),
        "speakers": speaker_stats(segments),
        "stats": {"words": len(full.split()), "sentences": len(sents), "speakers": len(speakers), "duration": duration},
    }


# ---------- cross-meeting follow-up detection ----------
def follow_up_matches(pending_tasks, segments, threshold=0.28):
    """For each earlier pending task, find whether this meeting mentions it,
    and whether it was mentioned as done. Returns {task_id: {status, quote, start, seg}}."""
    if not pending_tasks or not segments:
        return {}
    sents = _sentences(segments)
    texts = [s["text"] for s in sents]
    corpus = texts + [t["task"] for t in pending_tasks]
    vec = TfidfVectorizer(stop_words="english").fit(corpus)
    sim = cosine_similarity(vec.transform([t["task"] for t in pending_tasks]), vec.transform(texts))
    out = {}
    for i, task in enumerate(pending_tasks):
        j = int(sim[i].argmax())
        if sim[i, j] >= threshold:
            s = sents[j]
            status = "completed" if DONE_CUES.search(s["text"]) and not NOT_DONE.search(s["text"]) else "carried"
            out[task["id"]] = {"status": status, "quote": s["text"], "start": s["start"], "seg": s["seg"], "score": round(float(sim[i, j]), 2)}
    return out
