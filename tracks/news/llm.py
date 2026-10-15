"""N1 headline labels from two free-tier LLMs (research/prereg/N1.md). The prompt is fixed by the
pre-registration; models and cutoffs are fixed in research/results/N1/models.json.

Labels are cached one JSON line per headline per model, so a run that hits a daily quota stops and the
next run continues where it left off.
"""
import json
import os
import re
import time
from pathlib import Path

import requests

SYSTEM = ("You label financial news headlines. For each numbered headline about the named company, answer "
          "NEG if it is clearly bad news for that company's share price, POS if clearly good news, NEU "
          "otherwise (routine, mixed, unclear, or about someone else). Answer only with lines "
          "'number: LABEL'.")
BATCH = 50
LABELS = {"NEG", "POS", "NEU"}


class QuotaExhausted(Exception):
    """The provider's daily limit is used up; continue on the next run."""


def user_text(rows):
    """rows: [(symbol, headline)] -> the numbered user message."""
    return "\n".join(f"{i}. [{s}] {' '.join(str(h).split())}" for i, (s, h) in enumerate(rows, 1))


def parse(text, n):
    """Labels 1..n from a reply; anything missing or malformed is NEU (pre-registered)."""
    out = ["NEU"] * n
    for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*\**\s*(NEG|POS|NEU)\b", text or "", re.M | re.I):
        i = int(m.group(1))
        if 1 <= i <= n:
            out[i - 1] = m.group(2).upper()
    return out


def _post(url, headers, body, provider):
    for attempt in range(6):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 0) or 0)
            daily = "per day" in r.text.lower() or "perday" in r.text.lower() or wait > 120
            if daily or attempt == 5:
                raise QuotaExhausted(f"{provider}: {r.text[:200]}")
            time.sleep(max(wait, 15))
            continue
        if r.status_code >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{provider}: no answer after retries")


def gemini(model, text):
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingBudget": 0}}}
    j = _post(url, {"x-goog-api-key": key}, body, "gemini")
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def groq(model, text):
    key = os.environ["GROQ_API_KEY"]
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    j = _post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": f"Bearer {key}"}, body, "groq")
    return j["choices"][0]["message"]["content"]


CALLERS = {"gemini": gemini, "groq": groq}


def load_cache(path):
    p = Path(path)
    if not p.exists():
        return {}
    return {d["key"]: d["label"] for d in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if d}


def label_all(items, provider, model, cache_path, min_gap_sec, deadline=None, call=None):
    """items: [(key, symbol, headline)]. Labels every uncached item in batches of 50, appending to the cache.
    Returns True when every item is labelled, False when stopped by a quota or the deadline."""
    call = call or CALLERS[provider]
    done = load_cache(cache_path)
    todo = [x for x in items if x[0] not in done]
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[{provider}] {len(done)} cached, {len(todo)} to label")
    last = 0.0
    for b in range(0, len(todo), BATCH):
        if deadline is not None and time.monotonic() > deadline:
            return False
        chunk = todo[b:b + BATCH]
        time.sleep(max(0.0, min_gap_sec - (time.monotonic() - last)))
        last = time.monotonic()
        try:
            text = call(model, user_text([(s, h) for _, s, h in chunk]))
        except QuotaExhausted as e:
            print(f"[{provider}] daily quota reached after {b} new labels: {e}")
            return False
        labels = parse(text, len(chunk))
        with open(cache_path, "a", encoding="utf-8") as f:
            for (k, _, _), lab in zip(chunk, labels):
                f.write(json.dumps({"key": k, "label": lab}) + "\n")
    return True
