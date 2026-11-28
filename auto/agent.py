"""AUTO1's agent step (research/prereg/AUTO1.md): ask a free model for one change to candidate.py.

    python -m auto.agent --candidate auto/candidate.py --trials research/results/AUTO1/trials.jsonl --out /tmp/new.py

Writes the proposed file and prints its IDEA line. Never runs the code (the harness does, offline).
"""
import argparse
import json
import os
import re
from pathlib import Path

from tracks.news import llm

HISTORY = 15
GROQ_MODEL, GEMINI_MODEL = "openai/gpt-oss-120b", "gemini-2.5-flash"


def prompt(program, src, trials):
    hist = "\n".join(f"- score {t.get('score')}, {'KEPT' if t.get('kept') else 'thrown away'}: {t.get('idea', '')}"
                     + (f" (error: {t['error'][:120]})" if t.get("error") else "") for t in trials[-HISTORY:])
    best = max([t["score"] for t in trials if t.get("kept") and t.get("score") is not None], default=None)
    return (f"{program}\n\nBest kept score so far: {best}\n\nRecent trials:\n{hist or '- none yet'}\n\n"
            f"Current candidate.py:\n```python\n{src}\n```")


def extract(text):
    """(idea, code) from a reply; code None when there is no python block."""
    m = re.search(r"```python\s*\n(.*?)```", text or "", re.DOTALL)
    idea = re.search(r"^\s*IDEA:\s*(.+)$", text or "", re.MULTILINE)
    return (idea.group(1).strip()[:200] if idea else ""), (m.group(1) if m else None)


def ask(text):
    body = {"model": GROQ_MODEL, "temperature": 0.7, "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": text}]}
    if os.environ.get("GROQ_API_KEY"):
        try:
            j = llm._post("https://api.groq.com/openai/v1/chat/completions",
                          {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"}, body, "groq")
            return j["choices"][0]["message"]["content"]
        except llm.QuotaExhausted:
            pass
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    j = llm._post(url, {"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip()},
                  {"contents": [{"role": "user", "parts": [{"text": text}]}], "generationConfig": {"temperature": 0.7}},
                  "gemini")
    return "".join(p.get("text", "") for p in (j.get("candidates") or [{}])[0].get("content", {}).get("parts", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="auto/candidate.py")
    ap.add_argument("--trials", default="research/results/AUTO1/trials.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tf = Path(a.trials)
    trials = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()] if tf.exists() else []
    program = (Path(__file__).parent / "program.md").read_text()
    idea, code = extract(ask(prompt(program, Path(a.candidate).read_text(), trials)))
    Path(a.out).write_text(code or "")
    print(json.dumps({"idea": idea, "has_code": code is not None}))


if __name__ == "__main__":
    main()
