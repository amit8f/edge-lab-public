"""AUTO1's agent step (research/prereg/AUTO1.md): ask a free model for one change to candidate.py.

    python -m auto.agent --candidate auto/candidate.py --trials research/results/AUTO1/trials.jsonl --out /tmp/new.py

Writes the proposed file and prints its IDEA line. Never runs the code (the harness does, offline).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from tracks.news import llm

HISTORY = 15
# free models, each with its own daily quota, tried in order
GROQ_MODELS = ("openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "llama-3.3-70b-versatile")
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")


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
    if os.environ.get("GROQ_API_KEY"):
        head = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"}
        for model in GROQ_MODELS:
            body = {"model": model, "temperature": 0.7, "messages": [{"role": "user", "content": text}],
                    **({"reasoning_effort": "medium"} if "gpt-oss" in model else {})}
            try:
                j = llm._post("https://api.groq.com/openai/v1/chat/completions", head, body, "groq")
                return j["choices"][0]["message"]["content"]
            except (llm.QuotaExhausted, RuntimeError, OSError) as e:
                print(f"[agent] {model}: {str(e)[:120]}", file=sys.stderr)
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            j = llm._post(url, {"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip()},
                          {"contents": [{"role": "user", "parts": [{"text": text}]}],
                           "generationConfig": {"temperature": 0.7}}, "gemini")
            parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except (llm.QuotaExhausted, RuntimeError, OSError) as e:
            print(f"[agent] {model}: {str(e)[:120]}", file=sys.stderr)
    raise llm.QuotaExhausted("every free model is out of quota")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="auto/candidate.py")
    ap.add_argument("--trials", default="research/results/AUTO1/trials.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--program", default=str(Path(__file__).parent / "program.md"))
    a = ap.parse_args()
    tf = Path(a.trials)
    trials = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()] if tf.exists() else []
    program = Path(a.program).read_text()
    idea, code = extract(ask(prompt(program, Path(a.candidate).read_text(), trials)))
    Path(a.out).write_text(code or "")
    print(json.dumps({"idea": idea, "has_code": code is not None}))


if __name__ == "__main__":
    main()
