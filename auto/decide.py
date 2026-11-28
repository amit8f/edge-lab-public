"""AUTO1's keep-or-throw step: append the trial to the log and keep the new candidate only if it beats the best
kept score by KEEP_MARGIN (research/prereg/AUTO1.md).

    python -m auto.decide --new /tmp/new.py --idea-json /tmp/idea.json --score-json /tmp/score.json
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from auto.harness import KEEP_MARGIN


def best_kept(trials):
    return max([t["score"] for t in trials if t.get("kept") and t.get("score") is not None], default=None)


def keep(score, best):
    return score is not None and (best is None or score >= best + KEEP_MARGIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--idea-json", required=True)
    ap.add_argument("--score-json", required=True)
    ap.add_argument("--candidate", default="auto/candidate.py")
    ap.add_argument("--trials", default="research/results/AUTO1/trials.jsonl")
    a = ap.parse_args()
    tf = Path(a.trials)
    tf.parent.mkdir(parents=True, exist_ok=True)
    trials = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()] if tf.exists() else []
    idea = json.loads(Path(a.idea_json).read_text() or "{}")
    raw = Path(a.score_json).read_text().strip().splitlines()
    res, err = None, None
    try:
        res = json.loads(raw[-1]) if raw else None
    except json.JSONDecodeError:
        err = (raw[-1] if raw else "")[:300]
    if res is None and err is None:
        err = "no score (crash, timeout or rejected)"
    score = res.get("sharpe") if res else None
    best = best_kept(trials)
    first = not trials                     # trial 0 scores the starting candidate itself
    kept = first or keep(score, best)
    src = Path(a.new).read_text()
    row = {"n": len(trials), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "idea": idea.get("idea", ""),
           "score": score, "result": res, "kept": bool(kept and score is not None), "error": err,
           "sha256": hashlib.sha256(src.encode()).hexdigest()}
    with tf.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    if row["kept"] and not first:
        shutil.copyfile(a.new, a.candidate)
    print(json.dumps({k: row[k] for k in ("n", "score", "kept", "idea")}))


if __name__ == "__main__":
    main()
