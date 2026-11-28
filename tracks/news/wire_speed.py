"""WIRE0 (research/prereg/WIRE0.md, descriptive): how much earlier do the free press-release wires show a
headline than Benzinga (the source NX3 trades on)? Matches the listener's log (core/listener.py, state-listen)
with Benzinga's headlines from Alpaca by normalised-title hash.

    python -m tracks.news.wire_speed --listen state-listen --out research/results/WIRE0 [--days 20]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import listener as LI
from tracks.main.longterm import news_history as H

WIRES = ("PR Newswire", "Business Wire")


def wire_items(root, days):
    rows = []
    for day in days:
        for ln in LI.lines_of(root, day):
            r = json.loads(ln)
            if r.get("kind") == "item" and r.get("source") in WIRES and r.get("title_sha"):
                rows.append({"source": r["source"], "sha": r["title_sha"], "first_seen": r["first_seen_utc"],
                             "window_s": r.get("window_s"), "tickers": ",".join(r.get("tickers") or [])})
    d = pd.DataFrame(rows, columns=["source", "sha", "first_seen", "window_s", "tickers"])
    d["first_seen"] = pd.to_datetime(d["first_seen"], utc=True, format="ISO8601")
    return d


def leads(w, bz):
    """Benzinga created_at minus our first sight, seconds, per matched headline (first Benzinga copy). Items
    from a day's first read (window_s null: an old backlog, not a discovery) are left out."""
    w = w[w["window_s"].notna()].sort_values("first_seen").drop_duplicates("sha")
    b = bz.assign(sha=bz["headline"].map(LI.title_sha), created=pd.to_datetime(bz["created_at"], utc=True))
    b = b.sort_values("created").drop_duplicates("sha")[["sha", "created", "symbols"]]
    m = w.merge(b, on="sha", how="inner")
    m["lead_s"] = (m["created"] - m["first_seen"]).dt.total_seconds()
    return m


def summary(m):
    out = {}
    for s, g in [("all", m), *m.groupby("source")]:
        x = g["lead_s"].to_numpy(float)
        if not len(x):
            continue
        out[s] = {"matched": len(x), "median_lead_s": float(np.median(x)), "p25_s": float(np.percentile(x, 25)),
                  "p75_s": float(np.percentile(x, 75)), "share_earlier_30s": float((x >= 30).mean()),
                  "share_earlier_60s": float((x >= 60).mean()), "share_earlier_5min": float((x >= 300).mean()),
                  "with_ticker": int((g["symbols"].fillna("") != "").sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="state-listen")
    ap.add_argument("--out", default="research/results/WIRE0")
    ap.add_argument("--days", type=int, default=20)
    a = ap.parse_args()
    all_days = sorted(p.name for p in (Path(a.listen) / "listen" / "live").iterdir() if p.is_dir())
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    days = [d for d in all_days if d < today][-a.days:]
    w = wire_items(a.listen, days)
    parts = []
    for d in days:
        s = pd.Timestamp(d, tz="America/New_York").tz_convert("UTC")
        parts.append(H.fetch_window(s, s + pd.Timedelta(days=1)))
    bz = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["headline", "created_at", "symbols"])
    m = leads(w, bz)
    res = {"days": [days[0], days[-1], len(days)] if days else None, "wire_items": len(w),
           "benzinga_items": len(bz), "summary": summary(m)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "WIRE0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
