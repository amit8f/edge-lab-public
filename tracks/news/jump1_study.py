"""JUMP1 (research/prereg/JUMP1.md): do NX3-HI's buys on news out by the open, with a big jump already made before
entry, make money on data LONG1 never saw? Reads HI3's and HI3K's saved day files and settled trades; no download.

    python -m tracks.news.jump1_study --hi3 cache/hi3 --hi3k cache/hi3k --cut research/results/LONG/jump_cut.json \
        --out research/results/JUMP1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tracks.main.longterm.stats import nw_tstat

ARM = "XGB-HI"          # NX3-HI as HI3 replays it
PRE_OPEN_MIN = 1.0      # news at or before 9:31 (entry one minute after t), as LONG1's "before or at the open"
MIN_TRADES = 30


def jump_of(inputs):
    """Entry price / previous close - 1 from the decided inputs: (1 + r_open)(1 + r_react) - 1."""
    ro, rr = inputs.get("r_open"), inputs.get("r_react")
    if ro is None or rr is None or not (np.isfinite(ro) and np.isfinite(rr)):
        return np.nan
    return (1 + ro) * (1 + rr) - 1


def buys(cache, source):
    """XGB-HI's settled buys with each one's minutes from the open and jump before entry."""
    rows = []
    for f in sorted((Path(cache) / "days").glob("*.json")):
        rec = json.loads(f.read_text())
        g = Path(cache) / "settled" / f"{rec['day']}.parquet"
        if not g.exists():
            continue
        t = pd.read_parquet(g)
        if t.empty:
            continue
        t = t[(t["arm"] == ARM) & (t["role"] == "picks") & (t["side"] > 0)]
        ev = {x["symbol"]: x for x in rec["events"] if x.get("status") == "decided"}
        for r in t.itertuples():
            x = ev.get(r.symbol)
            if x is None:
                continue
            inp = x.get("inputs") or {}
            rows.append({"source": source, "part": r.part, "day": r.day, "symbol": r.symbol, "net": r.net,
                         "mins_open": inp.get("mins_open"), "jump": jump_of(inp)})
    return pd.DataFrame(rows, columns=["source", "part", "day", "symbol", "net", "mins_open", "jump"])


def verdict(b, cut):
    """The primary test on pooled buys `b` (net, mins_open, jump, day)."""
    b = b.dropna(subset=["net"])
    sel = (b["mins_open"].astype(float) <= PRE_OPEN_MIN) & (b["jump"].astype(float) > cut)
    j, rest = b[sel], b[~sel]
    out = {"buys": len(b), "jump1": len(j), "rest": len(rest)}
    if len(j) < MIN_TRADES:
        return out | {"passes": None, "note": f"under {MIN_TRADES} JUMP1 buys: no verdict"}
    daily = j.groupby("day")["net"].mean()
    m, _se, t = nw_tstat(daily.to_numpy(), 5) if len(daily) > 2 else (float(daily.mean()), None, None)
    w = stats.ttest_ind(j["net"], rest["net"], equal_var=False) if len(rest) > 1 else None
    gap = float(j["net"].mean() - rest["net"].mean()) if len(rest) else None
    out |= {"net_per_trade": float(j["net"].mean()), "win": float((j["net"] > 0).mean()), "days": int(daily.size),
            "daily_mean": m, "daily_t": t, "rest_net_per_trade": float(rest["net"].mean()) if len(rest) else None,
            "gap": gap, "gap_t": float(w.statistic) if w is not None else None}
    out["passes"] = bool(out["net_per_trade"] > 0 and t is not None and t >= 2 and gap is not None and gap > 0
                         and out["gap_t"] is not None and out["gap_t"] >= 1.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi3", default="cache/hi3")
    ap.add_argument("--hi3k", default="cache/hi3k")
    ap.add_argument("--cut", default="research/results/LONG/jump_cut.json")
    ap.add_argument("--out", default="research/results/JUMP1")
    a = ap.parse_args()
    cut = json.loads(Path(a.cut).read_text())["jump_cut"]
    b = pd.concat([buys(a.hi3, "HI3"), buys(a.hi3k, "HI3K")], ignore_index=True)
    res = {"cut": cut, "pooled": verdict(b, cut),
           "by_source": {s: verdict(b[b["source"] == s], cut) for s in ("HI3", "HI3K")}}
    cells = {}
    for name, m in {"pre_open": b["mins_open"].astype(float) <= PRE_OPEN_MIN,
                    "later": b["mins_open"].astype(float) > PRE_OPEN_MIN}.items():
        for jn, jm in {"high_jump": b["jump"].astype(float) > cut, "other_jump": b["jump"].astype(float) <= cut}.items():
            x = b[m & jm]["net"].dropna()
            cells[f"{name}/{jn}"] = {"n": len(x), "net": float(x.mean()) if len(x) else None}
    res["cells"] = cells
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "JUMP1.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
