"""M6: analyst actions read from Benzinga headlines (research/prereg/M6.md). Pure functions."""
import re

import numpy as np
import pandas as pd

WINDOW_DAYS = 90
_I = re.IGNORECASE
UP = re.compile(r"\bupgrade[sd]?\b", _I)
DOWN = re.compile(r"\bdowngrade[sd]?\b", _I)
RAISE = re.compile(r"\b(?:raises|raised|boosts|lifts|increases)\b.{0,30}\bprice target\b|"
                   r"\bprice target\b\s+(?:raised|increased|boosted)\b", _I)
CUT = re.compile(r"\b(?:lowers|lowered|cuts|cut|reduces|trims|slashes)\b.{0,30}\bprice target\b|"
                 r"\bprice target\b\s+(?:lowered|cut|reduced)\b", _I)


def classify(headline, ratings_only=False):
    """+1 upgrade / target raise, -1 downgrade / target cut, 0 otherwise (pre-registered rules)."""
    h = headline or ""
    up, down = bool(UP.search(h)), bool(DOWN.search(h))
    if up and down:
        return 0
    if up:
        return 1
    if down:
        return -1
    if ratings_only:
        return 0
    r, c = bool(RAISE.search(h)), bool(CUT.search(h))
    if r and not c:
        return 1
    if c and not r:
        return -1
    return 0


def events(articles, ratings_only=False):
    """Articles (headline, created_at, symbols as a list or comma string) -> one row per counted action:
    (symbol, day = New York date of created_at, direction)."""
    a = articles.copy()
    syms = a["symbols"].map(lambda s: s if isinstance(s, (list, np.ndarray)) else
                            [x for x in str(s or "").split(",") if x])
    a = a[syms.map(len) == 1].assign(symbol=syms[syms.map(len) == 1].map(lambda s: s[0]))
    a["direction"] = a["headline"].map(lambda h: classify(h, ratings_only))
    a = a[a["direction"] != 0]
    ts = pd.to_datetime(a["created_at"], utc=True).dt.tz_convert("America/New_York")
    a["day"] = ts.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    a["hkey"] = a["headline"].str.strip().str.lower()
    a = a.sort_values("day")
    keep, last = [], {}
    for i, (s, h, d) in enumerate(zip(a["symbol"], a["hkey"], a["day"])):
        k = (s, h)
        if k in last and (d - last[k]).days <= 3:
            continue
        last[k] = d
        keep.append(i)
    return a.iloc[keep][["symbol", "day", "direction", "headline"]].reset_index(drop=True)


def trailing_counts(keys, ev, window_days=WINDOW_DAYS):
    """up and down counts per (symbol, date) over events with day in (date - window, date]."""
    k = keys.reset_index(drop=True)
    up = np.zeros(len(k), dtype=int)
    down = np.zeros(len(k), dtype=int)
    groups = {s: g.sort_values("day") for s, g in ev.groupby("symbol")}
    dates = pd.to_datetime(k["date"]).astype("datetime64[ns]").to_numpy()
    for i, (s, t) in enumerate(zip(k["symbol"].to_numpy(), dates)):
        g = groups.get(s)
        if g is None:
            continue
        d = g["day"].to_numpy()
        lo = np.searchsorted(d, t - np.timedelta64(window_days, "D"), side="right")
        hi = np.searchsorted(d, t, side="right")
        w = g["direction"].to_numpy()[lo:hi]
        up[i], down[i] = int((w > 0).sum()), int((w < 0).sum())
    return up, down


def score_m6(uni, ev, seed, window_days=WINDOW_DAYS):
    """Score = (up - down) / sqrt(up + down), 0 without actions; rank per date with the pre-registered
    tie-breaks (more actions first, then a seeded random order)."""
    d = uni.reset_index(drop=True).copy()
    up, down = trailing_counts(d[["symbol", "date"]], ev, window_days)
    tot = up + down
    d["an_up"], d["an_down"] = up, down
    d["score"] = np.where(tot > 0, (up - down) / np.sqrt(np.maximum(tot, 1)), 0.0)
    d["an_total"] = tot
    out = []
    for t, g in d.groupby("date", sort=True):
        g = g.copy()
        g["_rnd"] = np.random.default_rng(seed + t.toordinal()).random(len(g))
        g = g.sort_values(["score", "an_total", "_rnd"], ascending=[False, False, True])
        g["rank"] = np.arange(1, len(g) + 1)
        g["score_pct"] = g["score"].rank(pct=True)
        out.append(g.drop(columns="_rnd"))
    return pd.concat(out, ignore_index=True)
