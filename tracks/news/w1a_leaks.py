"""W1a (research/prereg/W1.md): published leak and rumour reports by source group. Benzinga headlines
2016-01-01 .. 2025-09-23 that cite a listed source and carry a deal or short-report word; entry 30
minutes after the article, exit at the close of trading day +5, signed and SPY-adjusted, after costs.

Stages (each resumable; run in order with --stage all):
1. data: the shared W1 daily bars, filings and headline months (tracks/news/w1_data.py).
2. events: classify every headline, keep the first per symbol and source group in 5 trading days, add the
   universe row of the prior trading day, write coverage counts and the 50-event hand check.
3. details: author, summary and URL of every classified article (one news request per symbol-month).
4. gdelt: per-year article counts of the small and foreign outlets in GDELT (exploratory coverage).
5. measure: 1-minute SIP bars of the stock, SPY and a same-day random control; one JSON per event.
   Refuses to run until the hand check passes (>= 40 of 50).
6. report: per-group primary tests, the secondary gap test, hit rates, stories, coverage.

No raw headline text is written to the results folder (Alpaca's terms); the hand-check file carries the
article URL and only the matched source and deal words.

Usage: python -m tracks.news.w1a_leaks --cache <dir> --out <dir> [--stage data|events|details|gdelt|measure|report|all]
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import alpaca, stats
from core.env import alpaca_headers
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import l1_study as L1
from tracks.news import w1_data as D

ET = D.ET
SEED, N_BOOT = 20261017, 10_000
FIRST, LAST = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-09-24")   # article created_at in [FIRST, LAST)
WAIT_MIN, HOLD_DAYS, DEDUP_DAYS, STORY_DAYS = 30, 5, 5, 2
MIN_EVENTS, CHECK_N, CHECK_PASS = 100, 50, 40
RESOLVE_DAYS, RESOLVE_LAST = 365, pd.Timestamp("2025-09-30")
PRIOR_A, PRIOR_B, MIN_RESOLVED, GAP, PREMIUM = 1.0, 2.0, 5, 0.15, 0.30
GROUPS = ["short_seller", "small", "foreign", "big", "unnamed"]   # first match wins, in this order

# Cue words that make an ambiguous outlet name a citation ("according to The Deal", "The Information
# reports"); matched case-insensitively, the outlet name itself case-sensitively.
_CUE_BEFORE = r"(?i:according to|per|via|citing|cites|reported by|report (?:by|from)|says|said|:)\s+"
_CUE_AFTER = r"(?i:\s+(?:says|said|reports?|reported|report|scoop))"


def _plain(names):
    return r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b"


def _cited(names):
    alt = "|".join(re.escape(n) for n in names)
    return rf"(?:{_CUE_BEFORE}(?:the\s+)?(?:{alt})\b|\b(?:{alt}){_CUE_AFTER}\b)"


SOURCES = {
    "short_seller": re.compile("|".join([
        _plain(["Muddy Waters", "Spruce Point", "Culper", "Kerrisdale", "Fuzzy Panda", "Blue Orca", "J Capital",
                "Hindenburg", "Citron", "Viceroy"]),
        r"\b(?:Grizzly|Iceberg|Scorpion|Morpheus|Wolfpack|Gotham City|Bleecker Street)\s+(?:Research|Capital|Partners)\b",
        r"\bHunterbrook\b(?=.*(?i:short))"])),
    "small": re.compile("|".join([
        _plain(["Betaville", "Dealreporter", "Semafor", "Axios", "Business Insider", "New York Post", "NY Post",
                "Sky News", "Sunday Times", "Fox Business", "Capitol Forum", "Debtwire", "Octus", "Reorg",
                "StreetInsider", "Hunterbrook", "Breakingviews", "Barron's", "Barrons", "MarketWatch"]),
        _cited(["The Information", "The Times", "Puck", "The Deal", "The Fly"])])),
    "foreign": re.compile("|".join([
        _plain(["Calcalist", "TheMarker", "Handelsblatt", "Il Sole 24 Ore", "Les Echos", "Korea Economic Daily",
                "Yonhap", "Economic Times", "Caixin", "Jiji"]),
        _cited(["Globes", "Nikkei", "Expansion", "Expansión"])])),
    "big": re.compile(_plain(["Bloomberg", "Reuters", "Wall Street Journal", "WSJ", "Financial Times", "FT",
                              "CNBC", "New York Times", "NYT", "Dow Jones"])),
    "unnamed": re.compile(r"(?i)\bpeople familiar\b|\bsources? say\b|\baccording to sources\b|\bchatter\b|"
                          r"\brumou?rs?\b|\brumou?red\b|\breport says\b"),
}
DEAL = re.compile(r"(?i)\btakeover\b|\bbuyout\b|\bbids?\b|\bapproach(?:ed|es)?\b|\bacqui(?:re|res|red|ring|sition)s?\b|"
                  r"\bmerg(?:er|e|ing)s?\b|\bexplor\w*\s+(?:a\s+)?(?:potential\s+)?sale\b|\bstrategic alternatives\b|"
                  r"\bgo(?:ing)?[- ]private\b|\btake[- ]private\b|\bstake\b|\bactivist\b|\bin talks\b")
SHORT = re.compile(r"(?i)\bshort[- ](?:report|seller|sellers|position)\b|\bfraud\w*\b")
SHORT_SELLER_WORD = re.compile(r"(?i)\bshort\w*\b|\bfraud\w*\b|\breport\b")
GDELT_DOMAINS = {
    "Betaville": "betaville.co.uk", "Sky News": "news.sky.com", "New York Post": "nypost.com",
    "The Information": "theinformation.com", "Semafor": "semafor.com", "Axios": "axios.com",
    "Business Insider": "businessinsider.com", "The Times": "thetimes.co.uk", "Fox Business": "foxbusiness.com",
    "Puck": "puck.news", "The Deal": "thedeal.com", "Capitol Forum": "thecapitolforum.com",
    "StreetInsider": "streetinsider.com", "The Fly": "thefly.com", "Hunterbrook": "hntrbrk.com",
    "Barron's": "barrons.com", "MarketWatch": "marketwatch.com", "Calcalist": "calcalistech.com",
    "Globes": "en.globes.co.il", "TheMarker": "themarker.com", "Nikkei": "asia.nikkei.com",
    "Handelsblatt": "handelsblatt.com", "Il Sole 24 Ore": "ilsole24ore.com", "Les Echos": "lesechos.fr",
    "Expansion": "expansion.com", "Korea Economic Daily": "kedglobal.com", "Yonhap": "en.yna.co.kr",
    "Economic Times": "economictimes.indiatimes.com", "Caixin": "caixinglobal.com", "Jiji": "jiji.com"}
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEAL_FORMS = {"SC TO-T", "SC 14D9", "DEFM14A", "425", "SC 13D", "SCHEDULE 13D"}


# ------------------------------------------------------------------ pure rules (tested offline)

def classify(headline):
    """(group, sign, source words, deal words) of one headline, or None. Group order as registered: short
    seller, small, foreign, big, unnamed. A short-seller name counts only beside a short, fraud or report
    word; otherwise the next groups are tried. Sign is -1 for short-seller headlines and short-report or
    fraud words, +1 for deal and stake words."""
    if not headline:
        return None
    deal, short = DEAL.findall(headline), SHORT.findall(headline)
    for g in GROUPS:
        m = SOURCES[g].search(headline)
        if not m:
            continue
        if g == "short_seller":
            w = SHORT_SELLER_WORD.findall(headline)
            if not w:
                continue
            return g, -1, m.group(0).strip(), ", ".join(sorted({x.lower() for x in w}))
        if not deal and not short:
            return None
        words = ", ".join(sorted({x.lower() for x in deal + short}))
        return g, (-1 if short else +1), m.group(0).strip(), words
    return None


def classified(news):
    """Classified articles of one news frame (id, created_at, headline, symbols) tagged with 1-2 symbols,
    exploded to one row per symbol, in [FIRST, LAST)."""
    n = news[news["symbols"].fillna("") != ""].copy()
    n["syms"] = n["symbols"].str.split(",")
    n = n[n["syms"].map(len) <= 2]
    c = [classify(h) for h in n["headline"]]
    n["c"] = c
    n = n[n["c"].notna()]
    if n.empty:
        return pd.DataFrame(columns=["id", "t", "symbol", "group", "sign", "src_words", "deal_words", "n_symbols"])
    out = pd.DataFrame({"id": n["id"].to_numpy(), "t": D.et_time(n["created_at"]).to_numpy(),
                        "syms": n["syms"].to_numpy(), "n_symbols": n["syms"].map(len).to_numpy(),
                        "group": [x[0] for x in n["c"]], "sign": [x[1] for x in n["c"]],
                        "src_words": [x[2] for x in n["c"]], "deal_words": [x[3] for x in n["c"]]})
    out = out.explode("syms").rename(columns={"syms": "symbol"})
    return out[(out["t"] >= FIRST) & (out["t"] < LAST)].reset_index(drop=True)


def first_per_window(arts, days, cal, keys=("symbol", "group")):
    """Keep an article only if no kept article with the same keys came in the previous `days` trading days
    (by the trading day each article falls on: the next session for articles after the close)."""
    cal = pd.DatetimeIndex(sorted(cal))
    a = arts.sort_values(["t", "id"]).copy()
    a["tday"] = cal.searchsorted((a["t"] - pd.Timedelta(hours=16)).dt.normalize() + pd.Timedelta(days=1))
    keep, last = [], {}
    for i, r in zip(a.index, a[list(keys) + ["tday"]].itertuples(index=False)):
        k = tuple(r[:-1])
        if k in last and r[-1] - last[k] < days:
            keep.append(False)
            continue
        last[k] = r[-1]
        keep.append(True)
    return a[keep].drop(columns="tday").reset_index(drop=True)


def stories(arts, cal):
    """Story id per classified article: the same symbol, the same sign (deal vs short), within STORY_DAYS
    trading days of the story's first article."""
    s = first_per_window(arts.assign(_s=arts["sign"]), STORY_DAYS, cal, keys=("symbol", "_s"))
    s = s[["id", "symbol", "_s", "t", "group"]].rename(columns={"t": "story_t", "group": "first_group"})
    a = arts.assign(_s=arts["sign"]).sort_values("t")
    out = pd.merge_asof(a, s.sort_values("story_t"), left_on="t", right_on="story_t", by=["symbol", "_s"],
                        suffixes=("", "_story"))
    return out.rename(columns={"id_story": "story_id"}).drop(columns="_s")


def entry_time(t, cal):
    """(entry day, earliest bar time) for an article at t (naive ET): t + 30 minutes if that falls inside a
    regular session (before its last minute), else the 10:00 bar of the next session after t."""
    t0 = (t + pd.Timedelta(minutes=WAIT_MIN)).ceil("min")
    for day in cal["date"][cal["date"] >= t.normalize()].iloc[:8]:
        o, c = L1.session_bounds(cal, day)
        if o <= t0 <= c - pd.Timedelta(minutes=1):
            return day, t0
        if t0 < o:
            return day, max(o, pd.Timestamp(f"{day:%Y-%m-%d} 10:00"))
    return None, None


def hold_trade(bars, spy, cal, day, t0, hold=HOLD_DAYS):
    """Entry at the open of the first bar at or after t0 on `day`, exit at the close of the last bar of the
    session `hold` trading days later; SPY over the same minutes."""
    i = cal.index[cal["date"] == day][0]
    if i + hold >= len(cal):
        return None
    _, c = L1.session_bounds(cal, day)
    xo, xc = L1.session_bounds(cal, cal["date"].iloc[i + hold])
    e, se = L1.first_at_or_after(bars, t0, c), None
    if e is None:
        return None
    se = L1.first_at_or_after(spy, e["t"], c)
    x, sx = L1.last_before(bars, xc, xo), L1.last_before(spy, xc, xo)
    if se is None or x is None or sx is None:
        return None
    return {"entry_t": e["t"], "exit_t": x["t"], "exit_day": cal["date"].iloc[i + hold], "entry_px": e["o"],
            "r": x["c"] / e["o"] - 1, "r_spy": sx["c"] / se["o"] - 1}


def resolve(events, filings):
    """Did each event come true within RESOLVE_DAYS (amendment 3)? Adds `resolvable` (the window ends by
    RESOLVE_LAST, the last filing date we may read), `hit` and `resolve_date` (the resolving filing's date,
    or the window's end for a miss: when the outcome became known)."""
    ev = events.copy()
    ev["d"] = ev["t"].dt.normalize()
    ev["w_end"] = ev["d"] + pd.Timedelta(days=RESOLVE_DAYS)
    ev["resolvable"] = ev["w_end"] <= RESOLVE_LAST
    f = filings.copy()
    is8k = f["form"].isin(["8-K", "8-K/A"])
    items = f["items"].map(lambda x: set(x) if x is not None else set())
    f["deal"] = f["form"].isin(DEAL_FORMS) | (is8k & items.map(lambda s: bool(s & {"1.01", "2.01"})))
    f["short"] = f["form"].isin(["NT 10-K"]) | (is8k & items.map(lambda s: bool(s & {"4.01", "4.02"})))
    hits, dates = [], []
    by = {c: g.sort_values("filed") for c, g in f.groupby("cik")}
    for e in ev.itertuples(index=False):
        g = by.get(e.cik)
        col = "short" if e.sign < 0 else "deal"
        if g is not None:
            x = g[(g[col]) & (g["filed"] > e.d) & (g["filed"] <= e.w_end)]
        if g is not None and len(x):
            hits.append(True)
            dates.append(x["filed"].iloc[0])
        else:
            hits.append(False)
            dates.append(e.w_end)
    ev["hit"], ev["resolve_date"] = hits, dates
    return ev.drop(columns=["d", "w_end"])


def beta_post(hits, n, a=PRIOR_A, b=PRIOR_B):
    """Posterior mean and 95% interval of a hit rate with a Beta(a, b) prior."""
    from scipy.stats import beta
    A, B = a + hits, b + n - hits
    return {"n": int(n), "hits": int(hits), "mean": A / (A + B), "ci95": [float(beta.ppf(.025, A, B)),
                                                                        float(beta.ppf(.975, A, B))]}


def pit_rate(ev, t, author, group):
    """Point-in-time hit rate for an event at t: the author's posterior mean from rumours resolved before
    t if it has >= MIN_RESOLVED, else the group's, else the prior mean (amendment 4)."""
    known = ev[(ev["resolvable"]) & (ev["resolve_date"] < t.normalize()) & (ev["sign"] > 0)]
    for sel in ((known["author"] == author) if isinstance(author, str) and author else None, known["group"] == group):
        if sel is None:
            continue
        k = known[sel]
        if len(k) >= MIN_RESOLVED:
            return (PRIOR_A + k["hit"].sum()) / (PRIOR_A + PRIOR_B + len(k)), len(k)
    return PRIOR_A / (PRIOR_A + PRIOR_B), 0


def implied_prob(entry_px, prior_close):
    return float(min(1.0, max(0.0, (entry_px - prior_close) / (PREMIUM * prior_close))))


def gap_side(h, p):
    d = h - p
    return +1 if d > GAP else -1 if d < -GAP else 0


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def group_stats(m, col="net"):
    """Date-clustered NW t (date-level means, 2 lags) and block-bootstrap CI of one group's trades."""
    ok = m.dropna(subset=[col])
    per_day = ok.groupby("entry_day")[col].mean().sort_index()
    if len(per_day) < 3:
        return {"n": int(len(ok)), "dates": int(len(per_day)), "mean": float("nan"), "t_hac": float("nan"),
                "ci95": [float("nan")] * 2}
    mean, se, t = nw_tstat(per_day.to_numpy(), 2)
    ci = block_bootstrap_ci(per_day.to_numpy(), 5, N_BOOT, SEED) if len(per_day) > 5 else (float("nan"),) * 2
    return {"n": int(len(ok)), "dates": int(len(per_day)), "mean": mean, "se_hac": se, "t_hac": t,
            "ci95": list(ci), "hit_rate": float((ok[col] > 0).mean())}


def check_passed(out):
    f = Path(out) / "W1a_check.csv"
    if not f.exists():
        return False, "W1a_check.csv is missing (run the events stage)"
    c = pd.read_csv(f, dtype=str).fillna("")
    filled = c["ok"].str.lower().isin(["y", "n"])
    if not filled.all():
        return False, f"{(~filled).sum()} of {len(c)} hand-check rows are not filled"
    good = int((c["ok"].str.lower() == "y").sum())
    return good >= CHECK_PASS, f"{good} of {len(c)} right (need {CHECK_PASS})"


# ------------------------------------------------------------------ stages

def stage_events(cache, out):
    cal = D.calendar()
    parts = [classified(D.news_month(cache, per)) for per in pd.period_range("2016-01", D.NEWS_LAST, freq="M")]
    arts = pd.concat([p for p in parts if len(p)], ignore_index=True)
    arts.to_parquet(D.w1dir(cache) / "w1a_articles.parquet")
    ev = first_per_window(arts, DEDUP_DAYS, cal["date"])
    ev["prior_day"] = prior_sessions(ev["t"], cal["date"])
    parts = []
    for y in sorted(ev["prior_day"].dt.year.unique()):
        u = D.universe_year(cache, y).rename(columns={"date": "prior_day"})
        parts.append(ev[ev["prior_day"].dt.year == y].merge(u, on=["symbol", "prior_day"], how="inner"))
    ev = pd.concat(parts, ignore_index=True)
    ev["price_group"] = np.where(ev["close"] < L1.PENNY_MAX, "penny", "main")
    ev = ev.sort_values(["t", "id", "symbol"]).reset_index(drop=True)
    ev.to_parquet(D.w1dir(cache) / "w1a_events.parquet")
    out.mkdir(parents=True, exist_ok=True)
    cov = arts.assign(year=arts["t"].dt.year).groupby(["group", "src_words", "year"]).size().rename("articles")
    cov.reset_index().to_csv(out / "W1a_coverage.csv", index=False)
    print(ev.groupby(["group", "price_group"]).size().to_string())


def write_check(cache, out):
    """50 random events for the hand check, with the article URL and only the matched words (no headline
    text). Written once; Claude fills `ok` (y/n) and `note`."""
    f = out / "W1a_check.csv"
    if f.exists():
        return
    ev = pd.read_parquet(D.w1dir(cache) / "w1a_events.parquet")
    rng = np.random.default_rng(SEED)
    k = np.sort(rng.choice(len(ev), size=min(CHECK_N, len(ev)), replace=False))
    chk = ev.iloc[k][["id", "t", "symbol", "group", "sign", "src_words", "deal_words"]]
    chk = chk.merge(load_details(cache)[["id", "url"]], on="id", how="left")
    out.mkdir(parents=True, exist_ok=True)
    chk.assign(ok="", note="").to_csv(f, index=False)


def prior_sessions(ts, days):
    """The last trading day whose session ended before each time t (naive ET; sessions end 16:00)."""
    days = pd.DatetimeIndex(sorted(days))
    out = []
    for t in ts:
        d = t.normalize()
        i = days.searchsorted(d)
        same = i < len(days) and days[i] == d and t.hour * 60 + t.minute >= 16 * 60
        out.append(days[i] if same else days[i - 1] if i > 0 else pd.NaT)
    return pd.DatetimeIndex(out)


def stage_details(cache, out, budget_min):
    """Author, summary and URL of every classified article, by symbol-month (amendment 1)."""
    deadline = time.monotonic() + budget_min * 60
    h = alpaca_headers()
    arts = pd.read_parquet(D.w1dir(cache) / "w1a_articles.parquet")
    arts["month"] = arts["t"].dt.to_period("M").astype(str)
    ddir = D.w1dir(cache) / "details"
    ddir.mkdir(parents=True, exist_ok=True)
    want = set(arts["id"])
    for (sym, mon), g in arts.groupby(["symbol", "month"]):
        f = ddir / f"{sym}_{mon}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            print("[details] time budget reached; rerun to continue")
            return False
        start = pd.Timestamp(f"{mon}-01", tz=ET)
        p = {"symbols": sym, "start": start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": (start + pd.offsets.MonthBegin(1)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "sort": "asc", "limit": 50, "include_content": "false"}
        rows, tok = [], None
        while True:
            if tok:
                p["page_token"] = tok
            j = alpaca.get(alpaca.NEWS_URL, p, headers=h)
            rows += [{"id": a.get("id"), "author": a.get("author"), "summary": a.get("summary"), "url": a.get("url")}
                     for a in (j.get("news") or []) if a.get("id") in want]
            tok = j.get("next_page_token")
            time.sleep(0.3)
            if not tok:
                break
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=["id", "author", "summary", "url"]).to_parquet(tmp)
        tmp.replace(f)
    write_check(cache, out)
    return True


def load_details(cache):
    parts = [pd.read_parquet(p) for p in (D.w1dir(cache) / "details").glob("*.parquet")]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "author", "summary", "url"])
    return d.drop_duplicates("id")


def stage_gdelt(out, get=requests.get):
    """Per-year GDELT article counts with deal words for each small and foreign outlet (amendment 5)."""
    res = {}
    q = "(acquisition OR takeover OR merger OR buyout OR stake)"
    for name, dom in GDELT_DOMAINS.items():
        res[name] = {}
        for y in range(2016, 2026):
            p = {"query": f"domain:{dom} {q}", "mode": "timelinevolraw", "format": "json",
                 "startdatetime": f"{y}0101000000", "enddatetime": f"{y}1231235959" if y < 2025 else "20250930235959"}
            try:
                r = get(GDELT_URL, params=p, timeout=60)
                tl = r.json().get("timeline", []) if r.status_code == 200 else None
            except (requests.RequestException, ValueError):
                tl = None
            if tl is None:
                res[name][y] = "not available"
                continue
            res[name][y] = int(sum(pt.get("value", 0) for s in tl for pt in s.get("data", [])))
            time.sleep(5)
    out.mkdir(parents=True, exist_ok=True)
    (out / "W1a_gdelt.json").write_text(json.dumps(res, indent=1))


def stage_measure(cache, out, budget_min):
    ok, why = check_passed(out)
    if not ok:
        print(f"[measure] hand check not passed: {why}; measure stage refused")
        return False
    deadline = time.monotonic() + budget_min * 60
    h = alpaca_headers()
    cal = D.calendar()
    ev = pd.read_parquet(D.w1dir(cache) / "w1a_events.parquet")
    c0p = Path("research/results/C0/C0_costs.json")
    c0 = json.loads(c0p.read_text()) if c0p.exists() else None
    mdir = D.w1dir(cache) / "w1a_measure"
    mdir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in mdir.glob("*.json")}
    spy, rngs, uni_year, uni = {}, {}, None, None
    for e in ev.to_dict("records"):
        key = f"{e['id']}_{e['symbol']}"
        if key in done:
            continue
        if time.monotonic() > deadline:
            print("[measure] time budget reached; rerun to continue")
            return False
        rec = {k: (str(v) if isinstance(v, pd.Timestamp) else v.item() if hasattr(v, "item") else v) for k, v in e.items()}
        day, t0 = entry_time(e["t"], cal)
        if day is None or cal.index[cal["date"] == day][0] + HOLD_DAYS >= len(cal):
            rec["status"] = "no entry day"
        else:
            i = cal.index[cal["date"] == day][0]
            win = (L1.session_bounds(cal, day)[0], L1.session_bounds(cal, cal["date"].iloc[i + HOLD_DAYS])[1])
            if day not in spy:
                spy[day] = L1.minute_bars("SPY", *win, h)
            tr = hold_trade(L1.minute_bars(e["symbol"], *win, h), spy[day], cal, day, t0)
            rec["entry_day"] = f"{day:%Y-%m-%d}"
            grp = e["price_group"]
            if tr:
                held = (tr["exit_day"] - day).days
                cost = L1.cost_of(grp, e["close"], e["dv20"], c0)
                rec.update(entry_t=str(tr["entry_t"]), exit_t=str(tr["exit_t"]), entry_px=tr["entry_px"], r=tr["r"],
                           r_spy=tr["r_spy"], cost=cost, held=held,
                           net=L1.net(e["sign"], tr["r"], tr["r_spy"], cost, grp, held))
            pday = pd.Timestamp(e["prior_day"])
            if uni_year != pday.year:
                uni_year = pday.year
                uni = {d_: g.sort_values("symbol") for d_, g in D.universe_year(cache, uni_year).groupby("date")}
            pool = uni.get(pday, pd.DataFrame(columns=["symbol", "close", "dv20"]))
            pool = pool[((pool["close"] < L1.PENNY_MAX) == (grp == "penny")) & (pool["symbol"] != e["symbol"])]
            g = rngs.setdefault(day, np.random.default_rng(SEED + day.toordinal()))
            if len(pool):
                cs = pool.iloc[int(g.integers(len(pool)))]
                ct = hold_trade(L1.minute_bars(cs["symbol"], *win, h), spy[day], cal, day, t0)
                rec["control_symbol"] = cs["symbol"]
                if ct:
                    held = (ct["exit_day"] - day).days
                    rec["control_net"] = L1.net(e["sign"], ct["r"], ct["r_spy"], L1.cost_of(grp, cs["close"], cs["dv20"], c0),
                                                grp, held)
            rec["status"] = "ok" if tr else "no bars"
        tmp = mdir / f"{key}.tmp"
        tmp.write_text(json.dumps(rec, default=str))
        tmp.replace(mdir / f"{key}.json")
        done.add(key)
        if len(done) % 200 == 0:
            print(f"[measure] {len(done)}/{len(ev)} events")
    return True


def factor_alpha_events(per_day, cal, hold=HOLD_DAYS):
    """Alpha of the per-entry-day mean return on FF5 + momentum summed over each trade's days (N1's
    method; clarification 11)."""
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fac = ffd.join(umdd, how="inner")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umdd.columns[0]]]
    days = pd.DatetimeIndex(cal["date"])
    wins = [days[days.searchsorted(pd.Timestamp(e)):days.searchsorted(pd.Timestamp(e)) + hold + 1] for e in per_day.index]
    a = stats.ols_hac(per_day.to_numpy(), stats.window_factors(fac, wins), 2)
    a["factors"] = list(fac.columns) + [f"(summed over each trade's {hold + 1} days)"]
    return a, sha5, shau


def stage_report(cache, out):
    cal = D.calendar()
    rows = [json.loads(p.read_text()) for p in sorted((D.w1dir(cache) / "w1a_measure").glob("*.json"))]
    m = pd.DataFrame(rows)
    m["t"] = pd.to_datetime(m["t"])
    det = load_details(cache)
    m = m.merge(det[["id", "author"]], on="id", how="left")
    p2, both = stats.idea_counts()
    res = {"idea": "W1a", "prereg": "research/prereg/W1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "status_counts": m["status"].value_counts().to_dict(), "groups": {}}
    okm = m[m["status"] == "ok"]
    for g in GROUPS:
        x = okm[okm["group"] == g]
        s = group_stats(x)
        if s["dates"] > 2:
            per_day = x.groupby("entry_day")["net"].mean().sort_index()
            a, sha5, shau = factor_alpha_events(per_day, cal)
            s["factor_alpha"] = a
            s["deflated_sharpe"] = stats.deflated_sharpe(per_day.to_numpy(), both)
            s["passes"] = bool(s["n"] >= MIN_EVENTS) and stats.screen_pass(s, a)
            s["reversal_flag"] = bool(s["t_hac"] <= -3.0)
            res["ff5_daily_sha256"], res["umd_daily_sha256"] = sha5, shau
        else:
            s["passes"] = False
        s["control"] = group_stats(x.rename(columns={"net": "_n", "control_net": "net"}))
        s["by_price_group"] = {pg: group_stats(x[x["price_group"] == pg]) for pg in ("main", "penny")}
        res["groups"][g] = s
    # resolution, hit rates and the secondary gap test (amendments 3-4)
    filings = D.load_filings(cache, ["cik", "form", "filed", "items"])
    filings["items"] = filings["items"].map(lambda x: list(x) if x is not None else [])
    allev = resolve(pd.read_parquet(D.w1dir(cache) / "w1a_events.parquet").merge(det[["id", "author"]], on="id", how="left"),
                    filings)
    rv = allev[allev["resolvable"]]
    res["hit_rates"] = {"by_group": {f"{g}/{'short' if s < 0 else 'deal'}": beta_post(int(x["hit"].sum()), len(x))
                                     for (g, s), x in rv.groupby(["group", "sign"])},
                        "by_author": {a: beta_post(int(x["hit"].sum()), len(x))
                                      for a, x in rv[rv["sign"] > 0].groupby("author") if a and len(x) >= MIN_RESOLVED}}
    gap = okm[okm["sign"] > 0].copy()
    hs, ps, sides = [], [], []
    for e in gap.itertuples(index=False):
        hh, _ = pit_rate(allev, e.t, e.author, e.group)
        pp = implied_prob(e.entry_px, e.ac) if e.ac and math.isfinite(e.ac) else float("nan")
        hs.append(hh)
        ps.append(pp)
        sides.append(gap_side(hh, pp) if math.isfinite(pp) else 0)
    gap["h"], gap["p"], gap["side"] = hs, ps, sides
    gt = gap[gap["side"] != 0].copy()
    gt["net_gap"] = [L1.net(sd, r, rs, c, pg, hd) for sd, r, rs, c, pg, hd in
                     zip(gt["side"], gt["r"], gt["r_spy"], gt["cost"], gt["price_group"], gt["held"])]
    sec = group_stats(gt, "net_gap")
    if sec["dates"] > 2:
        a, _, _ = factor_alpha_events(gt.groupby("entry_day")["net_gap"].mean().sort_index(), cal)
        sec["factor_alpha"] = a
        sec["passes"] = bool(sec["n"] >= MIN_EVENTS) and stats.screen_pass(sec, a)
    res["secondary_gap"] = sec
    res["gap_counts"] = {"long": int((gap["side"] > 0).sum()), "short": int((gap["side"] < 0).sum()),
                         "none": int((gap["side"] == 0).sum())}
    # exploratory: stories and first publishers, confirmed vs unconfirmed, the skipped 30 minutes
    arts = stories(pd.read_parquet(D.w1dir(cache) / "w1a_articles.parquet"), cal["date"])
    res["stories"] = {"articles": int(len(arts)), "stories": int(arts["story_id"].nunique()),
                      "first_group_counts": arts.drop_duplicates("story_id")["first_group"].value_counts().to_dict()}
    okr = okm.merge(allev[["id", "symbol", "resolvable", "hit"]], on=["id", "symbol"], how="left")
    res["exploratory"] = {f"{g}/{'confirmed' if h else 'unconfirmed'}": group_stats(x)
                          for (g, h), x in okr[okr["resolvable"] == True].groupby(["group", "hit"])}  # noqa: E712
    res["passes_any"] = [g for g, s in res["groups"].items() if s.get("passes")] + (["secondary_gap"] if sec.get("passes") else [])
    out.mkdir(parents=True, exist_ok=True)
    (out / "W1a_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("status_counts", "passes_any")}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["data", "events", "details", "gdelt", "measure", "report", "all"])
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    st, deadline = a.stage, time.monotonic() + a.budget_min * 60
    if st in ("data", "all") and not D.run(cache, deadline):
        print("[data] unfinished; rerun to continue")
        return
    if st in ("events", "all"):
        stage_events(cache, out)
    if st in ("details", "all") and not stage_details(cache, out, (deadline - time.monotonic()) / 60):
        return
    if st in ("gdelt", "all"):
        stage_gdelt(out)
    if st in ("measure", "all") and not stage_measure(cache, out, (deadline - time.monotonic()) / 60):
        return
    if st in ("report", "all"):
        stage_report(cache, out)


if __name__ == "__main__":
    main()
