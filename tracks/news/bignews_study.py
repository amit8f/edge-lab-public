"""BIGNEWS (research/prereg/BIGNEWS.md, Amendment 1): headline events and their 1-minute entries, 2019-01 .. 2025-09.

Reads NX1's cache: the trading calendar (cal.parquet), the Benzinga headline months (g1/news) and W1's daily bar
batches (w1/daily). Stages, one file per month, atomic writes, resumable:
  events   <out>/events/ev_YYYY-MM.parquet: one row per (headline, symbol) that passes the point-in-time universe
           filter (prior-session raw close >= $1, 20-session median dollar volume >= $1M), with session, tier,
           event type and novelty
  minutes  <out>/minutes/m_YYYY-MM.parquet: raw SIP 1-minute bars 4:00-20:00 of each event day, reduced to entry
           prices (trigger+1, trigger+5, 9:30 open, pre-market), the move to 15:55, the jump before entry, the
           first 5 minutes' volume, and the next closes (all-adjusted, with the day's factor to restate them)

Usage: python -m tracks.news.bignews_study --stage events|minutes --cache cache/nx1 --out cache/bignews
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.env import alpaca_headers
from tracks.news import nx1_study as N
from tracks.news import w1_data as W

ET = "America/New_York"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
MONTHS = pd.period_range("2019-01", "2025-09", freq="M")
MIN_PRICE, MIN_DV20, MAIN_PRICE = 1.0, 1_000_000.0, 5.0
NOVEL_DAYS = 3
SYM_CHUNK = 200
WAIT = pd.Timedelta(minutes=1)
MAX_WAIT = pd.Timedelta(minutes=10)
T_PRE, T_OPEN, T_CLOSE, T_END = (pd.Timedelta(hours=4), pd.Timedelta(hours=9, minutes=30),
                                 pd.Timedelta(hours=16), pd.Timedelta(hours=20))
T_LAST = pd.Timedelta(hours=15, minutes=55)
ENTRIES = ("m1", "m5", "op", "pre")       # trigger+1, trigger+5, 9:30 open, pre-market
EV_COLS = ["id", "symbol", "t", "session_date", "when", "tier", "prev_c", "dv20", "event_type", "novel", "headline"]
MEAS = [f"{p}_{k}" for k in ENTRIES for p in ("px", "t", "mfe", "mae", "jump")] + ["p1555", "v5", "c_next", "c_5",
                                                                                   "adj_f"]
MIN_COLS = ["id", "symbol", "t", "session_date", "when", "tier", "prev_c", "event_type", "novel"] + MEAS
EMPTY_BARS = pd.DataFrame({"t": pd.Series(dtype="datetime64[ns]"), **{c: pd.Series(dtype=float) for c in "ohlcv"}})

# Ordered: the first pattern that matches gives the type.
EVENT_RX = {
    "bankruptcy": r"bankruptcy|chapter (7|11)\b|insolven|going concern",
    "delisting": r"delist|(nasdaq|nyse)\b.*\b(notice|deficiency)|minimum bid|compliance notice|trading halt",
    "fraud_short": r"fraud|short[- ]seller|short report|subpoena|\bsec (probe|investigation|inquiry)|\bindict|"
                   r"accounting irregular|\brestat(e|ed|ement)\b|class action",
    "fda": r"\bfda\b|\bpdufa\b|complete response letter|breakthrough therapy",
    "trial": r"\btrial\b|\bphase\s*(?:[123]|i{1,3}|1/2|2/3)\b|\breadout\b|topline|\bpivotal\b",
    "merger": r"merger|acqui(re|res|red|sition)|buyout|take[- ]private|to be bought|tender offer",
    "guidance": r"guidance|outlook|forecast|preliminary (results|revenue|earnings)|"
                r"(raises|cuts|lowers|reaffirms) .*(year|annual|quarter)",
    "offering": r"offering|registered direct|private placement|at[- ]the[- ]market|\bdilut|"
                r"convertible (notes?|debentures?)|shelf registration",
    "contract": r"\bcontracts?\b|\bawarded\b|purchase orders?|supply agreement|licen[cs]e (deal|agreement)",
}

def event_type(headline):
    h = N.norm(headline)
    for name, rx in EVENT_RX.items():
        if re.search(rx, h, re.I):
            return name
    return "other"


def session_of(t, dates):
    """(session_date, when) of naive-ET trigger times t (a Series) given the sorted trading dates. A trading day
    before 16:00 is its own session: pre (4:00-9:30), regular (9:30-16:00), overnight (before 4:00). Otherwise
    (16:00 on, or a non-trading day) the next trading day after it: after (16:00-20:00 on a trading day) or
    overnight."""
    d = t.dt.normalize()
    tod = t - d
    trade = d.isin(dates).to_numpy()
    tod_np = tod.to_numpy()
    same = trade & (tod < T_CLOSE).to_numpy()
    pos = dates.searchsorted(d.to_numpy(), side="right")
    nxt = dates.to_numpy()[np.minimum(pos, len(dates) - 1)]
    valid = pos < len(dates)
    sess = np.where(same, d.to_numpy(), nxt)
    sess = np.where(same | valid, sess, np.datetime64("NaT"))
    pre = same & (tod_np >= T_PRE.to_timedelta64()) & (tod_np < T_OPEN.to_timedelta64())
    reg = same & (tod_np >= T_OPEN.to_timedelta64())
    aft = trade & (tod_np >= T_CLOSE.to_timedelta64()) & (tod_np < T_END.to_timedelta64())
    when = np.select([pre, reg, aft], ["pre", "regular", "after"], default="overnight")
    return pd.Series(sess, index=t.index), pd.Series(when, index=t.index)


def novelty(rows):
    """rows: id, symbol, t, event_type. True where no earlier row for the same symbol and event type lies within
    the prior NOVEL_DAYS calendar days (the latest earlier one decides, since the groups are time-sorted)."""
    # "other" is a catch-all, so there only the same normalized headline (a repost) counts as not new
    key = rows["event_type"].where(rows["event_type"] != "other",
                                   "other:" + rows.get("headline", pd.Series("", index=rows.index)).fillna("").map(N.norm))
    r = rows.assign(_k=key).sort_values(["symbol", "_k", "t", "id"])
    prev_t = r.groupby(["symbol", "_k"])["t"].shift()
    return (prev_t.isna() | (r["t"] - prev_t > pd.Timedelta(days=NOVEL_DAYS))).reindex(rows.index)


def entry_at(b, target):
    """Close of the first bar starting at or after target, if it starts within MAX_WAIT of it: (price, bar time)."""
    a = b[b["t"] >= target]
    if a.empty or a["t"].iloc[0] - target > MAX_WAIT:
        return np.nan, pd.NaT
    return float(a["c"].iloc[0]), a["t"].iloc[0]


def moves(b, et, px, end):
    """Highest high and lowest low (as moves from px) of the bars after the entry bar up to 15:55 of the session."""
    if pd.isna(px) or pd.isna(et):
        return np.nan, np.nan
    w = b[(b["t"] > et) & (b["t"] <= end)]
    if w.empty:
        return np.nan, np.nan
    return float(w["h"].max() / px - 1), float(w["l"].min() / px - 1)


def minute_row(b, t, sess, prev_c, when):
    """Entries and moves of one headline from its bars b (naive ET, covering the trigger day and the session)."""
    end, open_t = sess + T_LAST, sess + T_OPEN
    targets = {"m1": t + WAIT, "m5": t + pd.Timedelta(minutes=5),
               "op": open_t if when in ("pre", "after", "overnight") else None,
               "pre": t + WAIT if when == "pre" else None}
    out = {}
    for k in ENTRIES:
        px, et = entry_at(b, targets[k]) if targets[k] is not None else (np.nan, pd.NaT)
        if k == "pre" and pd.notna(et) and et >= open_t:          # pre-market only before the open
            px, et = np.nan, pd.NaT
        out[f"px_{k}"], out[f"t_{k}"] = px, et
        out[f"mfe_{k}"], out[f"mae_{k}"] = moves(b, et, px, end)
        out[f"jump_{k}"] = px / prev_c - 1
    last = b[b["t"] <= end]
    out["p1555"] = float(last["c"].iloc[-1]) if len(last) else np.nan
    if len(b):
        s = t.ceil("min")
        out["v5"] = float(b[(b["t"] >= s) & (b["t"] < s + pd.Timedelta(minutes=5))]["v"].sum())
    else:
        out["v5"] = np.nan
    return out


def calendar(cache, out):
    """Trading dates from a cached calendar (cache/cal.parquet or NX1's nx1dir), else Alpaca's (saved to out)."""
    for f in (Path(cache) / "cal.parquet", N.nx1dir(cache) / "cal.parquet", Path(out) / "cal.parquet"):
        if f.exists():
            cal = pd.read_parquet(f)
            break
    else:
        cal = alpaca.trading_calendar("2018-11-01", "2025-10-31")
        N.atomic_parquet(cal, Path(out) / "cal.parquet")
    return pd.DatetimeIndex(sorted(pd.to_datetime(cal["date"]).unique()))


def news_rows(cache, per):
    """One month's headlines exploded to (id, symbol, t, headline, event_type), one row per symbol."""
    n = W.news_month(cache, per)
    n = n[n["symbols"].fillna("") != ""]
    n = n.assign(t=W.et_time(n["created_at"]), event_type=n["headline"].map(event_type))
    x = n.assign(symbol=n["symbols"].str.split(",")).explode("symbol")
    x = x.assign(symbol=x["symbol"].str.strip())
    x = x[x["symbol"] != ""].drop_duplicates(["id", "symbol"])
    return x[["id", "symbol", "t", "headline", "event_type"]].reset_index(drop=True)


def month_events(cache, per, dates):
    cur = news_rows(cache, per).assign(own=True)
    prev_per = per - 1
    if (W.news_dir(cache) / f"news_{prev_per}.parquet").exists():
        prev = news_rows(cache, prev_per).assign(own=False)
        allr = pd.concat([prev, cur], ignore_index=True)
    else:
        allr = cur
    allr = allr.assign(novel=novelty(allr))
    cur = allr[allr["own"]].copy()
    if cur.empty:
        return pd.DataFrame(columns=EV_COLS)
    cur["session_date"], cur["when"] = session_of(cur["t"], dates)
    cur = cur[cur["session_date"].notna()]
    prev_of = pd.Series(dates[:-1], index=dates[1:])
    cur["P"] = cur["session_date"].map(prev_of)
    daily = W.read_rows(cache, ["symbol", "date", "c", "dv20"],
                        [("date", ">=", cur["P"].min()), ("date", "<=", cur["P"].max())])
    daily = daily.rename(columns={"date": "P", "c": "prev_c"})
    cur = cur.merge(daily, on=["symbol", "P"], how="inner")
    cur = cur[(cur["prev_c"] >= MIN_PRICE) & (cur["dv20"] >= MIN_DV20)]
    cur["tier"] = np.where(cur["prev_c"] >= MAIN_PRICE, "main", "penny")
    return cur[EV_COLS].reset_index(drop=True)


def read_daily(cache, lo, hi):
    return W.read_rows(cache, ["symbol", "date", "c", "ac"], [("date", ">=", lo), ("date", "<=", hi)])


class Api:
    """Raw SIP 1-minute bars, extended hours. Tests pass a fake with the same bars_1min method."""

    def __init__(self, headers):
        self.headers = headers

    def bars_1min(self, symbols, start, end):
        """{symbol: DataFrame(t naive ET, o, h, l, c, v)} for naive-ET start..end, capped at now - 16 minutes."""
        s = pd.Timestamp(start, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        e = min(pd.Timestamp(end, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), alpaca.sip_end(end))
        out = {}
        for i in range(0, len(symbols), SYM_CHUNK):
            tok = None
            while True:
                p = {"symbols": ",".join(symbols[i:i + SYM_CHUNK]), "timeframe": "1Min", "start": s, "end": e,
                     "feed": "sip", "adjustment": "raw", "limit": 10000}
                if tok:
                    p["page_token"] = tok
                j = alpaca.get(BARS_URL, p, headers=self.headers)
                for sym, bs in (j.get("bars") or {}).items():
                    out.setdefault(sym, []).extend((b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]) for b in bs)
                tok = j.get("next_page_token")
                if not tok:
                    break
        res = {}
        for sym, rows in out.items():
            d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
            d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
            res[sym] = d.sort_values("t").reset_index(drop=True)
        return res


def month_minutes(ev, cache, dates, api):
    if ev.empty:
        return pd.DataFrame(columns=MIN_COLS)
    dt, s = ev["t"].dt.normalize(), ev["session_date"]
    got = {}
    for x in sorted(set(dt) | set(s)):
        syms = sorted(set(ev.loc[(dt == x) | (s == x), "symbol"]))
        got[x] = api.bars_1min(syms, x + pd.Timedelta(hours=4), x + pd.Timedelta(hours=20))
    rows = []
    for r in ev.itertuples(index=False):
        keys = sorted({r.t.normalize(), r.session_date})
        frames = [got[k][r.symbol] for k in keys if r.symbol in got[k]]
        b = pd.concat(frames, ignore_index=True).sort_values("t") if frames else EMPTY_BARS
        rows.append(minute_row(b, r.t, r.session_date, r.prev_c, r.when))
    base = ev[["id", "symbol", "t", "session_date", "when", "tier", "prev_c", "event_type", "novel"]]
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(rows, columns=MEAS)], axis=1)
    # Daily closes from the symbol's own rows on the exact session dates: a missing row (halt, delisting, gap) is NaN.
    pos = dates.get_indexer(pd.DatetimeIndex(out["session_date"]))
    lo, hi = out["session_date"].min(), dates[min(pos.max() + 5, len(dates) - 1)]
    dl = read_daily(cache, lo, hi).drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])

    def at(col, k):
        d = dates[np.minimum(pos + k, len(dates) - 1)]
        return dl[col].reindex(pd.MultiIndex.from_arrays([out["symbol"].to_numpy(), d])).to_numpy()

    out["c_next"], out["c_5"] = at("ac", 1), at("ac", 5)
    out["adj_f"] = at("ac", 0) / at("c", 0)
    return out


def stage_events(cache, out, deadline, months=MONTHS):
    out = Path(out)
    dates = calendar(cache, out)
    for per in months:
        f = out / "events" / f"ev_{per}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        ev = month_events(cache, per, dates)
        N.atomic_parquet(ev.reset_index(drop=True), f)
        print(f"[bignews events] {per}: {len(ev)} events, {int(ev['novel'].sum())} novel", flush=True)
    return True


def stage_minutes(cache, out, deadline, months=MONTHS, api=None):
    out = Path(out)
    dates = calendar(cache, out)
    for per in months:
        f = out / "minutes" / f"m_{per}.parquet"
        if f.exists():
            continue
        ev_f = out / "events" / f"ev_{per}.parquet"
        if not ev_f.exists():
            raise FileNotFoundError(f"{ev_f}: run --stage events first")
        if time.monotonic() > deadline:
            return False
        if api is None:
            api = Api(alpaca_headers())
        m = month_minutes(pd.read_parquet(ev_f), cache, dates, api)
        N.atomic_parquet(m.reset_index(drop=True), f)
        print(f"[bignews minutes] {per}: {len(m)} events, {int(m['px_m1'].notna().sum())} with a 1-minute entry",
              flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "minutes"])
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="cache/bignews")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    deadline = time.monotonic() + a.budget_min * 60
    stage = stage_events if a.stage == "events" else stage_minutes
    if stage(a.cache, a.out, deadline):
        print(f"[bignews {a.stage}] done")
    else:
        print(f"[bignews {a.stage}] time budget used; rerun to continue")


if __name__ == "__main__":
    main()
