"""PICK2-BAD forward book (research/prereg/PICK2.md, Amendment 2): PICK-MAIN's frozen COMBINED model buying only
on bad-market rows. A $5,000 ledger book: nothing is ever sent to a broker.

Each run (after the close) logs every finished session from START not yet logged, oldest first, through the
same code path as PICK's history test (tracks/main/pick_test.score_day and picks_of):
1. The universe at the previous close: active common-stock-like Alpaca assets, raw close >= $5 and 20-day median
   dollar volume >= $5M (pick_data.universe, main track), from daily SIP bars (raw).
2. The day's 5-minute SIP bars (raw), NX1's headline rows of Benzinga articles from the prior session's 16:00
   ET to 14:05 ET (FinBERT at NX1's pinned commit, no text kept) and SPY's daily parts.
3. Every decision row 09:35-14:00 uses only bars that finished before it and headlines before it. Rows on
   bad-market minutes (market score below PICK's frozen cut) whose COMBINED score clears PICK's frozen bad-day
   bar are picks: first clearing time per stock, sized 10-40% of the book by score (PICK's rule, the top
   from PICK_test.json), each with PICK's random same-minute control. Exits: +8%, the tier's stop or the
   15:50 bar's close, PICK's costs.
The usual 30-minute swing needs 10 earlier sessions; the first run warms it on the WARM sessions before START.

    python -m tracks.main.pick2_bad_live --state state
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.ledger import Ledger
from tracks.main import pick_data as P
from tracks.main import pick_study as S
from tracks.main import pick_test as T
from tracks.main.longterm import news_history
from tracks.main.longterm.stats import nw_tstat
from tracks.news import nx1_study as N
from tracks.news import w1_data as W

ET = "America/New_York"
START = "2026-11-16"
JUDGED = "2027-12-31"
WARM = 20                                    # sessions before START that warm the swing (P.SWING_N)
DAILY_DAYS = 45                              # calendar days of daily bars behind each session (dv20)
FIT = Path("research/results/PICK/PICK_fit.json")
TEST = Path("research/results/PICK/PICK_test.json")
NX1_META = Path("research/results/NX1/nx1_v1_meta.json")
BOOK = T.BOOK
VERSION, TRACK = "COMBINED", "main"
MIN_DAYS, MIN_TRADES, LAGS = 60, 30, 5


# ---------- pure helpers (tested) ----------

def universe_at(rows, prev):
    """{symbol: (track, prev close, dv20)} of the main track for the session after `prev`, from daily rows."""
    d = rows[rows["date"] == prev]
    u = P.universe(d[["symbol", "c", "dv20"]].dropna())
    return {s: v for s, v in u.items() if v[0] == TRACK}


def bad_only(R):
    """History test rows (pick_test.score_day) -> the same rows with only bad-market picks (controls kept)."""
    return R[(R["role"] == "ctrl") | R["bad"].astype(bool)].reset_index(drop=True)


def day_record(day, pk, n_rows, n_bad):
    """One session's picks (pick_test.picks_of on bad rows) -> the logged record."""
    tk = pk[pk["size"] > 0]
    trades = [{"symbol": r.symbol, "t": pd.Timestamp(r.t).strftime("%H:%M"), "score": float(r.score),
               "size": float(r.size), "usd": float(r.size * BOOK), "net": float(r.net), "exit": r.exit,
               "ctrl_net": None if not np.isfinite(r.ctrl_net) else float(r.ctrl_net), "tier": int(r.tier)}
              for r in tk.itertuples(index=False)]
    book = float(sum(x["size"] * x["net"] for x in trades))
    ctrl = float(sum(x["size"] * x["ctrl_net"] for x in trades if x["ctrl_net"] is not None))
    return {"day": str(pd.Timestamp(day).date()), "rows": int(n_rows), "bad_rows": int(n_bad),
            "picks_before_book": len(pk), "trades": trades, "book_ret": book, "ctrl_ret": ctrl,
            "book_usd": book * BOOK}


def summary(days):
    """The dashboard's view of the logged days (oldest first)."""
    tr = [t for d in days for t in d["trades"]]
    book = np.array([d["book_ret"] for d in days], float)
    diff = np.array([np.mean([t["net"] - t["ctrl_net"] for t in d["trades"] if t["ctrl_net"] is not None])
                     for d in days if any(t["ctrl_net"] is not None for t in d["trades"])], float)
    eq = BOOK * np.cumprod(1 + book) if len(book) else np.array([])
    out = {"first_day": days[0]["day"] if days else None, "last_day": days[-1]["day"] if days else None,
           "days": len(days), "days_with_trades": int(sum(1 for d in days if d["trades"])), "trades": len(tr),
           "book_usd": float(eq[-1]) if len(eq) else BOOK, "mean_net": float(np.mean([t["net"] for t in tr])) if tr else None,
           "mean_ctrl": float(np.nanmean([t["ctrl_net"] if t["ctrl_net"] is not None else np.nan for t in tr])) if tr else None,
           "judged": JUDGED, "min_days": MIN_DAYS, "min_trades": MIN_TRADES,
           "curve": [[d["day"], round(float(e), 2)] for d, e in zip(days, eq)]}
    if len(book) >= 3:
        out["book_t"] = float(nw_tstat(book, LAGS)[2])
    if len(diff) >= 3:
        out["minus_ctrl_t"] = float(nw_tstat(diff, LAGS)[2])
    return out


def update_swings(swings, bars):
    for sym, b in bars.items():
        s = P.swing30(b)
        if np.isfinite(s):
            swings[sym] = (swings.get(sym, []) + [s])[-P.SWING_N:]
    return swings


# ---------- sources ----------

class Api:
    def __init__(self):
        self.h = alpaca_headers()

    def calendar(self, start, end):
        return alpaca.trading_calendar(start, end)

    def symbols(self):
        a = alpaca.active_assets()
        return sorted(set(a.loc[a["excl_reason"].isna(), "base_symbol"]))

    def daily(self, symbols, start, end, adjustment):
        out = {}
        for i in range(0, len(symbols), 500):
            out |= to_frames(fetch_daily_bars(symbols[i:i + 500], start, alpaca.sip_end(end), adjustment=adjustment,
                                              headers=self.h))
        return out

    def bars(self, symbols, day):
        return P.bars_5min(symbols, day, self.h)

    def news(self, start, end):
        return news_history.fetch_window(start, end)

    def finbert(self):
        return N.finbert_scorer(NX1_META)


def daily_rows(api, symbols, first, last):
    raw = api.daily(symbols + ["SPY"], first, last, "raw")
    adj = api.daily(["SPY"], first, last, "all")
    rows = [W.daily_rows(s, raw[s], raw[s]) for s in symbols if s in raw]
    rows = pd.concat([r for r in rows if len(r)], ignore_index=True) if rows else pd.DataFrame(columns=W.COLS)
    spy = adj["SPY"][["o", "c"]].rename(columns={"o": "ao", "c": "ac"}).join(raw["SPY"][["c"]])
    return rows, spy.rename_axis("date").reset_index()


def tone_of(api, finbert, prev, day):
    s = pd.Timestamp(f"{prev.date()} 16:00", tz=ET).tz_convert("UTC")
    e = pd.Timestamp(f"{day.date()} 14:05", tz=ET).tz_convert("UTC")
    return N.headline_rows(api.news(s, e), finbert)


# ---------- job ----------

def finished(cal, now):
    """Sessions whose 5-minute bars are complete (close + 16 minutes passed)."""
    c = pd.to_datetime(cal["date"].dt.strftime("%Y-%m-%d") + " " + cal["close"]).dt.tz_localize(ET)
    return pd.DatetimeIndex(cal["date"][(c + alpaca.SIP_DELAY) <= now].to_numpy())


def run(state, api=None, now=None, finbert=None):
    api = api or Api()
    now = now or pd.Timestamp.now(tz=ET)
    root = Path(state) / "pick2bad"
    led = Ledger(root / "ledger")
    logged = sorted(e["date"] for e in led.entries() if e["kind"] == "day")
    start = pd.Timestamp(START)
    cal = api.calendar((start - pd.Timedelta(days=60)).date(), now.date())
    dates = pd.DatetimeIndex(sorted(cal["date"]))
    done = finished(cal, now)
    todo = [d for d in done if d >= start and (not logged or d > pd.Timestamp(logged[-1]))]
    if not todo:
        print("[pick2-bad] no finished session to log", flush=True)
    else:
        fit = json.loads(FIT.read_text())
        F = T.Frozen(fit, TRACK)
        top = float(json.loads(TEST.read_text())["tops"][TRACK][VERSION])
        sf = root / "swings.json"
        swings = json.loads(sf.read_text()) if sf.exists() else {}
        syms = api.symbols()
        warm = [d for d in dates[dates < start][-WARM:]] if not sf.exists() else []
        first = (min(warm + todo) - pd.Timedelta(days=DAILY_DAYS)).date()
        rows, spy = daily_rows(api, syms, first, max(todo).date())
        sess = np.unique(S.to_ns(spy["date"]))
        parts = S.spy_parts(spy)
        for day in warm:
            prev = dates[dates.get_loc(day) - 1]
            update_swings(swings, api.bars(sorted(universe_at(rows, prev)) + ["SPY"], day))
        if warm:
            print(f"[pick2-bad] swing warmed on {len(warm)} sessions, {len(swings)} symbols", flush=True)
        finbert = finbert or api.finbert()
        for day in todo:
            prev = dates[dates.get_loc(day) - 1]
            uni = universe_at(rows, prev)
            bars = api.bars(sorted(uni) + ["SPY"], day)
            D = T.day_inputs(day, uni, bars, swings)
            if len(D):
                R = T.score_day(D, tone_of(api, finbert, prev, day), sess, parts, {TRACK: F}, bars)
                R["d"] = pd.to_datetime(R["d"])
                pk = T.picks_of(bad_only(R), F.v[VERSION], VERSION, TRACK, top)
                J = S.join_rows(D, pd.DataFrame(columns=S.NEWS_TONE), sess, parts)
                n_bad = int((np.asarray(S.market_score(J, F.mkt)) < F.cut).sum())
            else:
                pk, n_bad = pd.DataFrame(columns=["size"]), 0
            rec = day_record(day, pk, len(D), n_bad) | {"universe": len(uni)}
            led.write_json("day", rec["day"], f"days/{rec['day']}.json", rec)
            update_swings(swings, bars)
            tmp = sf.with_suffix(".tmp")
            tmp.write_text(json.dumps(swings))
            tmp.replace(sf)
            print(f"[pick2-bad] {rec['day']}: {len(D)} rows, {n_bad} bad, {len(rec['trades'])} trades, "
                  f"book {100 * rec['book_ret']:+.2f}%", flush=True)
    probs = led.verify()
    if probs:
        raise RuntimeError(f"ledger problems: {probs[:3]}")
    days = [json.loads(f.read_text()) for f in sorted((led.root / "days").glob("*.json"))]
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "summary.tmp"
    tmp.write_text(json.dumps(summary(days), indent=1))
    tmp.replace(root / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
