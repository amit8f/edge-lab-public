"""L1 forward shadow books (research/prereg/FWD1.md, PROTOCOL rules 19-20): the two near-miss L1 cells,
offering/penny and late filing/main, traded forward exactly as research/prereg/L1.md and
tracks/news/l1_study.py do: short from the first regular-session minute opening at least 1 minute after
EDGAR accepts the filing, cover at the next session's close, against SPY, with one random control stock
from the same price group over the same minutes. No orders: each event is logged to the hash-chained
ledger in <state>/l1fwd once its exit has happened.

A filing day f is processed when the session two trading days after f has closed (a filing accepted after
hours enters the next session and exits the one after). Shortability is Alpaca's easy-to-borrow flag when
the event is logged (one or two sessions after the signal; the screen had no such flag).
The 5-most-confident variant is computed at evaluation: per entry day, the 5 events with the largest move
against the trade from the previous close to the entry (r_open, signed).

Usage: python -m tracks.news.l1_live --state <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, leads_sec, sec
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.ledger import Ledger
from tracks.main.longterm import bars_batch, sec_data
from tracks.news import l1_study as S

ET = "America/New_York"
FIRST_DAY = pd.Timestamp("2026-10-26")       # first filing day after FWD1 was signed
CELLS = {("offering", "penny"), ("late_filing", "main")}
FORMS = S.OFFER | {"NT 10-K", "NT 10-Q"}
LOOKBACK = 45                                 # calendar days of daily bars for dv20


def group_rows(frames, day):
    """Universe rows (symbol, close, dv20, group) on `day` from raw daily frames, by S.universe_rows."""
    out = []
    for sym, d in frames.items():
        u = S.universe_rows(sym, d)
        u = u[u["date"] == day]
        if len(u):
            out.append(u)
    if not out:
        return pd.DataFrame(columns=["symbol", "date", "close", "dv20", "group"])
    u = pd.concat(out, ignore_index=True)
    u["group"] = np.where(u["close"] < S.PENNY_MAX, "penny", "main")
    return u


def candidates(idx, tmap):
    """The filing day's rows of the two cells' forms, with today's ticker for the CIK (one row per CIK and
    category, the first accession)."""
    f = idx[idx["form"].isin(FORMS)].copy()
    f["cik"] = f["cik"].astype("int64")
    f["category"] = [S.category(x, [], None, set()) for x in f["form"]]
    f = f.merge(tmap[["symbol", "cik"]], on="cik", how="inner")
    return f.sort_values("file").drop_duplicates(["cik", "category"]).reset_index(drop=True)


def r_open(bars, prev_day, cal, entry_o):
    """Entry open over the previous session's last regular-session close in the same bars, minus 1."""
    po, pc = S.session_bounds(cal, prev_day)
    p = S.last_before(bars, pc, po)
    return float(entry_o / p["c"] - 1) if p is not None and p["c"] > 0 else None


def event(row, acc, cal, uni_prev, bars_of, spy_of, shortable, c0=None):
    """One logged event, or a record with a status explaining why it has no trade."""
    rec = {"accession": row["accession"], "form": row["form"], "category": row["category"], "symbol": row["symbol"],
           "cik": int(row["cik"]), "filed": f"{row['filed']:%Y-%m-%d}", "accepted_et": None if acc is None else str(acc)}
    if acc is None:
        return rec | {"status": "no header"}
    day, t0 = S.entry_point(pd.Timestamp(acc), cal)
    if day is None:
        return rec | {"status": "no entry day"}
    me = uni_prev[uni_prev["symbol"] == row["symbol"]]
    if me.empty:
        return rec | {"status": "not in universe", "entry_day": f"{day:%Y-%m-%d}"}
    grp = me["group"].iloc[0]
    rec.update(entry_day=f"{day:%Y-%m-%d}", group=grp, close=float(me["close"].iloc[0]), dv20=float(me["dv20"].iloc[0]))
    if (row["category"], grp) not in CELLS:
        return rec | {"status": "other cell"}
    i = int(cal.index[cal["date"] == day][0])
    prev_day = cal["date"].iloc[i - 1]
    win = (S.session_bounds(cal, prev_day)[0], S.session_bounds(cal, cal["date"].iloc[i + 1])[1])
    spy = spy_of(day, win)
    b = bars_of(row["symbol"], win)
    tr = S.trade(b, spy, cal, day, t0)
    if tr is None:
        return rec | {"status": "no bars"}
    sign, held = S.SIGN[row["category"]], (tr["exit_day"] - day).days
    cost = S.cost_of(grp, rec["close"], rec["dv20"], c0)
    e = S.first_at_or_after(b, t0, S.session_bounds(cal, day)[1])
    rec.update(status="ok", entry_t=str(tr["entry_t"]), exit_t=str(tr["exit_t"]), r=tr["r"], r_spy=tr["r_spy"],
               cost=cost, net=S.net(sign, tr["r"], tr["r_spy"], cost, grp, held),
               r_open=r_open(b, prev_day, cal, e["o"]), shortable=shortable.get(row["symbol"]))
    pool = uni_prev[(uni_prev["group"] == grp) & (uni_prev["symbol"] != row["symbol"])].sort_values("symbol")
    g = np.random.default_rng(S.SEED + day.toordinal() + int(row["cik"]) % 100_000)
    if len(pool):
        cs = pool.iloc[int(g.integers(len(pool)))]
        ct = S.trade(bars_of(cs["symbol"], win), spy, cal, day, t0)
        rec["control_symbol"] = cs["symbol"]
        if ct:
            ccost = S.cost_of(grp, float(cs["close"]), float(cs["dv20"]), c0)
            rec["control_net"] = S.net(sign, ct["r"], ct["r_spy"], ccost, grp, (ct["exit_day"] - day).days)
    return rec


def ready_days(cal, now_et, cursor):
    """Filing days after the cursor whose second following session has closed by now."""
    days = list(cal["date"])
    out = []
    for f in pd.bdate_range(max(cursor + pd.Timedelta(days=1), FIRST_DAY), now_et.normalize().tz_localize(None)):
        later = [d for d in days if d > f]
        if len(later) < 2:
            break
        if now_et.tz_localize(None) < S.session_bounds(cal, later[1])[1] + pd.Timedelta(minutes=30):
            break
        out.append(f)
    return out


def run(state, now=None):
    now = now if now is not None else alpaca.utcnow()
    root = Path(state) / "l1fwd"
    led = Ledger(root)
    cur_f = root / "cursor.json"
    cursor = pd.Timestamp(json.loads(cur_f.read_text())["last_day"]) if cur_f.exists() else FIRST_DAY - pd.Timedelta(days=1)
    now_et = pd.Timestamp(now).tz_convert(ET)
    cal = alpaca.trading_calendar((FIRST_DAY - pd.Timedelta(days=60)).date(), (now_et + pd.Timedelta(days=10)).date())
    cal = cal.reset_index(drop=True)
    days = ready_days(cal, now_et, cursor)
    if not days:
        print("l1fwd: no filing day ready")
        return 0
    h = alpaca_headers()
    tmap = sec_data.current_ticker_map()
    assets = bars_batch.assets_all()
    common = assets[(assets["excl_reason"].isna()) & (assets["status"] == "active")]
    ok = common["shortable"].fillna(False).astype(bool) & common["easy_to_borrow"].fillna(False).astype(bool)
    shortable = dict(zip(common["base_symbol"], ok))
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    start = (days[0] - pd.Timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")
    frames = to_frames(fetch_daily_bars(syms, start, alpaca.sip_end(now_et.normalize(), now), adjustment="raw", headers=h))
    frames = {s: d[["c", "v"]] for s, d in frames.items()}
    c0p = Path("research/results/C0/C0_costs.json")
    c0 = json.loads(c0p.read_text()) if c0p.exists() else None
    spy_cache, uni_cache = {}, {}

    def bars_of(sym, win):
        return S.minute_bars(sym, *win, h)

    def spy_of(day, win):
        if day not in spy_cache:
            spy_cache[day] = S.minute_bars("SPY", *win, h)
        return spy_cache[day]

    n = 0
    for f in days:
        idx = sec.daily_form_index(f)
        if idx is None:
            print(f"l1fwd: {f:%Y-%m-%d} index not published yet; stopping")
            break
        recs = []
        for row in candidates(idx, tmap).to_dict("records"):
            accn, acc, _ = leads_sec.header(row["cik"], row["file"])
            row["accession"] = accn
            day, _ = S.entry_point(pd.Timestamp(acc), cal) if acc else (None, None)
            prev = cal["date"].iloc[cal.index[cal["date"] == day][0] - 1] if day is not None else None
            if prev is not None and prev not in uni_cache:
                uni_cache[prev] = group_rows(frames, prev)
            recs.append(event(row, acc, cal, uni_cache.get(prev, group_rows({}, prev)), bars_of, spy_of, shortable, c0))
        if not led.has("L1_fwd", f"{f:%Y-%m-%d}"):
            led.write_json("L1_fwd", f"{f:%Y-%m-%d}", f"{f:%Y-%m-%d}.json",
                           {"idea": "L1", "book": "FWD1 near-miss shadow book", "filed": f"{f:%Y-%m-%d}",
                            "cells": sorted("/".join(c) for c in CELLS), "events": recs})
        cur_f.write_text(json.dumps({"last_day": f"{f:%Y-%m-%d}"}) + "\n")
        ok = sum(r.get("status") == "ok" for r in recs)
        print(f"l1fwd: {f:%Y-%m-%d}: {len(recs)} candidate filings, {ok} traded in the two cells")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    run(a.state)


if __name__ == "__main__":
    main()
