"""P1 and P2 paper trading, one run a night (research/prereg/P1.md, P2.md, with their clarifications).

Each trading day d is processed once, after 23:30 ET: Benzinga articles up to d 23:30 ET, SEC's daily
form index of d, and d's daily bars. The event rules are phase 1's frozen code (tracks/penny/frozen/), and
the event step below follows breakout-scanner's catalyst/nightly.py process_day (commit 111b71d), cut to
the penny universe and with NT 10-K / NT 10-Q filings added (P1's NT class).

Per day the job writes one hash-chained ledger file (kind "paper_day"): the universe funnel, the new
events, the P1/P2 signals and the skipped ones with reasons, the controls, and both paper books after the
day (the real one and the control one). The 30 calendar days before START are processed for events only
(warm-up), so clusters and the controls' 20-day exclusion windows start full.

When the last complete day has been processed and the next open is still ahead, the penny book's orders
for that open are also sent to the Alpaca *paper* account (market-on-open for entries and stop exits,
market-on-close for time exits due that day), and the fills of earlier orders are read back, so each
trade gets its implementation shortfall (kind "paper_mirror"). The paper book, not the mirror, is P1's
and P2's record.

Usage: python -m tracks.penny.nightly --state <dir> [--no-mirror]
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, borrow, leads_sec, sec, stats
from core.ledger import Ledger
from paper import book as B
from tracks.penny.frozen import rules as R

ET = "America/New_York"
START = pd.Timestamp("2026-10-19")      # first signal day (entries from the open of 2026-10-20; P1 clarification 12)
WARMUP = pd.Timedelta(days=30)
RUN_AFTER = pd.Timedelta(hours=23, minutes=30)
LOOKBACK_FIRST = pd.Timedelta(days=3)
HIST_DAYS = 2                            # earlier days of articles kept for first-article times and C1 signs
BARS_BACK = pd.Timedelta(days=200)
NT_FORMS = {"NT 10-K", "NT 10-Q"}
P1_CLASSES = {"A8", "A6", "A9", "NT"}
P1_HOLD, P2_HOLD = 20, 5
P1_SEED, P2_SEED = 20261015, 20261016
CONTROL_WINDOW = 20                      # trading days, the signal day included
DILUTION_DAYS, A9_DAYS, PUMP = 90, 180, 0.50
MIN_PRICE, MAX_PRICE, MIN_DV20, MIN_BARS = 1.0, 5.0, 5e5, 60
GAP_DAYS = 15
LIMIT_OFFSET = 0.01                      # X1: limit 1% through the signal close (buy above, sell below)
EVENT_COLS = ["event_id", "night_day", "cls", "sign", "symbol", "series_id", "cik", "T", "source", "src", "id",
              "accession", "headline"]


# ------------------------------------------------------------------ pure rules (tested offline)

def annotate(raw):
    """Raw daily bars (symbol, date, o, h, l, c, v) -> plus seg (series segment: a >15-calendar-day gap
    starts a new one), nbar (bars so far in the segment) and dv20 (median raw dollar volume of the last
    20 bars, within the segment)."""
    r = raw[raw["v"] > 0].sort_values(["symbol", "date"]).reset_index(drop=True)
    gap = r.groupby("symbol")["date"].diff().dt.days.gt(GAP_DAYS)
    r["seg"] = gap.astype(int).groupby(r["symbol"]).cumsum()
    g = r.groupby(["symbol", "seg"], sort=False)
    r["nbar"] = g.cumcount() + 1
    r["dv20"] = (r["c"] * r["v"]).groupby([r["symbol"], r["seg"]]).transform(lambda x: x.rolling(20).median())
    return r


def atr_ratio(adj, d):
    """symbol -> ATR(20) / close on day d, from adjusted bars (so splits inside the 20 days don't distort
    it), within the current segment."""
    a = annotate(adj[adj["date"] <= d])
    prev = a.groupby(["symbol", "seg"])["c"].shift(1)
    tr = np.maximum(a["h"], prev.fillna(a["h"])) - np.minimum(a["l"], prev.fillna(a["l"]))
    a["atr"] = tr.groupby([a["symbol"], a["seg"]]).transform(lambda x: x.rolling(20).mean())
    x = a[a["date"] == d]
    return dict(zip(x["symbol"], x["atr"] / x["c"]))


def universe_on(ann, d, ciks):
    """P1's penny universe on day d: a bar on d, raw close $1-5, dv20 >= $500k, >= 60 bars in the current
    series. ciks: symbol -> CIK (today's SEC map). series_id is the symbol (see the P1 clarifications)."""
    x = ann[(ann["date"] == d) & (ann["c"] >= MIN_PRICE) & (ann["c"] < MAX_PRICE) & (ann["dv20"] >= MIN_DV20)
            & (ann["nbar"] >= MIN_BARS)]
    u = pd.DataFrame({"symbol": x["symbol"].values, "series_id": x["symbol"].values, "close": x["c"].values,
                      "dv20": x["dv20"].values})
    u["cik"] = u["symbol"].map(ciks)
    return u.reset_index(drop=True)


def etb_of(asset):
    """Easy-to-borrow flag as phase 1 recorded it: easy_to_borrow, else borrow_status, else None."""
    a = asset or {}
    e = a.get("easy_to_borrow")
    if e is None and a.get("borrow_status") is not None:
        e = a.get("borrow_status") == "easy_to_borrow"
    return e


def split_ratio(raw, adj, symbol, d):
    """New shares per old share for a split effective on d (from the raw/adjusted close ratio jump), or
    None. Cash dividends move the ratio by far less than the 20% threshold."""
    m = raw[raw["symbol"] == symbol].merge(adj[adj["symbol"] == symbol], on=["symbol", "date"],
                                           suffixes=("_r", "_a"))
    m = m[m["date"] <= d].sort_values("date").tail(2)
    if len(m) < 2 or m["date"].iloc[-1] != d:
        return None
    q = m["c_r"] / m["c_a"]
    k = q.iloc[-1] / q.iloc[-2]
    return float(1 / k) if (k >= 1.2 or k <= 1 / 1.2) else None


def a9_dates(raw, adj, symbol, lo, hi):
    """Days in [lo, hi] on which the frozen reverse-split bar rule fires for `symbol`."""
    r, a = raw[raw["symbol"] == symbol], adj[adj["symbol"] == symbol]
    days = r.loc[(r["date"] >= lo) & (r["date"] <= hi), "date"]
    return [d for d in days if len(R.a9_items(r, a, d))]


def is_a8_filing(form, items):
    return form in R.OFFER_FORMS or (form == "8-K" and "3.02" in (items or []))


def recent_a8(subs, lo, hi):
    """A8 filings (offering forms, or 8-K item 3.02) of one company filed in [lo, hi], from SEC's
    submissions JSON (filings.recent)."""
    rec = (subs or {}).get("filings", {}).get("recent", {})
    out = []
    for form, filed, items in zip(rec.get("form", []), rec.get("filingDate", []), rec.get("items", [])):
        f = pd.Timestamp(filed)
        if lo <= f <= hi and is_a8_filing(form, [i.strip() for i in (items or "").split(",")]):
            out.append({"form": form, "filed": filed})
    return out


def p2_filter(ev, d, raw, adj, store, subs, tdays):
    """P2's three skip rules, in order; returns the reason or None."""
    lo = d - pd.Timedelta(days=DILUTION_DAYS)
    st = store[store["series_id"] == ev["series_id"]] if len(store) else store
    night = pd.to_datetime(st["night_day"]) if len(st) else pd.Series(dtype="datetime64[ns]")
    if recent_a8(subs, lo, d) or (len(st) and ((st["cls"] == "A8") & (night >= lo) & (night <= d)).any()):
        return "dilution: A8 in the prior 90 days"
    a = adj[(adj["symbol"] == ev["symbol"]) & (adj["date"] <= d)].sort_values("date")
    i = tdays.index(d)
    if len(a) >= 6 and a["date"].iloc[-1] == d and a["date"].iloc[-6] == tdays[i - 5] \
            and a["c"].iloc[-1] / a["c"].iloc[-6] - 1 > PUMP:
        return "pump: 5-day return above +50%"
    lo9 = d - pd.Timedelta(days=A9_DAYS)
    if a9_dates(raw, adj, ev["symbol"], lo9, d) or \
            (len(st) and ((st["cls"] == "A9") & (night >= lo9) & (night <= d)).any()):
        return "reverse split: A9 in the prior 180 days"
    return None


def control_pool(u, store, classes, d, tdays, exclude, need_etb):
    """Universe names with no event of `classes` in the CONTROL_WINDOW trading days ending on d, not in
    `exclude`, and (for shorts) shortable and easy to borrow. Sorted by symbol."""
    i = tdays.index(d)
    lo = tdays[max(0, i - CONTROL_WINDOW + 1)]
    bad = set()
    if len(store):
        n = pd.to_datetime(store["night_day"])
        bad = set(store.loc[store["cls"].isin(classes) & (n >= lo) & (n <= d), "series_id"])
    p = u[~u["series_id"].isin(bad) & ~u["symbol"].isin(exclude) & np.isfinite(u["atr_ratio"].astype(float))]
    if need_etb:
        p = p[p["shortable"].fillna(False).astype(bool) & p["etb"].fillna(False).astype(bool)]
    return p.sort_values("symbol").reset_index(drop=True)


def control_cfg(cfg):
    """The control book: the same trade rules with no position, drawdown or account limits (a control
    exists to give its trade a same-day benchmark, never to be refused)."""
    c = copy.deepcopy(cfg)
    c["account"]["pause_drawdown"] = 1e9
    for k in ("max_gross", "max_name", "max_open_risk"):
        c["account"].pop(k, None)
    for t in c["tracks"].values():
        t.update(max_positions=10 ** 6, max_per_industry=10 ** 6, pause_drawdown=1e9)
    return c


# ------------------------------------------------------------------ data access (network)

class Api:
    """Every network call the job makes; the tests replace it with a fake."""

    def calendar(self, start, end):
        return alpaca.trading_calendar(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")

    def assets(self):
        return alpaca.active_assets()

    def tickers(self):
        from tracks.main.longterm import sec_data
        return sec_data.current_ticker_map()

    def news(self, start_et, end_et):
        """Articles created in [start, end] (naive ET), as in phase 1's catalyst/sources.py news()."""
        rows, tok = [], None
        s = pd.Timestamp(start_et, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        e = pd.Timestamp(end_et, tz=ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        while True:
            j = alpaca.news(s, e, page_token=tok)
            for n in j.get("news", []):
                rows.append({k: (",".join(n.get(k) or []) if k == "symbols" else n.get(k))
                             for k in ("id", "created_at", "headline", "summary", "symbols", "url")})
            tok = j.get("next_page_token")
            if not tok:
                break
        d = pd.DataFrame(rows, columns=["id", "created_at", "headline", "summary", "symbols", "url"]).drop_duplicates("id")
        d["created_at"] = pd.to_datetime(d["created_at"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
        return d

    def bars(self, symbols, start, end, adjustment):
        from core.bars import fetch_daily_bars, to_frames
        from core.env import alpaca_headers
        fr = to_frames(fetch_daily_bars(sorted(set(symbols)), f"{start:%Y-%m-%d}", alpaca.sip_end(end),
                                        adjustment=adjustment, headers=alpaca_headers()))
        parts = [f[["o", "h", "l", "c", "v"]].assign(symbol=s).rename_axis("date").reset_index() for s, f in fr.items()]
        cols = ["symbol", "date", "o", "h", "l", "c", "v"]
        return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)

    def form_index(self, d):
        return sec.daily_form_index(d)

    def header(self, cik, file):
        return leads_sec.header(cik, file)

    def submissions(self, cik):
        r = sec.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        return r.json() if r is not None else None

    # Alpaca paper account (orders never go anywhere else)
    def positions(self):
        return {p["symbol"]: p for p in alpaca.get(f"{alpaca.TRADE}/positions", {})}

    def order(self, body):
        return alpaca.post(f"{alpaca.TRADE}/orders", body)

    def order_by_client_id(self, cid):
        try:
            return alpaca.get(f"{alpaca.TRADE}/orders:by_client_order_id", {"client_order_id": cid}, tries=3)
        except Exception as e:  # noqa: BLE001 - an unknown order is recorded, not fatal
            return {"error": str(e)[:200]}


# ------------------------------------------------------------------ one day's events

def find_events(api, d, u, names, store, arts_all, prev_cutoff, raw, adj):
    """New events of day d over universe u, or None when SEC's index of d is not published yet."""
    cutoff = d + RUN_AFTER
    start = prev_cutoff if prev_cutoff is not None else cutoff - pd.Timedelta(days=1) - LOOKBACK_FIRST
    arts = arts_all[(arts_all["created_at"] > start) & (arts_all["created_at"] <= cutoff)]
    hist = arts_all[(arts_all["created_at"] > start - pd.Timedelta(days=HIST_DAYS)) & (arts_all["created_at"] <= cutoff)]
    rows_all = R.article_rows(hist, names)
    rows = rows_all[rows_all["id"].isin(set(arts["id"]))]
    text_by_id = dict(zip(rows_all["id"], rows_all["text"]))
    tx = R.text_items(rows)

    idx = api.form_index(d)
    if idx is None:
        return None
    ucik = {int(c): (s, sym, "PENNY") for c, s, sym in zip(u["cik"], u["series_id"], u["symbol"]) if pd.notna(c)}
    f = idx[idx["form"].isin({"8-K"} | R.OFFER_FORMS | NT_FORMS) & idx["cik"].isin(set(ucik))].drop_duplicates("file")
    heads = []
    for cik, form, file in zip(f["cik"], f["form"], f["file"]):
        accn, acc, items = api.header(cik, file)
        heads.append({"found": acc is not None, "accepted_et": pd.Timestamp(acc) if acc else None, "ciks": [int(cik)],
                      "type": form, "items": items, "accession": accn})
    fi = R.filing_items([h for h in heads if h["type"] not in NT_FORMS], ucik)
    nt = pd.DataFrame([{"cls": "NT", "symbol": ucik[h["ciks"][0]][1], "t_item": h["accepted_et"], "src": h["type"],
                        "accession": h["accession"], "penny_only": False}
                       for h in heads if h["type"] in NT_FORMS and h["found"]],
                      columns=["cls", "symbol", "t_item", "src", "accession", "penny_only"])
    fit = []
    for r in pd.concat([x for x in (fi, nt) if len(x)] or [fi], ignore_index=True).itertuples(index=False):
        hit = R.first_article_time(r.symbol, r.t_item, rows_all)
        T, aid, src = (hit[0], hit[1], "benzinga") if hit else (r.t_item, None, "sec")
        fit.append({"cls": r.cls, "symbol": r.symbol, "T": T, "source": src, "id": aid, "src": r.src,
                    "accession": r.accession, "main_only": False, "penny_only": r.penny_only})
    fit = pd.DataFrame(fit)

    syms = set(u["symbol"])
    lo = d - pd.Timedelta(days=12)
    a9 = R.a9_items(raw[raw["symbol"].isin(syms) & (raw["date"] >= lo) & (raw["date"] <= d)],
                    adj[adj["symbol"].isin(syms) & (adj["date"] >= lo) & (adj["date"] <= d)], d)

    funnel = {"universe": int(len(u)), "universe_with_cik": int(u["cik"].notna().sum()), "articles": int(len(arts)),
              "article_symbol_rows": int(len(rows)), "text_items": int(len(tx)),
              "sec": f"index ok: {len(f)} filings, {sum(h['found'] for h in heads)} headers",
              "filing_items": int(len(fit)), "a9_items": int(len(a9))}
    parts = [x for x in (tx, fit, a9) if len(x)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLS), funnel
    items = pd.concat(parts, ignore_index=True)
    items["T"] = pd.to_datetime(items["T"])
    m = items.merge(u[["symbol", "series_id", "cik"]], on="symbol", how="inner")
    funnel["items_not_in_universe"] = int(len(items) - len(m))
    m = m[~m["main_only"].fillna(False).astype(bool)]      # the A8 headline rule is MAIN-only (frozen)
    base = store[["series_id", "cls", "T"]] if len(store) else pd.DataFrame(columns=["series_id", "cls", "T"])
    # first night: articles of the lookback are context only (they seed clusters, never become events)
    win_start = prev_cutoff if prev_cutoff is not None else cutoff - pd.Timedelta(days=1)
    is_ctx = (m["src"] == "text") & (m["T"] < win_start)
    if is_ctx.any():
        ctx, _ = R.cluster_new(m[is_ctx], base)
        base = pd.concat([base, ctx[["series_id", "cls", "T"]]], ignore_index=True).sort_values("T")
    funnel["lookback_context_items"] = int(is_ctx.sum())
    new, joined = R.cluster_new(m[~is_ctx], base)
    funnel["joined_existing_cluster"] = int(len(joined))
    if not len(new):
        return pd.DataFrame(columns=EVENT_COLS), funnel
    new["sign"] = ["-" if c == "NT" else s for c, s in zip(new["cls"], R.signs(new, text_by_id, rows_all))]
    hl = dict(zip(rows_all["id"], rows_all["headline"]))
    new["headline"] = [None if os.environ.get("NO_HEADLINE_TEXT") else hl.get(i) for i in new["id"]]
    new["night_day"] = f"{d:%Y-%m-%d}"
    new["event_id"] = [f"{d:%Y%m%d}-{i:04d}" for i in range(len(new))]
    for c in EVENT_COLS:
        if c not in new:
            new[c] = None
    funnel["events"] = int(len(new))
    funnel["by_class"] = new.groupby(["cls", "sign"]).size().rename("n").reset_index().to_dict("records")
    return new[EVENT_COLS].reset_index(drop=True), funnel


# ------------------------------------------------------------------ one day's signals and books

def make_signals(api, d, events, u, store, raw, adj, tdays):
    """P1 and P2 signals of day d from its new events, the P2 skips, and the controls' pools."""
    sig, skipped = [], []
    by = u.set_index("symbol")
    ev = events.sort_values(["T", "event_id"])
    seen = set()
    for e in ev.to_dict("records"):
        idea = "P1" if e["cls"] in P1_CLASSES and e["sign"] == "-" else \
            "P2" if e["cls"] == "C4" and e["sign"] == "+" else None
        if idea is None:
            continue
        if (idea, e["symbol"]) in seen:
            skipped.append({"event_id": e["event_id"], "idea": idea, "symbol": e["symbol"], "reason": "same stock, same night"})
            continue
        seen.add((idea, e["symbol"]))
        if idea == "P2":
            subs = api.submissions(e["cik"]) if pd.notna(e["cik"]) else None
            reason = p2_filter(e, d, raw, adj, store, subs, tdays)
            if reason:
                skipped.append({"event_id": e["event_id"], "idea": idea, "symbol": e["symbol"], "reason": reason})
                continue
        r = by.loc[e["symbol"]]
        if not (isinstance(r["atr_ratio"], float) and math.isfinite(r["atr_ratio"])):
            skipped.append({"event_id": e["event_id"], "idea": idea, "symbol": e["symbol"], "reason": "no ATR(20)"})
            continue
        sig.append({"track": "penny", "idea": idea, "symbol": e["symbol"], "side": "short" if idea == "P1" else "long",
                    "hold_days": P1_HOLD if idea == "P1" else P2_HOLD, "atr20": float(r["atr_ratio"] * r["close"]),
                    "close": float(r["close"]), "shortable": bool(r["shortable"]) if pd.notna(r["shortable"]) else None,
                    "easy_to_borrow": bool(r["etb"]) if pd.notna(r["etb"]) else None, "event_id": e["event_id"],
                    "cls": e["cls"]})
    return sig, skipped


def draw_controls(d, accepted, u, store, tdays, control_state):
    """One control per accepted signal, drawn per idea from that night's pool with the registered seed,
    without replacement within the night and skipping names the control book already holds."""
    out = []
    held = {x["symbol"] for x in control_state["open"] + control_state["pending_entries"]}
    for idea, seed, classes, etb in (("P1", P1_SEED, P1_CLASSES, True), ("P2", P2_SEED, {"C4", "A8"}, False)):
        mine = [s for s in accepted if s["idea"] == idea]
        if not mine:
            continue
        pool = control_pool(u, store, classes, d, tdays, held, etb)
        rng = np.random.default_rng(seed + d.toordinal())
        picks = rng.choice(len(pool), size=min(len(mine), len(pool)), replace=False) if len(pool) else []
        for s, k in zip(mine, picks):
            r = pool.iloc[int(k)]
            out.append(dict(s, idea=idea + "c", symbol=r["symbol"], atr20=float(r["atr_ratio"] * r["close"]),
                            close=float(r["close"]), shortable=True if etb else s.get("shortable"),
                            easy_to_borrow=True if etb else s.get("easy_to_borrow"), pair_id=s["event_id"],
                            event_id=s["event_id"] + "c", cls="control"))
            held.add(r["symbol"])
        for s in mine[len(picks):]:
            out.append({"pair_id": s["event_id"], "idea": idea + "c", "symbol": None, "reason": "empty control pool"})
    return out


# ------------------------------------------------------------------ the Alpaca paper mirror

def mirror_orders(state, tomorrow):
    """Orders for tomorrow's session from the penny book after tonight: (client_order_id, body, kind)."""
    out = []
    for e in state["pending_entries"]:
        if e["track"] == "penny" and e["entry_date"] == tomorrow:
            qty = int(e["notional"] // e["close"])
            body = {"symbol": e["symbol"], "qty": str(qty), "side": "buy" if e["side"] == "long" else "sell",
                    "type": "market", "time_in_force": "opg", "client_order_id": f"el-{e['event_id']}-in"}
            out.append((body["client_order_id"], body if qty >= 1 else None, "entry"))
    for p in state["open"]:
        if p["track"] != "penny":
            continue
        side = "sell" if p["side"] == "long" else "buy"
        if p.get("pending_exit") and p["pending_exit"]["date"] == tomorrow:
            cid, tif, kind = f"el-{p['event_id']}-stop", "opg", "stop exit"
        elif p["days_held"] == p["hold_days"] - 1:
            cid, tif, kind = f"el-{p['event_id']}-time", "cls", "time exit"
        else:
            continue
        out.append((cid, {"symbol": p["symbol"], "qty": None, "side": side, "type": "market", "time_in_force": tif,
                          "client_order_id": cid}, kind))
    return out


def run_mirror(api, state, tomorrow, pending_ids):
    """Send tomorrow's orders and read back the fills of earlier ones. Returns the ledger record."""
    assert alpaca.TRADE.startswith("https://paper-api.alpaca.markets/"), "orders go to the paper account only"
    fills = []
    for cid in pending_ids:
        o = api.order_by_client_id(cid)
        fills.append({"client_order_id": cid, "status": o.get("status"), "filled_qty": o.get("filled_qty"),
                      "filled_avg_price": o.get("filled_avg_price"), "filled_at": o.get("filled_at"),
                      "error": o.get("error")})
    sent = []
    pos = None
    for cid, body, kind in mirror_orders(state, tomorrow):
        if body is None:
            sent.append({"client_order_id": cid, "kind": kind, "status": "not sent: under one share"})
            continue
        if body["qty"] is None:
            pos = api.positions() if pos is None else pos
            q = abs(float((pos.get(body["symbol"]) or {}).get("qty", 0)))
            if q < 1:
                sent.append({"client_order_id": cid, "kind": kind, "status": "not sent: no Alpaca position"})
                continue
            body = dict(body, qty=str(int(q)))
        code, js = api.order(body)
        sent.append({"client_order_id": cid, "kind": kind, "body": body, "http": code,
                     "status": js.get("status") if code < 300 else "rejected", "message": js.get("message")})
    return {"tomorrow": tomorrow, "sent": sent, "fills": fills}


def shortfall(fills, book_states):
    """Implementation shortfall of each filled entry vs the book's decision price (the signal close) and
    its simulated fill (the open bar)."""
    trades = {}
    for s in book_states:
        for x in s["open"] + s["closed"]:
            trades[x["event_id"]] = x
    out = []
    for f in fills:
        eid = f["client_order_id"][3:].rsplit("-", 1)[0]
        t = trades.get(eid)
        if not t or not f.get("filled_avg_price") or not f["client_order_id"].endswith("-in"):
            continue
        side = 1 if t["side"] == "long" else -1
        px = float(f["filled_avg_price"])
        out.append({"event_id": eid, "fill": px, "is_vs_close": stats.implementation_shortfall(t["close"], px, side),
                    "is_vs_book_open": stats.implementation_shortfall(t["entry_price"], px, side)})
    return out


# ------------------------------------------------------------------ the nightly run

def _days_logged(led):
    return sorted(e["date"] for e in led.entries() if e["kind"] == "paper_day")


def _read(led, kind, date):
    e = led.entry(kind, date)
    return json.loads((led.root / e["file"]).read_text()) if e else None


def _store(led):
    rows = []
    for e in led.entries():
        if e["kind"] == "paper_day":
            rows += json.loads((led.root / e["file"]).read_text())["events"]
    s = pd.DataFrame(rows, columns=EVENT_COLS)
    s["T"] = pd.to_datetime(s["T"])
    return s


def run(state_dir, now=None, api=None, start=START, mirror=True):
    api = api or Api()
    now = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(ET).tz_localize(None)
    if now < start + RUN_AFTER:
        print(f"paper: nothing before the first signal night ({start:%Y-%m-%d} 23:30 ET)")
        return 0
    led = Ledger(Path(state_dir) / "paper")
    cfg = B.load_config()
    ccfg = control_cfg(cfg)
    cal = api.calendar(start - WARMUP - BARS_BACK, now + pd.Timedelta(days=15))
    cal["date"] = pd.to_datetime(cal["date"])
    tdays = list(cal["date"])
    complete = [d for d in tdays if d + RUN_AFTER <= now]
    logged = _days_logged(led)
    first = pd.Timestamp(logged[-1]) + pd.Timedelta(days=1) if logged else start - WARMUP
    todo = [d for d in complete if d >= first]
    if not todo:
        print("paper: up to date")
    tds = [f"{d:%Y-%m-%d}" for d in tdays]

    last = _read(led, "paper_day", logged[-1]) if logged else None
    state = last["book"] if last and last.get("book") else B.new_state(cfg)
    cstate = last["control_book"] if last and last.get("control_book") else B.new_state(ccfg)
    lstate = last["limit_book"] if last and last.get("limit_book") else B.new_state(cfg)
    if todo:
        assets = api.assets()
        common = assets[assets["excl_reason"].isna()]
        flags = {r["symbol"]: r for r in common.to_dict("records")}
        tk = api.tickers()
        ciks = dict(zip(tk["symbol"], tk["cik"]))
        names = R.symbol_names(dict(zip(common["symbol"], common["name"])), dict(zip(tk["symbol"], tk["title"])))
        held = {x["symbol"] for s in (state, cstate, lstate) for x in s["open"] + s["pending_entries"]}
        raw = api.bars(set(common["symbol"]) | held, todo[0] - BARS_BACK, todo[-1], "raw")
        ann = annotate(raw)
        unis = {d: universe_on(ann, d, ciks) for d in todo}
        adj_syms = set().union(*[set(u["symbol"]) for u in unis.values()]) | held
        adj = api.bars(adj_syms, todo[0] - BARS_BACK, todo[-1], "all")
        first_cut = (pd.Timestamp(logged[-1]) + RUN_AFTER) if logged else None
        a0 = (first_cut or todo[0] + RUN_AFTER - pd.Timedelta(days=1) - LOOKBACK_FIRST) - pd.Timedelta(days=HIST_DAYS)
        arts_all = api.news(a0, todo[-1] + RUN_AFTER)
        store = _store(led)
        fees, fee_day = borrow.latest(state_dir)
        fee_by = fees.set_index("symbol")[["fee", "available"]].to_dict("index") if fees is not None else {}
        prev_cutoff = first_cut
        for d in todo:
            u = unis[d]
            u["atr_ratio"] = u["symbol"].map(atr_ratio(adj[adj["symbol"].isin(set(u["symbol"]))], d))
            u["shortable"] = u["symbol"].map(lambda s: (flags.get(s) or {}).get("shortable"))
            u["etb"] = u["symbol"].map(lambda s: etb_of(flags.get(s)))
            res = find_events(api, d, u, names, store, arts_all, prev_cutoff, raw, adj)
            if res is None:
                print(f"paper: {d:%Y-%m-%d} SEC index not published yet; stopping")
                break
            events, funnel = res
            ds = f"{d:%Y-%m-%d}"
            rec = {"day": ds, "warmup": bool(d < start), "funnel": funnel,
                   "events": json.loads(events.to_json(orient="records", date_format="iso")),
                   "assets_fetched_utc": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ}"}
            if d >= start:
                splits = []
                for st in (state, cstate, lstate):
                    for sym in sorted({p["symbol"] for p in st["open"]}):
                        k = split_ratio(raw, adj, sym, d)
                        if k:
                            splits.append({"symbol": sym, "ratio": k})
                if splits:
                    for x in splits:
                        state = B.apply_split(state, x["symbol"], x["ratio"])
                        cstate = B.apply_split(cstate, x["symbol"], x["ratio"])
                        lstate = B.apply_split(lstate, x["symbol"], x["ratio"])
                bars = {(r.symbol, ds): {"o": float(r.o), "h": float(r.h), "l": float(r.l), "c": float(r.c)}
                        for r in raw[raw["date"] == d].itertuples(index=False)}
                sig, skipped = make_signals(api, d, events, u, store, raw, adj, tdays)
                for x in sig:          # P1 clarification 11: the night's IBKR fee, logged, not used by the book
                    f = fee_by.get(x["symbol"]) or {}
                    x.update({k: (None if pd.isna(f.get(c)) else float(f[c])) for k, c in
                              (("borrow_fee", "fee"), ("borrow_available", "available"))}, borrow_day=fee_day)
                state = B.process_day(state, cfg, ds, bars, sig, tds)
                accepted = [x for x in state["pending_entries"] if x["signal_date"] == ds]
                ctrl = draw_controls(d, accepted, u, store, tdays, cstate)
                cstate = B.process_day(cstate, ccfg, ds, bars, [c for c in ctrl if c.get("symbol")], tds)
                lsig = [dict(x, limit=round(x["close"] * (1 + LIMIT_OFFSET if x["side"] == "long" else 1 - LIMIT_OFFSET), 4))
                        for x in sig]
                lstate = B.process_day(lstate, cfg, ds, bars, lsig, tds)
                rec.update(signals=sig, skipped=skipped, controls=ctrl, splits=splits, book=state, control_book=cstate, limit_book=lstate,
                           summary=B.summary(state, cfg))
            led.write_json("paper_day", ds, f"days/{ds}.json", rec)
            print(f"paper: {ds}: {funnel.get('events', 0)} events, {len(rec.get('signals', []))} signals"
                  + (" (warm-up)" if d < start else ""))
            ev = events.copy()
            store = pd.concat([store, ev], ignore_index=True) if len(ev) else store
            prev_cutoff = d + RUN_AFTER

    logged = _days_logged(led)
    if not (mirror and logged and logged[-1] >= f"{start:%Y-%m-%d}" and complete and logged[-1] == f"{complete[-1]:%Y-%m-%d}"):
        return len(todo)
    if led.has("paper_mirror", logged[-1]):
        return len(todo)
    i = tds.index(logged[-1])
    tomorrow = tds[i + 1]
    opens = pd.Timestamp(f"{tomorrow} {cal['open'].iloc[i + 1]}")
    rec = {"day": logged[-1]}
    if now >= opens - pd.Timedelta(minutes=5):
        rec.update(status="not sent: the run came after the next open", tomorrow=tomorrow)
    else:
        pending = []
        for e in led.entries():
            if e["kind"] == "paper_mirror":
                m = json.loads((led.root / e["file"]).read_text())
                done = {f["client_order_id"] for f in m.get("fills", []) if f.get("status") in ("filled", "canceled",
                                                                                                  "expired", "rejected")}
                pending = [c for c in pending if c not in done]
                pending += [s["client_order_id"] for s in m.get("sent", []) if s.get("http", 999) < 300]
        rec.update(run_mirror(api, state, tomorrow, pending))
        rec["shortfall"] = shortfall(rec["fills"], [state])
    led.write_json("paper_mirror", logged[-1], f"mirror/{logged[-1]}.json", rec)
    print(f"paper: mirror {rec.get('status') or str(len(rec['sent'])) + ' orders'}")
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    run(a.state, mirror=not a.no_mirror and os.environ.get("PAPER_MIRROR", "on") != "off")


if __name__ == "__main__":
    main()
