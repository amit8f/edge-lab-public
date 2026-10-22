"""P4 live (SYS1 strategy, research/prereg/P4.md signal and exits): insider cluster buys in penny stocks,
traded in the EL-P4 Alpaca paper account.

Each complete day F (after 22:00 ET, when every Form 4 dated F is in) is processed once, after the leads
job has logged F's form index (core/leads_sec.py):
1. The P4 penny universe on F: raw close $1-5, median 20-day raw dollar volume >= $1M, 60 days of
   volatility, NYSE / Nasdaq / NYSE American, one listing per company; volatility decile and MAX rank.
2. Issuers in it with a Form 4 dated F: their purchases (code P, acquired, officer / director / 10% owner)
   filed in the 10 days ending F are read from the Form 4 XML, and the routine-buyer test uses DERA's
   quarterly purchases of the three prior calendar years (tracks/main/longterm/sec_data.py).
3. P4's cluster rule and exclusions (p4_screen.clusters, qualifies, exclusion_reason, filing_flags).
4. When F is the last complete session and the next open is ahead, each new signal gets a bracket buy for
   the next session: day limit at F's close + 1%, stop 20% and take-profit 40% from the limit, sized by
   adaptive.sizing (penny limits) on the account's $500. A position is sold at the close of the 21st
   trading day after its entry (its bracket legs are cancelled the night before).
The night's record (universe funnel, purchases, signals, exclusions, random controls, orders) goes to the
hash-chained ledger <state>/p4/ledger.jsonl. Dry run unless ADAPTIVE_SEND=1.

Usage: python -m tracks.penny.p4_live --state <dir> --cache <dir>
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET_XML
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive import accounts, orders, sizing
from core import alpaca, sec
from core.ledger import Ledger
from tracks.penny import p4_screen as S

ET = "America/New_York"
START = pd.Timestamp("2026-10-23")       # first signal day
RUN_AFTER = pd.Timedelta(hours=22)       # Forms 3/4/5 accepted until 22:00 ET keep that day's date
BARS_BACK = pd.Timedelta(days=200)
BUDGET = 500.0
ACCOUNT = "EL-P4"
STOP, TAKE = 0.20, 0.40
DERA_YEARS = 3


# ------------------------------------------------------------------ Form 4 XML (pure)

def _txt(node, path):
    x = node.find(path)
    if x is None:
        return None
    v = x.find("value")
    s = (v.text if v is not None else x.text) or ""
    return s.strip() or None


def _flag(s):
    return (s or "").strip().lower() in ("1", "true")


def parse_form4(xml_text, accn, filed):
    """Open-market purchases (non-derivative, code P, acquired, shares and price > 0) of one Form 4, one
    row per (transaction, officer / director / 10% owner): cik, filed, accn, sk, owner, tdate, value."""
    root = ET_XML.fromstring(xml_text)
    if (root.findtext("documentType") or "").strip() not in ("4", "5"):
        return []
    cik = int(root.findtext("issuer/issuerCik"))
    owners = []
    for o in root.findall("reportingOwner"):
        rel = o.find("reportingOwnerRelationship")
        if rel is not None and any(_flag(rel.findtext(k)) for k in ("isDirector", "isOfficer", "isTenPercentOwner")):
            owners.append(int(o.findtext("reportingOwnerId/rptOwnerCik")))
    out = []
    for sk, t in enumerate(root.findall("nonDerivativeTable/nonDerivativeTransaction"), 1):
        if _txt(t, "transactionCoding/transactionCode") != "P":
            continue
        if _txt(t, "transactionAmounts/transactionAcquiredDisposedCode") != "A":
            continue
        try:
            sh = float(_txt(t, "transactionAmounts/transactionShares"))
            px = float(_txt(t, "transactionAmounts/transactionPricePerShare"))
        except (TypeError, ValueError):
            continue
        if sh <= 0 or px <= 0:
            continue
        td = _txt(t, "transactionDate")
        for ow in sorted(set(owners)):
            out.append({"cik": cik, "filed": pd.Timestamp(filed), "accn": accn, "sk": str(sk), "owner": ow,
                        "tdate": pd.Timestamp(td[:10]) if td else pd.NaT, "value": sh * px})
    return out


# ------------------------------------------------------------------ universe (pure)

def universe_rows(raw, adj, listed, ciks, d):
    """P4's universe on day d (p4_screen.symbol_rows' rules) with ret5, max21 and the 90-day reverse-split
    flag, then one row per company and the day's volatility decile and MAX rank (p4_screen.universe)."""
    rows = []
    adj_by = {s: g for s, g in adj.groupby("symbol")}
    for sym, r in raw.groupby("symbol"):
        if sym not in listed or sym not in ciks or sym not in adj_by:
            continue
        r, a = r.set_index("date").sort_index(), adj_by[sym].set_index("date").sort_index()
        idx = r.index.intersection(a.index)
        idx = idx[idx <= d]
        if not len(idx) or idx[-1] != d:
            continue
        seg = (idx.to_series().diff().dt.days > S.GAP_DAYS).cumsum()
        idx = idx[(seg == seg.iloc[-1]).to_numpy()]
        if len(idx) < S.VOL_DAYS + 1:
            continue
        rr, aa = r.loc[idx], a.loc[idx]
        close = float(rr["c"].iloc[-1])
        dv20 = float((rr["c"] * rr["v"]).iloc[-20:].median())
        if not (S.MIN_PRICE <= close < S.MAX_PRICE and dv20 >= S.MIN_DV20):
            continue
        lr = np.log(aa["c"]).diff()
        rs = S.reverse_split_days(rr, aa)
        rows.append({"date": d, "symbol": sym, "cik": ciks[sym], "close": close, "dv20": dv20,
                     "vol60": float(lr.iloc[-S.VOL_DAYS:].std()),
                     "ret5": float(aa["c"].iloc[-1] / aa["c"].iloc[-1 - S.RUNUP_DAYS] - 1),
                     "max21": float(aa["c"].pct_change().iloc[-S.MAX_DAYS:].max()),
                     "rsplit90": bool(((rs > d - pd.Timedelta(days=S.FILING_LOOKBACK)) & (rs <= d)).any())})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "cik", "close", "dv20", "vol60", "ret5", "max21", "rsplit90",
                                     "vol_dec", "max_pct"])
    return S.universe(pd.DataFrame(rows))


def signals(u, buys, history, filings, d):
    """Qualifying, non-excluded clusters on day d: (signal rows with an `excl` column, all cluster rows)."""
    if not len(buys):
        return pd.DataFrame(), pd.DataFrame()
    history = history[history["cik"].isin(set(buys["cik"]))] if len(history) else history
    allb = pd.concat([history, buys], ignore_index=True) if len(history) else buys
    cl = S.clusters(allb, start=d, end=d)
    cl = cl[cl["cik"].isin(set(buys["cik"]))]
    q = cl[S.qualifies(cl)] if len(cl) else cl
    if not len(q):
        return pd.DataFrame(), cl
    sig = q.merge(u.drop(columns=["date"]), on="cik", how="inner")
    if not len(sig):
        return sig, cl
    sig["offer90"], sig["delist90"] = S.filing_flags(sig, filings)
    sig["excl"] = S.exclusion_reason(sig)
    return sig, cl


def controls(sig, u, d):
    """One random company per accepted signal from the same volatility decile without a signal
    (default_rng(P4 seed + day ordinal)); ledger-only."""
    rng = np.random.default_rng(S.SEED + pd.Timestamp(d).toordinal())
    pool = u[~u["cik"].isin(set(sig["cik"]))].sort_values("symbol")
    out, used = [], set()
    for _, s in sig.sort_values("symbol").iterrows():
        p = pool[(pool["vol_dec"] == s["vol_dec"]) & ~pool["cik"].isin(used)]
        if not len(p):
            out.append({"for": s["symbol"], "symbol": None})
            continue
        c = p.iloc[int(rng.integers(0, len(p)))]
        used.add(c["cik"])
        out.append({"for": s["symbol"], "symbol": c["symbol"], "close": float(c["close"])})
    return out


def entry_order(s, day):
    limit = float(s["close"]) * (1 + S.ENTRY_LIMIT)
    shares, why = sizing.size(BUDGET, "long", STOP, limit, float(s["dv20"]) / 78, 0.0, sizing.PENNY)
    if shares < 1:
        return {"kind": "skip", "symbol": s["symbol"], "why": f"under one share ({why})"}
    return {"kind": "entry", "symbol": s["symbol"], "why": why,
            "body": orders.bracket(s["symbol"], "long", shares, limit, limit * (1 - STOP), limit * (1 + TAKE),
                                   client_id=f"el-P4-{day}-{s['symbol']}-in")}


def blocked(led, d, tdays):
    """Companies with an accepted signal in the last HOLD + 1 trading days: their trade may still be open
    (P4: a stock re-enters only after its previous trade has closed; a trade lasts at most HOLD days)."""
    i = tdays.index(d)
    since = f"{tdays[max(0, i - S.HOLD - 1)]:%Y-%m-%d}"
    out = set()
    for e in led.entries():
        if e["kind"] == "P4_day" and e["date"] >= since:
            out |= set(json.loads((led.root / e["file"]).read_text()).get("open_ciks", []))
    return out


def held_days(filled_at, today, tdays):
    """Trading days from the entry day (day 0) to `today`."""
    e = pd.Timestamp(filled_at).tz_convert(ET).tz_localize(None).normalize() if pd.Timestamp(filled_at).tzinfo \
        else pd.Timestamp(filled_at).normalize()
    return sum(1 for t in tdays if e < t <= today)


# ------------------------------------------------------------------ network

class Api:
    def __init__(self, hdrs=None):
        self.h = hdrs

    def calendar(self, start, end):
        c = alpaca.trading_calendar(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
        c["date"] = pd.to_datetime(c["date"])
        return c

    def assets(self):
        return alpaca.active_assets()

    def tickers(self):
        from tracks.main.longterm import sec_data
        return sec_data.current_ticker_map()

    def bars(self, symbols, start, end, adjustment):
        from tracks.penny.nightly import Api as PennyApi
        return PennyApi().bars(symbols, start, end, adjustment)

    def submissions(self, cik, first, last):
        from tracks.news.l1_study import submissions_rows
        return submissions_rows(cik, first=first, last=last, docs=True)

    def form4_xml(self, cik, accn, doc):
        r = sec.get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{doc.split('/')[-1]}")
        return None if r is None else r.text

    def history(self, year, cache):
        from tracks.main.longterm import sec_data
        b = sec_data.dera_owner_buys(year - DERA_YEARS, pd.Period(f"{year - 1}Q4"), Path(cache) / "dera")
        return b[b["filed"].dt.year >= year - DERA_YEARS]

    # trading (paper only)
    def positions(self):
        return {p["symbol"]: p for p in alpaca.get(f"{alpaca.TRADE}/positions", {}, headers=self.h)}

    def open_orders(self):
        return alpaca.get(f"{alpaca.TRADE}/orders", {"status": "open", "nested": "true"}, headers=self.h)

    def order_by_client_id(self, cid):
        try:
            return alpaca.get(f"{alpaca.TRADE}/orders:by_client_order_id", {"client_order_id": cid},
                              headers=self.h, tries=3)
        except Exception as e:  # noqa: BLE001 - an unknown order is recorded, not fatal
            return {"error": str(e)[:200]}

    def cancel(self, order_id):
        import requests
        url = f"{alpaca.TRADE}/orders/{order_id}"
        assert url.startswith("https://paper-api.alpaca.markets/")
        return requests.delete(url, headers=self.h, timeout=60).status_code

    def order(self, body):
        return alpaca.post(f"{alpaca.TRADE}/orders", body, headers=self.h)


def form4_issuers(state, d, ciks):
    """Issuer CIKs in `ciks` that appear on a Form 4 row of d's logged form index, or None if not logged."""
    p = Path(state) / "leads" / "sec" / f"{d:%Y-%m-%d}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return sorted({r["cik"] for r in rows if r.get("form") == "4" and r["cik"] in ciks})


def window_buys(api, cik, d):
    """(purchases filed in the 10 days ending d, the issuer's filings over the 90-day lookback)."""
    subs = api.submissions(cik, d - pd.Timedelta(days=S.FILING_LOOKBACK + 5), d)
    lo = d - pd.Timedelta(days=S.WINDOW_DAYS - 1)
    rows = []
    for f in subs:
        if f["form"] in ("4", "5") and lo <= pd.Timestamp(f["filed"]) <= d and f.get("doc"):
            x = api.form4_xml(cik, f["accession"], f["doc"])
            if x:
                try:
                    rows += [r for r in parse_form4(x, f["accession"], f["filed"]) if r["cik"] == int(cik)]
                except ET_XML.ParseError:
                    print(f"[p4] {f['accession']}: unreadable XML")
    fl = pd.DataFrame([{"cik": int(cik), "accession": f["accession"], "form": f["form"], "filed": f["filed"],
                        "items": ",".join(f.get("items") or [])} for f in subs],
                      columns=["cik", "accession", "form", "filed", "items"])
    return rows, fl


def manage_exits(api, led, today, tdays):
    """Cancel the bracket legs and send a market-on-close sell for positions due out at the next session's
    close (the 21st trading day after entry)."""
    sent_in = {}
    for e in led.entries():
        rec = json.loads((led.root / e["file"]).read_text())
        for o in rec.get("orders", []):
            if o.get("kind") == "entry" and o.get("http", 999) < 300:
                sent_in[o["symbol"]] = o["body"]["client_order_id"]
    out = []
    pos = api.positions()
    nxt = [t for t in tdays if t > today]
    for sym, p in sorted(pos.items()):
        cid = sent_in.get(sym)
        if cid is None:
            continue
        o = api.order_by_client_id(cid)
        if not o.get("filled_at"):
            continue
        if held_days(o["filled_at"], today, tdays) + 1 < S.HOLD or not nxt:
            continue
        for oo in api.open_orders():
            for leg in [oo] + list(oo.get("legs") or []):
                if leg.get("symbol") == sym and leg.get("side") == "sell" and leg.get("status") in ("new", "held", "accepted"):
                    api.cancel(leg["id"])
        body = {"symbol": sym, "qty": str(int(abs(float(p["qty"])))), "side": "sell", "type": "market",
                "time_in_force": "cls", "client_order_id": f"el-P4-{nxt[0]:%Y-%m-%d}-{sym}-time"}
        code, js = api.order(body)
        out.append({"kind": "time exit", "symbol": sym, "body": body, "http": code,
                    "status": js.get("status") if code < 300 else "rejected", "message": js.get("message")})
    return out


def run(state, cache, now=None, api=None, send=None, start=START):
    send = (os.environ.get("ADAPTIVE_SEND") == "1") if send is None else send
    if api is None:
        api = Api(accounts.headers(ACCOUNT))
    now = (now if now is not None else alpaca.utcnow()).tz_convert(ET).tz_localize(None)
    led = Ledger(Path(state) / "p4")
    cal = api.calendar(start - BARS_BACK, now + pd.Timedelta(days=40))
    tdays = list(cal["date"])
    complete = [t for t in tdays if t >= start and t + RUN_AFTER <= now]
    done = {e["date"] for e in led.entries() if e["kind"] == "P4_day"}
    todo = [t for t in complete if f"{t:%Y-%m-%d}" not in done]
    if not todo:
        print("p4: up to date")
        return 0
    assets = api.assets()
    common = assets[assets["excl_reason"].isna()]
    listed = set(common[common["exchange"].isin(S.LISTED)]["symbol"])
    tk = api.tickers()
    ciks = dict(zip(tk["symbol"], tk["cik"].astype(int)))
    raw = api.bars(listed & set(ciks), todo[0] - BARS_BACK, todo[-1], "raw")
    hist = {}
    n = 0
    for d in todo:
        ds = f"{d:%Y-%m-%d}"
        day_raw = raw[raw["date"] <= d]
        last = day_raw[day_raw["date"] == d]
        cand = set(last[(last["c"] >= S.MIN_PRICE) & (last["c"] < S.MAX_PRICE)]["symbol"])
        adj = api.bars(cand, d - BARS_BACK, d, "all") if cand else raw.iloc[:0]
        u = universe_rows(day_raw[day_raw["symbol"].isin(cand)], adj, listed, ciks, d)
        issuers = form4_issuers(state, d, set(u["cik"]))
        if issuers is None:
            print(f"p4: {ds} form index not logged yet; stopping")
            break
        if d.year not in hist:
            hist[d.year] = api.history(d.year, cache)
        buys, fl = [], []
        for c in issuers:
            b, f = window_buys(api, c, d)
            buys += b
            fl.append(f)
        buys = pd.DataFrame(buys, columns=["cik", "filed", "accn", "sk", "owner", "tdate", "value"])
        filings = pd.concat(fl, ignore_index=True) if fl else pd.DataFrame(columns=["cik", "accession", "form", "filed", "items"])
        sig, cl = signals(u, buys, hist[d.year], filings, d)
        held = blocked(led, d, tdays)
        acc = sig[sig["excl"].isna() & ~sig["cik"].isin(held)] if len(sig) else sig
        rec = {"day": ds, "send": send, "universe": int(len(u)), "form4_issuers": len(issuers),
               "purchases": json.loads(buys.to_json(orient="records", date_format="iso")),
               "clusters": json.loads(cl.to_json(orient="records", date_format="iso")) if len(cl) else [],
               "signals": json.loads(sig.drop(columns=[c for c in ("rsplit90",) if c in sig]).to_json(
                   orient="records", date_format="iso")) if len(sig) else [],
               "controls": controls(acc, u, d) if len(acc) else [], "orders": []}
        i = tdays.index(d)
        nxt_open = pd.Timestamp(f"{tdays[i + 1]:%Y-%m-%d} {cal['open'].iloc[i + 1]}") if i + 1 < len(tdays) else None
        live = d == complete[-1] and nxt_open is not None and now < nxt_open - pd.Timedelta(minutes=5)
        if len(acc):
            rec["open_ciks"] = sorted(int(c) for c in acc["cik"])
            for _, s in acc.iterrows():
                o = entry_order(s, ds)
                if live and send and "body" in o:
                    code, js = api.order(o["body"])
                    o.update(http=code, status=js.get("status") if code < 300 else "rejected", message=js.get("message"))
                elif "body" in o:
                    o["status"] = "dry run" if live else "not sent: processed after the next open"
                rec["orders"].append(o)
        if live and send:
            rec["orders"] += manage_exits(api, led, d, tdays)
        led.write_json("P4_day", ds, f"days/{ds}.json", rec)
        print(f"p4: {ds}: universe {len(u)}, form 4 issuers {len(issuers)}, purchases {len(buys)}, "
              f"signals {len(sig)}, accepted {len(acc)}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    run(a.state, a.cache)


if __name__ == "__main__":
    main()
