"""DAY1, the all-day $1,000 paper bot (research/prereg/DAY1.md). Fake money only.

The sleeves' own scorers run beside this process in the same job (tracks/news/nx3_live.py and
tracks/main/chr_live.py, each with its own state under <state>/sleeves, no orders of their own). Every minute this
boss reads their new decisions, funds the ones that pass the written rules (expected move >= 2x the round trip,
$50-$200 by confidence, sleeve budgets, 50% overnight, -5% day stop, -10% pause), draws a random twin for each
funded trade from the same sleeve's same-day pool, and sends long orders to the Alpaca paper account EL-DAY1.
Shorts a sleeve wants are logged as hypothetical trades and never sent. Stops and targets are checked every
minute on IEX last trades, for the bot and the twin alike; time exits settle at the SIP daily close.

The forward test reads the ledger (kind day1_day), not Alpaca's fills: both books carry the same cost model.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive import accounts
from adaptive import costs
from core import alpaca
from core.ledger import Ledger

ET = "America/New_York"
ACCOUNT = "EL-DAY1"
BOOK = 1000.0
BUDGET = {"nx3": 0.50, "chrp": 0.25}            # the rest is cash; shadow sleeves earn a share by rule 20
MIN_USD, MAX_USD = 50.0, 200.0
OVERNIGHT = 0.50
DAY_STOP = 0.05
PAUSE = 0.10
COST_X = 2.0
TICK_S = 60
LATE = pd.Timedelta(minutes=15)                  # a sleeve decision older than this is logged "late", not funded
SETTLE_AFTER = pd.Timedelta(minutes=20)
SEED = 20261102
CHR_SESSIONS = 5
CHR_HIST_NOTIONAL = 250.0                        # CH's labels subtract costs at this size (ch_build.NOTIONAL)


# ---------- rules ----------

def band(price):
    return "main" if price >= 5 else "penny"


def qty_for(price, usd):
    """Main names by dollar amount (fractional), pennies in whole shares."""
    return round(usd / price, 4) if band(price) == "main" else float(int(usd // price))


MAX_QUOTED = 0.10                                # a quoted spread wider than this is treated as no quote


def quoted_spread(quote):
    """(ask - bid) / mid from an IEX (bid, ask) pair, or None when the quote is missing, locked, crossed or wider
    than MAX_QUOTED."""
    if not quote:
        return None
    bid, ask = quote
    if not (bid and ask and ask > bid > 0):
        return None
    s = (ask - bid) / ((ask + bid) / 2)
    return s if s <= MAX_QUOTED else None


def round_trip(price, usd, spread=None, moc_exit=False):
    """Round-trip cost as a fraction of the position (DAY1.md, "Cost per trade", 2026-11-02): half the trade's own
    quoted spread to enter, half again to exit unless the exit is a closing-auction (market-on-close) order, which
    pays no spread, plus sale fees (Alpaca charges no commission). Without a usable quote, the band's measured
    average spread (adaptive/costs.py) stands in."""
    sh = qty_for(price, usd)
    if sh <= 0:
        return math.inf
    value = sh * price
    s = costs.SPREAD_ROUND_TRIP[band(price)] if spread is None else spread
    return s / 2 + (0.0 if moc_exit else s / 2) + costs.regulatory(sh, price, True) / value


def size_usd(conf):
    """$50 at confidence 0 to $200 at 1 (linear)."""
    c = min(max(float(conf), 0.0), 1.0)
    return MIN_USD + (MAX_USD - MIN_USD) * c


def chr_cost(price):
    """The cost CH's labels already subtracted for a long (ch_build.cost_frac at its $250 basis, no borrow)."""
    sh = max(1, int(CHR_HIST_NOTIONAL // price))
    c = costs.fill_cost("penny", sh, price, False) + costs.fill_cost("penny", sh, price, True)
    return c / (sh * price)


# ---------- the sleeves' decisions ----------

def _jsonl(f):
    """Lines of a JSON-lines file; a line still being written by a sleeve is skipped until the next read."""
    f = Path(f)
    out = []
    for x in (f.read_text().splitlines() if f.exists() else []):
        try:
            out.append(json.loads(x))
        except ValueError:
            continue
    return out


def _json(f, default):
    f = Path(f)
    return json.loads(f.read_text()) if f.exists() else default


def nx3_cands(sleeves, d):
    """NX3's fired events of day d (no daily cap, as NX3-ALL): (candidates, shorts, pool). Expected move =
    e x vol20 x sqrt(1.5), the scale NX3's label was divided by. Confidence |e|/cut from 1 to 2 -> 0 to 1."""
    day = Path(sleeves) / "nx3" / "days" / f"{d:%Y-%m-%d}"
    cut = _json(day / "ctx.json", {}).get("cut")
    cands, shorts, pool = [], [], []
    for x in _jsonl(day / "decisions.jsonl"):
        if x.get("status") != "decided" or not x.get("entry_iex"):
            continue
        pool.append({"symbol": x["symbol"]})
        if not x.get("fired") or not cut:
            continue
        v = (x.get("inputs") or {}).get("vol20")
        if v is None or not np.isfinite(v):
            continue
        c = {"id": f"nx3:{x['key']}", "sleeve": "nx3", "symbol": x["symbol"], "at": x["decided_at"],
             "expected": abs(x["e"]) * v * math.sqrt(1.5), "conf": min(abs(x["e"]) / cut, 2.0) - 1.0,
             "exit": {"type": "close", "sessions": 1}}
        (cands if x["e"] > 0 else shorts).append(c)
    return cands, shorts, pool


def chr_cands(sleeves, d, labels):
    """CHR-P's triggers of day d whose chosen label passed its cut. Expected move = the net prediction plus the
    cost the label subtracted. Confidence from its 2%-10% size scale."""
    trig = _json(Path(sleeves) / "chrp" / "work" / f"{d:%Y-%m-%d}" / "triggers.json", {})
    cands, shorts, pool = [], [], []
    for sym, r in sorted(trig.items()):
        if not r.get("pred") or not np.isfinite(r.get("entry") or np.nan):
            continue
        pool.append({"symbol": sym, "atr": r.get("atr")})
        if r.get("reason") in ("below the cut", "late") or str(r.get("reason", "")).startswith("inputs failed"):
            continue
        lab = r["label"]
        side, k = lab.split("_")
        c = {"id": f"chrp:{d:%Y-%m-%d}:{sym}", "sleeve": "chrp", "symbol": sym, "at": r["decided"],
             "expected": float(r["pred"][labels.index(lab)]) + chr_cost(float(r["entry"])),
             "conf": (float(r.get("size") or 0.02) - 0.02) / 0.08,
             "exit": {"type": "barrier", "k": float(k), "atr": float(r["atr"]), "sessions": CHR_SESSIONS}}
        (cands if side == "long" else shorts).append(c)
    return cands, shorts, pool


# ---------- state ----------

class State:
    def __init__(self, root):
        self.root = Path(root) / "day1"
        self.ledger = Ledger(self.root / "ledger")
        self.work = self.root / "work"
        self.work.mkdir(parents=True, exist_ok=True)

    def load(self, name, default):
        return _json(self.work / name, default)

    def save(self, name, obj):
        f = self.work / name
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str))
        tmp.replace(f)

    def log(self, d, rec):
        f = self.work / f"{d:%Y-%m-%d}" / "decisions.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def seen(self, d):
        return {x["id"] for x in _jsonl(self.work / f"{d:%Y-%m-%d}" / "decisions.jsonl")}


def end_session(sessions, d, n):
    """The n-th session after d (n = 0 is d itself)."""
    later = [s for s in sessions if s >= d]
    return later[min(n, len(later) - 1)] if later else d


# ---------- broker ----------

class Broker:
    def __init__(self, h):
        self.h = h

    def order(self, body):
        return alpaca.post(f"{alpaca.TRADE}/orders", body, headers=self.h)


def broker():
    h = accounts.headers(ACCOUNT)
    return Broker(h) if h else None


def entry_order(pos):
    sym, px = pos["symbol"], pos["entry"]
    body = {"symbol": sym, "side": "buy", "time_in_force": "day",
            "client_order_id": f"day1-{pos['date'].replace('-', '')}-{sym}-in"}
    if band(px) == "main":
        return body | {"type": "market", "notional": f"{pos['usd']:.2f}"}
    return body | {"type": "limit", "qty": str(int(pos["qty"])), "limit_price": f"{px * 1.01:.2f}"}


def exit_order(pos, moc, tag):
    body = {"symbol": pos["symbol"], "side": "sell", "qty": f"{pos['qty']:g}",
            "client_order_id": f"day1-{pos['date'].replace('-', '')}-{pos['symbol']}-{tag}"}
    return body | ({"type": "market", "time_in_force": "cls"} if moc else {"type": "market", "time_in_force": "day"})


def send(br, body, sink):
    if br is None:
        sink.append({"body": body, "sent": False})
        return
    try:
        code, js = br.order(body)
        sink.append({"body": body, "http": code, "status": js.get("status"),
                     "message": None if code < 300 else js.get("message")})
        print(f"[day1-orders] {body['side']} {body['symbol']}: {code} {js.get('status') or js.get('message')}")
    except Exception as e:  # noqa: BLE001 - orders never stop the ledger book
        sink.append({"body": body, "error": f"{type(e).__name__}: {e}"[:200]})


# ---------- one minute ----------

def equity(st, marks):
    """Realised P&L plus open bot positions marked at `marks` (falls back to entry)."""
    acct = st.load("account.json", {"realised": 0.0, "twin_realised": 0.0, "peak": BOOK, "paused": False})
    pos = st.load("positions.json", [])
    unreal = sum(p["usd"] * (marks.get(p["symbol"], p["entry"]) / p["entry"] - 1) for p in pos if p["role"] == "bot")
    return BOOK + acct["realised"] + unreal, acct


def fund(st, api, br, now, d, cands, pools, sessions, short_c):
    """Decides every new candidate in time order (ties: bigger expected move first)."""
    seen = st.seen(d)
    new = [c for c in cands if c["id"] not in seen]
    shorts = [c for c in short_c if c["id"] not in seen]
    if not new and not shorts:
        return 0
    pos = st.load("positions.json", [])
    orders = st.load(f"{d:%Y-%m-%d}/orders.json", [])
    syms = sorted({c["symbol"] for c in new + shorts} | {p["symbol"] for p in pos if p["role"] == "bot"}
                  | {x["symbol"] for pl in pools.values() for x in pl})
    px = api.last_trades(syms) if syms else {}
    qt = api.quotes(sorted({c["symbol"] for c in new} | {x["symbol"] for pl in pools.values() for x in pl})) \
        if new else {}
    eq, acct = equity(st, {s: v[0] for s, v in px.items()})
    prev = acct.get("prev_close", BOOK)
    stopped = eq <= prev * (1 - DAY_STOP)
    if eq <= acct["peak"] * (1 - PAUSE) and not acct["paused"]:
        acct["paused"] = True
        acct["paused_at"] = str(now)
        st.save("account.json", acct)
        print(f"[day1] PAUSED: equity {eq:.2f} is 10% below the peak {acct['peak']:.2f}; Amit restarts it")
    rng_state = st.load(f"{d:%Y-%m-%d}/twin_rng.json", {"draws": 0})
    rng = np.random.default_rng(SEED + d.toordinal())
    rng.integers(1 << 30, size=rng_state["draws"])                  # replay earlier draws: same stream all day
    funded = 0
    for c in sorted(new, key=lambda c: (str(c["at"]), -c["expected"], c["symbol"])):
        rec = {"id": c["id"], "sleeve": c["sleeve"], "symbol": c["symbol"], "at": c["at"], "decided": str(now),
               "expected": c["expected"], "conf": c["conf"]}
        p = px.get(c["symbol"], (None, None))[0]
        usd = size_usd(c["conf"])
        moc = c["exit"]["type"] == "close"
        rt = round_trip(p, usd, quoted_spread(qt.get(c["symbol"])), moc) if p else math.inf
        rec |= {"usd": usd, "entry": p, "round_trip": rt, "quote": qt.get(c["symbol"])}
        end = f"{end_session(sessions, d, c['exit']['sessions']):%Y-%m-%d}"
        open_bot = [x for x in pos if x["role"] == "bot"]
        overnight = sum(x["usd"] for x in open_bot if x["end"] > f"{d:%Y-%m-%d}")
        why = None
        if acct["paused"]:
            why = "paused"
        elif stopped:
            why = "day stop"
        elif now - pd.Timestamp(c["at"]) > LATE:
            why = "late"
        elif not p:
            why = "no quote"
        elif c["expected"] < COST_X * rt:
            why = "cost check"
        elif any(x["symbol"] == c["symbol"] for x in open_bot):
            why = "duplicate"
        # a sleeve's budget counts what it will still hold after today's close: a position whose time exit is
        # today's close frees its share for the next day's entries (else NX3 could only trade every other day)
        elif sum(x["usd"] for x in open_bot if x["sleeve"] == c["sleeve"] and x["end"] > f"{d:%Y-%m-%d}") \
                + usd > BUDGET[c["sleeve"]] * BOOK + 1e-9:
            why = "sleeve budget"
        elif sum(x["usd"] for x in open_bot) + usd > BOOK + 1e-9:
            why = "fully invested"
        elif end > f"{d:%Y-%m-%d}" and overnight + usd > OVERNIGHT * BOOK + 1e-9:
            why = "overnight cap"
        if why:
            st.log(d, rec | {"funded": False, "reason": why})
            continue
        bot = {"id": c["id"], "role": "bot", "sleeve": c["sleeve"], "symbol": c["symbol"], "date": f"{d:%Y-%m-%d}",
               "entry": p, "entry_t": px[c["symbol"]][1], "usd": usd, "qty": qty_for(p, usd), "round_trip": rt,
               "end": end, "exit": c["exit"], "expected": c["expected"]}
        if bot["exit"]["type"] == "barrier":
            k, a = bot["exit"]["k"], bot["exit"]["atr"]
            bot["stop"], bot["target"] = p - k * a, p + 2 * k * a
        if bot["qty"] <= 0:
            st.log(d, rec | {"funded": False, "reason": "under one share"})
            continue
        pos.append(bot)
        send(br, entry_order(bot), orders)
        # the twin: a random other stock from the same sleeve's pool today, same size, minute and exit rule
        pool = sorted({x["symbol"]: x for x in pools[c["sleeve"]] if x["symbol"] != c["symbol"]}.items())
        twin = None
        for _ in range(5):
            if not pool:
                break
            i = int(rng.integers(len(pool)))
            rng_state["draws"] += 1
            s, info = pool[i]
            tp = px.get(s, (None, None))[0]
            if tp:
                twin = {**bot, "id": c["id"] + ":twin", "role": "twin", "symbol": s, "entry": tp,
                        "entry_t": px[s][1], "qty": qty_for(tp, usd),
                        "round_trip": round_trip(tp, usd, quoted_spread(qt.get(s)), moc)}
                if twin["exit"]["type"] == "barrier":
                    ta = float(info.get("atr") or np.nan)
                    twin["exit"] = {**twin["exit"], "atr": ta}
                    twin["stop"], twin["target"] = tp - twin["exit"]["k"] * ta, tp + 2 * twin["exit"]["k"] * ta
                pos.append(twin)
                break
            pool.pop(i)
        st.log(d, rec | {"funded": True, "twin": twin["symbol"] if twin else None})
        funded += 1
    for c in shorts:          # logged only: a $1,000 account can't short (DAY1.md)
        p = px.get(c["symbol"], (None, None))[0]
        usd = size_usd(c["conf"])
        rec = {"id": c["id"], "sleeve": c["sleeve"], "symbol": c["symbol"], "at": c["at"], "decided": str(now),
               "expected": c["expected"], "conf": c["conf"], "usd": usd, "entry": p, "funded": False,
               "reason": "short (logged)"}
        st.log(d, rec)
        if p:
            end = end_session(sessions, d, c["exit"]["sessions"])
            sh = {"id": c["id"], "role": "short", "sleeve": c["sleeve"], "symbol": c["symbol"],
                  "date": f"{d:%Y-%m-%d}", "entry": p, "usd": usd, "qty": usd / p, "round_trip": round_trip(p, usd),
                  "end": f"{end:%Y-%m-%d}", "exit": c["exit"]}
            if c["exit"]["type"] == "barrier":
                k, a = c["exit"]["k"], c["exit"]["atr"]
                sh["stop"], sh["target"] = p + k * a, p - 2 * k * a
            pos.append(sh)
    st.save("positions.json", pos)
    st.save(f"{d:%Y-%m-%d}/orders.json", orders)
    st.save(f"{d:%Y-%m-%d}/twin_rng.json", rng_state)
    return funded


def check_barriers(st, api, br, now, d):
    """Closes every barrier position whose stop or target the IEX last trade has reached (a trade beyond the
    stop, as after a gap, exits at that trade)."""
    pos = st.load("positions.json", [])
    live = [p for p in pos if p["exit"]["type"] == "barrier"]
    if not live:
        return 0
    px = api.last_trades(sorted({p["symbol"] for p in live}))
    closed = st.load("closed.json", [])
    orders = st.load(f"{d:%Y-%m-%d}/orders.json", [])
    keep, n = [], 0
    for p in pos:
        q = px.get(p["symbol"], (None, None))[0] if p in live else None
        hit = None
        if q is not None and np.isfinite(p.get("stop", np.nan)):
            sgn = -1 if p["role"] == "short" else 1
            if sgn * (q - p["stop"]) <= 0:
                hit = ("stop", min(q, p["stop"]) if sgn > 0 else max(q, p["stop"]))
            elif sgn * (q - p["target"]) >= 0:
                hit = ("target", p["target"])
        if hit:
            closed.append(close_rec(p, hit[1], f"{d:%Y-%m-%d}", hit[0]))
            if p["role"] == "bot":
                send(br, exit_order(p, False, hit[0]), orders)
            n += 1
        else:
            keep.append(p)
    st.save("positions.json", keep)
    st.save("closed.json", closed)
    st.save(f"{d:%Y-%m-%d}/orders.json", orders)
    return n


def close_rec(p, exit_px, day, why):
    sgn = -1 if p["role"] == "short" else 1
    net = sgn * (exit_px / p["entry"] - 1) - p["round_trip"]
    return p | {"exit_px": exit_px, "exit_day": day, "exit_why": why, "net": net, "pnl": p["usd"] * net}


def send_moc(st, br, d):
    """Once a session, market-on-close sells for every bot position whose time exit is today."""
    sent = st.load(f"{d:%Y-%m-%d}/moc.json", None)
    if sent is not None:
        return
    orders = []
    for p in st.load("positions.json", []):
        if p["role"] == "bot" and p["end"] == f"{d:%Y-%m-%d}":
            send(br, exit_order(p, True, "moc"), orders)
    st.save(f"{d:%Y-%m-%d}/moc.json", orders)


def after_close(st, api, d):
    """Settles every position whose time exit is today at the SIP daily close (raw), marks the rest, and logs the
    day once (ledger kind day1_day)."""
    ds = f"{d:%Y-%m-%d}"
    if st.ledger.has("day1_day", ds):
        return False
    pos = st.load("positions.json", [])
    closed = st.load("closed.json", [])
    syms = sorted({p["symbol"] for p in pos})
    closes = api.closes(syms, d) if syms else {}
    keep = []
    for p in pos:
        c = closes.get(p["symbol"])
        if p["end"] <= ds and c is not None:
            closed.append(close_rec(p, c, ds, "time"))
        else:
            keep.append(p)
    acct = st.load("account.json", {"realised": 0.0, "twin_realised": 0.0, "peak": BOOK, "paused": False})
    today = [x for x in closed if x["exit_day"] == ds]
    for role, key in (("bot", "realised"), ("twin", "twin_realised")):
        acct[key] = acct.get(key, 0.0) + sum(x["pnl"] for x in today if x["role"] == role)
    marks = {p["symbol"]: closes.get(p["symbol"], p["entry"]) for p in keep}
    eq = BOOK + acct["realised"] + sum(p["usd"] * (marks[p["symbol"]] / p["entry"] - 1) for p in keep
                                        if p["role"] == "bot")
    teq = BOOK + acct["twin_realised"] + sum(p["usd"] * (marks[p["symbol"]] / p["entry"] - 1) for p in keep
                                             if p["role"] == "twin")
    prev = acct.get("prev_close", BOOK)
    tprev = acct.get("twin_prev_close", BOOK)
    rec = {"day": ds, "equity": eq, "twin_equity": teq, "ret": eq / prev - 1, "twin_ret": teq / tprev - 1,
           "closed": [x for x in today], "open": keep, "decisions": _jsonl(st.work / ds / "decisions.jsonl"),
           "orders": st.load(f"{ds}/orders.json", []) + (st.load(f"{ds}/moc.json", []) or []),
           "paused": acct["paused"], "budget": BUDGET, "book": BOOK}
    acct |= {"prev_close": eq, "twin_prev_close": teq, "peak": max(acct["peak"], eq)}
    st.save("positions.json", keep)
    st.save("closed.json", closed)
    st.save("account.json", acct)
    st.ledger.write_json("day1_day", ds, f"days/{ds}.json", json.loads(json.dumps(rec, default=str)))
    print(f"[day1] {ds}: equity {eq:.2f} ({rec['ret']:+.3%}), twin {teq:.2f} ({rec['twin_ret']:+.3%}), "
          f"{sum(1 for x in today if x['role'] == 'bot')} bot trades closed")
    return True


# ---------- data ----------

class Api:
    def __init__(self):
        from core.env import alpaca_headers
        self.h = alpaca_headers()

    def last_trades(self, syms):
        out = {}
        for i in range(0, len(syms), 200):
            j = alpaca.get("https://data.alpaca.markets/v2/stocks/trades/latest",
                           {"symbols": ",".join(syms[i:i + 200]), "feed": "iex"}, headers=self.h)
            for s, tr in (j.get("trades") or {}).items():
                if tr and tr.get("p"):
                    out[s] = (float(tr["p"]), str(tr.get("t")))
        return out

    def quotes(self, syms):
        """{symbol: (bid, ask)} from IEX's latest quotes."""
        out = {}
        for i in range(0, len(syms), 200):
            j = alpaca.get("https://data.alpaca.markets/v2/stocks/quotes/latest",
                           {"symbols": ",".join(syms[i:i + 200]), "feed": "iex"}, headers=self.h)
            for s, q in (j.get("quotes") or {}).items():
                if q:
                    out[s] = (float(q.get("bp") or 0), float(q.get("ap") or 0))
        return out

    def closes(self, syms, d):
        """The SIP daily close of d (adjustment raw: the positions' own prices)."""
        from core.bars import fetch_daily_bars, to_frames
        fr = to_frames(fetch_daily_bars(syms, f"{d:%Y-%m-%d}", alpaca.sip_end(d), adjustment="raw", headers=self.h))
        out = {}
        for s, f in fr.items():
            f = f[f.index.normalize() == pd.Timestamp(d).normalize()] if len(f) else f
            if len(f):
                out[s] = float(f["c"].iloc[-1])
        return out

    def sessions(self, d):
        cal = alpaca.trading_calendar(d - pd.Timedelta(days=5), d + pd.Timedelta(days=20))
        return cal, sorted(pd.to_datetime(cal["date"]))


def tick(st, api, br, now, sleeves, labels, sessions):
    d = now.normalize()
    nc, ns, npool = nx3_cands(sleeves, d)
    cc, cs, cpool = chr_cands(sleeves, d, labels)
    n_ex = check_barriers(st, api, br, now, d)
    n_in = fund(st, api, br, now, d, nc + cc, {"nx3": npool, "chrp": cpool}, sessions, ns + cs)
    return n_in, n_ex


def run(state, minutes, sleeves, api=None, br=None, clock=None, sleep=time.sleep, labels=None):
    st = State(state)
    api = api or Api()
    br = br if br is not None else broker()
    if labels is None:
        from tracks.main import ch_build as B
        labels = B.LABELS
    clock = clock or (lambda: alpaca.utcnow().tz_convert(ET).tz_localize(None))
    stop = time.monotonic() + minutes * 60
    while time.monotonic() < stop:
        now = clock()
        d = now.normalize()
        cal, sessions = api.sessions(d)
        row = cal[pd.to_datetime(cal["date"]) == d]
        if row.empty:
            return 3
        hm = lambda s: d + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:5]))
        open_t, close_t = hm(row["open"].iloc[0]), hm(row["close"].iloc[0])
        if now >= close_t + SETTLE_AFTER:
            after_close(st, api, d)
            return 3
        if open_t <= now < close_t:
            n_in, n_ex = tick(st, api, br, now, sleeves, labels, sessions)
            if now >= close_t - pd.Timedelta(minutes=15):
                send_moc(st, br, d)
            print(f"[day1] {now:%H:%M} funded {n_in}, barrier exits {n_ex}")
        sleep(max(1.0, TICK_S - (clock() - now).total_seconds()))
    return 0


def resume(state):
    st = State(state)
    acct = st.load("account.json", {"realised": 0.0, "twin_realised": 0.0, "peak": BOOK, "paused": False})
    if acct.get("paused"):
        eq = BOOK + acct["realised"]
        acct |= {"paused": False, "peak": eq, "resumed": str(pd.Timestamp.now(tz=ET))}
        st.save("account.json", acct)
        st.ledger.write_json("day1_resume", f"{pd.Timestamp.now(tz=ET):%Y-%m-%d}", "resume.json", acct)
        print("[day1] resumed; the peak restarts at the current realised equity")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--sleeves", default="state/sleeves")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.resume:
        return resume(a.state)
    if a.check:
        api = Api()
        tr = api.last_trades(["SPY", "AAPL"])
        q = api.quotes(["SPY", "AAPL"])
        print(f"[day1 check] IEX last trades {tr}; quotes {q}; broker {'set' if broker() else 'missing (no orders)'}")
        return 0
    return run(a.state, a.minutes, a.sleeves)


if __name__ == "__main__":
    sys.exit(main())
