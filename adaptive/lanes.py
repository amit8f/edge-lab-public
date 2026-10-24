"""SYS2B lanes (research/prereg/SYS2B.md): what plain code watches, the shared checks, and how alerts are
grouped into events. Pure functions only; adaptive/scouts.py does the I/O.

An alert is one lane's sighting: {"lane", "symbol", "item", "src_t", "side_hint", "info"}. `item` names the
source (a news id, a filing accession, or lane:symbol:day for numbers), so two lanes that read the same
item land in the same event.
"""
import math
import re

import pandas as pd

ET = "America/New_York"

# lane id: (name, group, trigger, ready on 2026-10-27)
LANES = {
    "1": ("Earnings surprise", "fundamental", True, True),
    "2": ("Guidance change", "fundamental", True, True),
    "3": ("Fresh filing", "fundamental", True, True),
    "4": ("Insider buying", "ownership", True, True),
    "5": ("Activist stake", "ownership", True, True),
    "6": ("Analyst moves", "fundamental", True, True),
    "7": ("Share supply", "ownership", True, True),
    "8": ("Biotech catalyst", "fundamental", True, True),
    "9": ("Contract win", "fundamental", True, True),
    "10": ("Takeover", "fundamental", True, True),
    "11": ("Volume with no news", "trading", True, True),
    "13": ("Chart breakout", "trading", True, True),
    "14": ("Oversold bounce (experimental)", "trading", True, True),
    "15": ("Sympathy move", "trading", True, True),
    "17a": ("Squeeze long", "ownership", True, True),
    "17b": ("Hard-to-borrow short", "ownership", True, True),
    "21": ("Adverse event", "fundamental", True, True),
    "22": ("Buyback and dividend", "fundamental", True, True),
    "23": ("Index change", "trading", True, True),
    "12": ("Gap and hold", "trading", False, True),
    "16": ("Sector leader", "trading", False, True),
    "18": ("Social buzz", "social", False, True),
    "19": ("Small-outlet scoop", "news", False, True),
    "20": ("Politician trades", "ownership", False, True),
}
READY = [k for k, v in LANES.items() if v[3]]
TRIGGER = [k for k, v in LANES.items() if v[2]]

# ---------- headlines ----------
# (case-insensitive pattern, case-sensitive pattern) per lane; a headline can hit several lanes
I, S = re.IGNORECASE, 0
HEAD = {
    "1": (r"\b(q[1-4]|first[- ]quarter|second[- ]quarter|third[- ]quarter|fourth[- ]quarter|quarterly|fy ?\d{2,4}|"
          r"full[- ]year)\b.*\b(eps|earnings|results|revenue|sales|profit|loss)\b|\b(eps|earnings|revenue|sales)\b.*"
          r"\b(beats?|miss(es)?|tops?|above|below|exceeds?|falls? short|in[- ]line)\b", r"\bEPS\b"),
    "2": (r"\b(guidance|outlook|forecast)\b|\b(sees|expects) (q[1-4]|fy|full[- ]year)", None),
    "6": (r"\b(upgrades?|upgraded|downgrades?|downgraded|initiates? coverage|initiated|resumes? coverage|"
          r"price target|reiterat\w*|(raises?|lowers?|cuts?|boosts?) (its )?(price )?target)\b", r"\bPT\b"),
    "8": (r"\b(fda|pdufa|complete response letter|phase ?(1|2|3|i|ii|iii)\b|topline|top-line|pivotal|clinical (trial|hold)|"
          r"breakthrough therapy|orphan drug|fast track|510\(k\)|marketing authori[sz]ation)\b",
          r"\b(CRL|BLA|NDA|sNDA|EMA)\b"),
    "9": (r"\b(contract|awarded|wins? .{0,40}(order|deal)|purchase order|task order|idiq|selected by|"
          r"supply agreement)\b", None),
    "10": (r"\b(acquir\w*|acquisition|merger|merge|buyout|takeover|tender offer|go(ing)?[- ]private|"
           r"to be bought|strategic alternatives|explor\w* (a )?sale)\b", None),
    "21": (r"\b(sec (probe|investigation|charges|subpoena)|subpoena|wells notice|department of justice|indict\w*|"
           r"class action|lawsuit|sued|sues|recall\w*|restat\w*|material weakness|going concern|non-reliance|"
           r"auditor (resign|dismiss)\w*|delist\w*|deficiency notice|non-?compliance|default\w*|covenant|"
           r"bankruptcy|chapter 11|short (seller|report)|fraud)\b", r"\b(DOJ|FTC)\b"),
    "22": (r"\b(buyback|repurchase|dividend)\b", None),
    "23": (r"\b(s&p ?(500|400|600|midcap|smallcap)|nasdaq[- ]100|russell (1000|2000|3000))\b.*\b(join|added|add|"
           r"addition|replace|remov\w*|delet\w*|drop\w*|inclusion|rebalanc\w*)\b|\bset to join\b", None),
    "7": (r"\b(offering|registered direct|at[- ]the[- ]market|private placement|convertible (senior )?(notes|"
          r"debentures|preferred)|warrants?|shelf registration|lock-?up (expir\w*|release))\b", r"\b(ATM|PIPE)\b"),
}
RUMOUR = re.compile(r"\b(reportedly|report(s|ed)? says?|sources|people familiar|rumou?r\w*|in talks|considering|"
                    r"exploring|weighs|mulls|approach(ed)?)\b", I)


def headline_lanes(text):
    """Lanes a headline (or a filing's opening lines) hits."""
    t = " ".join(str(text).split())
    out = []
    for lane, (ci, cs) in HEAD.items():
        if (ci and re.search(ci, t, I)) or (cs and re.search(cs, t)):
            out.append(lane)
    return out


def rumoured(text):
    return bool(RUMOUR.search(str(text)))


# ---------- filings ----------
ITEMS_3 = {"1.01", "1.02", "2.01", "2.05", "2.06", "3.01", "4.01", "4.02", "5.02", "8.01"}
ITEMS_TEXT = {"1.01", "2.02", "7.01", "8.01"}      # 8-K items whose press release is read for headline lanes
SUPPLY = re.compile(r"^(S-1|S-1MEF|S-3|S-3ASR|S-3MEF|F-1|F-3|F-3ASR|424B[1-57])$")
OWN_ROLE = {"8-K": "Filer", "10-Q": "Filer", "10-K": "Filer", "DEF 14A": "Filer", "NT 10-K": "Filer",
            "NT 10-Q": "Filer", "SCHEDULE 13D": "Subject", "SCHEDULE 13D/A": "Subject", "SC TO-T": "Subject",
            "4": "Issuer"}


def filing_role(entry):
    """True when this feed entry names the company the filing is about (not a filer or owner)."""
    want = OWN_ROLE.get(entry["form"], "Filer")
    return entry["role"] == want


def filing_lanes(entry):
    """Lanes a feed entry hits before any text is read; 8-K text lanes are added by text_lanes()."""
    f, items = entry["form"], set(entry.get("items") or [])
    lanes = []
    if f == "8-K":
        if items & ITEMS_3:
            lanes.append("3")
        if items & {"4.01", "4.02", "3.01"}:
            lanes.append("21")
        if "3.02" in items:
            lanes.append("7")
        if "2.01" in items:
            lanes.append("10")
    elif f in ("10-Q", "10-K", "10-K/A", "DEF 14A"):
        lanes.append("3")
    elif f in ("NT 10-K", "NT 10-Q"):
        lanes.append("21")
    elif f == "SCHEDULE 13D":
        lanes.append("5")
    elif f == "SC TO-T":
        lanes.append("10")
    elif SUPPLY.match(f):
        lanes.append("7")
    return lanes


def needs_text(entry):
    """Filings whose text is read: 8-Ks with a press-release item, and 13D amendments (purpose change)."""
    return (entry["form"] == "8-K" and bool(set(entry.get("items") or []) & ITEMS_TEXT)) or \
        entry["form"] == "SCHEDULE 13D/A"


def text_lanes(entry, text):
    """Lanes from a filing's text: an 8-K press release's opening lines go through the headline lanes
    (lane 2 also needs item 2.02 or 7.01); a 13D/A counts only when it amends Item 4 (purpose)."""
    if entry["form"] == "SCHEDULE 13D/A":
        return ["5"] if re.search(r"item\s*4\b|purpose of (the )?transaction", text, I) else []
    head = text[:600]
    lanes = [x for x in headline_lanes(head) if x not in ("21", "7")]
    if "2" in lanes and not set(entry.get("items") or []) & {"2.02", "7.01"}:
        lanes.remove("2")
    return lanes


def insider_buy(f4, row=None):
    """Lane 4: an open-market purchase (code P) outside a 10b5-1 plan, with its flags; None otherwise."""
    if f4 is None or f4["plan_10b5_1"]:
        return None
    buys = [t for t in f4["transactions"] if t["code"] == "P" and t["ad"] == "A" and t["shares"]]
    if not buys:
        return None
    value = sum(t["shares"] * (t["price"] or 0.0) for t in buys)
    titles = " ".join(o["title"] for o in f4["owners"]).upper()
    flags = []
    if re.search(r"\b(CEO|CFO|CHIEF EXECUTIVE|CHIEF FINANCIAL)\b", titles):
        flags.append("CEO/CFO")
    if value >= 100_000:
        flags.append("$100k+")
    if row and row.get("hi252") and row["prev"] <= 0.8 * row["hi252"]:
        flags.append("after a 20% drawdown")
    return {"value": round(value, 2), "shares": sum(t["shares"] for t in buys), "owners": len(f4["owners"]),
            "flags": flags}


# ---------- numbers ----------

def pace(day_v, v20, sess, t):
    """Volume so far against a normal day at this time of day; None in the first 5% of the session."""
    o, c = sess
    f = (min(t, c) - o) / (c - o)
    if f < 0.05 or not v20:
        return None
    return day_v / (v20 * min(f, 1.0))


def number_alerts(row, live, sess, now, fee_cut, recent):
    """Lanes 11, 12, 13, 14, 17a and 17b for one stock. live: {"price", "day_v", "day_o", "t"}; recent: lanes
    seen for this stock in the last 3 sessions (with "news" when it had any headline)."""
    s, p = row["symbol"], live["price"]
    day = f"{now.tz_convert(ET):%Y-%m-%d}"
    move = p / row["prev"] - 1
    z = math.log(p / row["prev"]) / row["sigma"] if row["sigma"] > 0 else 0.0
    pc = pace(live.get("day_v") or 0.0, row["v20"], sess, min(live["t"], now))
    out = []

    def add(lane, side, **info):
        out.append({"lane": lane, "symbol": s, "item": f"{lane}:{s}:{day}", "src_t": live["t"].isoformat(),
                    "side_hint": side, "info": dict(info, move=round(move, 4), z=round(z, 2),
                                                    pace=None if pc is None else round(pc, 2))})
    fundamental = {k for k in recent if LANES.get(k, ("", ""))[1] == "fundamental"}
    if pc is not None and pc >= 3 and "news" not in recent and not fundamental and not recent & {"4", "5", "7"}:
        add("11", None)
    if pc is not None and pc >= 2:
        k = row["prev"] / row["prev_adj"] if row.get("prev_adj") else 1.0
        if row.get("hi252") and p > row["hi252"] * k:
            add("13", "long", level="52-week high")
        elif row.get("lo252") and p < row["lo252"] * k:
            add("13", "short", level="52-week low")
        elif row.get("hi20") and p > row["hi20"] * k:
            add("13", "long", level="20-day high")
        elif row.get("lo20") and p < row["lo20"] * k:
            add("13", "short", level="20-day low")
    if z <= -3 and "21" not in recent:
        add("14", "long")
    fee = row.get("fee")
    if fee is not None and fee_cut is not None and fee >= fee_cut:
        if move >= 0.05 and pc is not None and pc >= 2:
            add("17a", "long", fee=fee)
        if move <= -0.05 or "21" in recent:
            add("17b", "short", fee=fee)
    o = live.get("day_o")
    if o and abs(o / row["prev"] - 1) >= 0.04:
        hour_done = now >= sess[0] + pd.Timedelta(hours=1)
        held = (p - row["prev"]) / (o - row["prev"]) >= 0.5 if hour_done else None
        add("12", "long" if o > row["prev"] else "short", gap=round(o / row["prev"] - 1, 4), held=held)
    return out


LEADER_MIN = 3          # priced stocks a sector needs before lane 16 ranks it


def sector_leaders(rows, live, sectors, etf, sess, now):
    """Lane 16 (confirm-only), from 30 minutes after the open: today's strongest and weakest sector ETF, and in
    each of the two the stock with the largest and the smallest move today. sectors: {symbol: sector ETF};
    etf: the universe's {ETF: {"prev"}}."""
    if now < sess[0] + pd.Timedelta(minutes=30):
        return []
    day = f"{now.tz_convert(ET):%Y-%m-%d}"
    members = {}
    for r in rows:
        s, x = r["symbol"], sectors.get(r["symbol"])
        if x and s in live:
            members.setdefault(x, []).append((live[s]["price"] / r["prev"] - 1, s))
    smove = {x: live[x]["price"] / etf[x]["prev"] - 1 for x in members
             if len(members[x]) >= LEADER_MIN and x in live and etf.get(x, {}).get("prev")}
    if len(smove) < 2:
        return []
    out = []
    for x, kind in ((max(smove, key=lambda k: (smove[k], k)), "strongest"),
                    (min(smove, key=lambda k: (smove[k], k)), "weakest")):
        ranked = sorted(members[x])
        for (m, s), role, side in ((ranked[-1], "top", "long"), (ranked[0], "bottom", "short")):
            out.append({"lane": "16", "symbol": s, "item": f"16:{s}:{day}", "src_t": live[s]["t"].isoformat(),
                        "side_hint": side, "info": {"sector": x, "sector_rank": kind, "stock_rank": role,
                                                    "sector_move": round(smove[x], 4), "move": round(m, 4)}})
    return out


MOVER_MOVE, MOVER_Z = 0.08, 3.0   # lane 15: a big mover is up or down 8%+ and 3+ daily standard deviations
MOVERS, PEERS = 5, 3               # at most 5 movers a pass, each with its 3 most traded lagging peers
CATALYST = {"news", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "21", "22", "23"}


def sympathy(rows, live, sic, recent, sess, now):
    """Lane 15, from 30 minutes after the open: for today's biggest movers with a catalyst in the last 3
    sessions (a headline or a news or filing lane), the most traded stocks with the same 4-digit SIC code
    that have moved less than a third as far in the same direction. The hint follows the mover."""
    if now < sess[0] + pd.Timedelta(minutes=30):
        return []
    day = f"{now.tz_convert(ET):%Y-%m-%d}"
    move = {r["symbol"]: live[r["symbol"]]["price"] / r["prev"] - 1 for r in rows if r["symbol"] in live}
    big = []
    for r in rows:
        s = r["symbol"]
        if s not in move or not sic.get(s) or not recent.get(s, set()) & CATALYST or r["sigma"] <= 0:
            continue
        z = math.log(1 + move[s]) / r["sigma"]
        if abs(move[s]) >= MOVER_MOVE and abs(z) >= MOVER_Z:
            big.append((-abs(move[s]), s))
    out, used = [], set()
    for _, m in sorted(big)[:MOVERS]:
        d = 1 if move[m] > 0 else -1
        peers = sorted((-r["dv20"], r["symbol"]) for r in rows if r["symbol"] in move and r["symbol"] != m
                       and sic.get(r["symbol"]) == sic[m] and d * move[r["symbol"]] < abs(move[m]) / 3)
        for _, p in peers[:PEERS]:
            if p in used:
                continue
            used.add(p)
            out.append({"lane": "15", "symbol": p, "item": f"15:{p}:{day}", "src_t": live[p]["t"].isoformat(),
                        "side_hint": "long" if d > 0 else "short",
                        "info": {"mover": m, "mover_move": round(move[m], 4), "sic": sic[m],
                                 "move": round(move[p], 4)}})
    return out


SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|"
                    r"n\.?v|s\.?a|ag|se|lp|llc|trust|bancorp|class [a-c]|common stock|/[a-z]{2})\.?$", I)


def short_name(name):
    """A company's SEC name without its legal suffixes ("Apple Inc." -> "Apple")."""
    n = " ".join(str(name).replace("&amp;", "&").split())
    while True:
        m = SUFFIX.search(n)
        if not m or m.start() == 0:
            return n.strip(" ,.")
        n = n[:m.start()]


def name_index(names, symbols):
    """{short name: symbol} for the universe. A name counts only when it is at least 2 words or 6 letters and
    no other universe company shares it, so common words can't tag a stock."""
    seen = {}
    for s in sorted(symbols):
        if s in names:
            n = short_name(names[s])
            if len(n.split()) >= 2 or len(n) >= 6:
                seen.setdefault(n, set()).add(s)
    return {n: next(iter(v)) for n, v in seen.items() if len(v) == 1}


def name_hits(title, index, tickers):
    """Universe symbols an outlet's headline names: a whole-word, case-sensitive company name, or a ticker
    written as $XYZ or (NASDAQ: XYZ) / (NYSE: XYZ)."""
    out = {index[n] for n in index if re.search(r"(?<![\w&])" + re.escape(n) + r"(?![\w&])", title)}
    out |= {m for m in re.findall(r"\$([A-Z]{1,5})\b", title) if m in tickers}
    out |= {m for m in re.findall(r"\((?:NASDAQ|NYSE|NYSE American|AMEX|Nasdaq):\s*([A-Z.]{1,6})\)", title) if m in tickers}
    return sorted(out)


BUZZ_MIN, BUZZ_JUMP = 20, 3.0      # lane 18: 20+ Reddit mentions in 24 hours and 3x the day before
BUZZ_USERS = 10                    # distinct StockTwits accounts in 24 hours (one account can't make buzz)


def buzz_candidates(ape, trending, symbols):
    """Lane 18's stocks to check: Reddit mention jumps (ApeWisdom's 24-hour counts) and StockTwits trending,
    universe stocks only. {symbol: {"mentions", "mentions_before", "trending"}}."""
    out = {}
    for r in ape:
        s, m, b = r.get("ticker"), int(r.get("mentions") or 0), int(r.get("mentions_24h_ago") or 0)
        if s in symbols and m >= BUZZ_MIN and m >= BUZZ_JUMP * max(b, 1):
            out[s] = {"mentions": m, "mentions_before": b, "trending": False}
    for s in trending:
        if s in symbols:
            out.setdefault(s, {"mentions": None, "mentions_before": None, "trending": False})["trending"] = True
    return out


def buzz_alert(s, cand, msgs, now):
    """Lane 18 (confirm-only) alert for one candidate when its StockTwits stream has BUZZ_USERS or more
    distinct accounts in the last 24 hours, else None. msgs: [{"user", "t", "sentiment"}]."""
    day = now - pd.Timedelta(hours=24)
    recent = [m for m in msgs if m.get("t") and pd.Timestamp(m["t"]) >= day]
    users = {m["user"] for m in recent}
    if len(users) < BUZZ_USERS:
        return None
    bull = sum(m.get("sentiment") == "Bullish" for m in recent)
    bear = sum(m.get("sentiment") == "Bearish" for m in recent)
    return {"lane": "18", "symbol": s, "item": f"18:{s}:{now.tz_convert(ET):%Y-%m-%d}", "src_t": now.isoformat(),
            "side_hint": None, "info": dict(cand, users=len(users), messages=len(recent), bullish=bull, bearish=bear)}


# One stock row of a House periodic transaction report, as the PDF's text reads once whitespace is joined:
# "... (AZO) [ST] S 09/22/2026 10/14/2026 $15,001 - $50,000"
PTR_ROW = re.compile(r"\(([A-Z][A-Z0-9.]{0,6})\)\s*\[ST\]\s*(P|S \(partial\)|S)\s+(\d\d/\d\d/\d{4})\s+"
                     r"(\d\d/\d\d/\d{4})\s+(\$[\d,]+(?:\s*-\s*\$[\d,]+)?|Over \$[\d,]+)")


def ptr_trades(text):
    """Stock purchases and sales in one House PTR's text: [{"symbol", "type", "traded", "notified", "amount"}]."""
    out = []
    for m in PTR_ROW.finditer(" ".join((text or "").split())):
        out.append({"symbol": m.group(1), "type": "P" if m.group(2) == "P" else "S",
                    "traded": m.group(3), "notified": m.group(4), "amount": " ".join(m.group(5).split())})
    return out


def politician_alerts(report, trades, symbols, now):
    """Lane 20 (confirm-only): one alert per traded stock in the universe; a buy hints long, a sale short."""
    out = []
    for tr in trades:
        if tr["symbol"] not in symbols:
            continue
        out.append({"lane": "20", "symbol": tr["symbol"], "item": f"20:{report['doc_id']}:{tr['symbol']}",
                    "src_t": now.isoformat(), "side_hint": "long" if tr["type"] == "P" else "short",
                    "info": {"member": report["name"], "district": report["state_district"], "type": tr["type"],
                             "traded": tr["traded"], "filed": report["filed"], "amount": tr["amount"]}})
    return out


def fee_cutoff(fees):
    """The top-decile borrow fee among universe stocks with a fee, or None."""
    f = pd.Series([x for x in fees if x is not None and x == x], dtype=float)
    return float(f.quantile(0.9)) if len(f) >= 20 else None


# ---------- shared checks ----------

SIP_DELAY = pd.Timedelta(minutes=15)


def check(row, q, now, sess, bad_actions):
    """Reason a candidate fails the shared checks at this pass, or None. q: {"t" (latest trade, any feed),
    "iex_t" (latest IEX trade), "sip_t" (latest delayed-SIP trade), "bid", "ask" (NBBO, delayed)}."""
    if row["symbol"] in bad_actions:
        return bad_actions[row["symbol"]]
    # Age of the last trade on any feed: IEX is live but thin; delayed SIP sees every trade 15 minutes late.
    ages = [now - q["iex_t"]] if q.get("iex_t") is not None else []
    if q.get("sip_t") is not None:
        ages.append(now - q["sip_t"] - SIP_DELAY)
    elif q.get("t") is not None and not ages:
        ages.append(now - q["t"])
    age = min(ages) if ages else None
    if age is not None and now - sess[0] >= pd.Timedelta(minutes=30) and age > pd.Timedelta(minutes=30):
        return "no trade in 30 minutes (halted?)"
    if age is None or age > pd.Timedelta(minutes=15):
        return "no quote in 15 minutes"
    bid, ask = q.get("bid") or 0.0, q.get("ask") or 0.0
    if bid > 0 and ask > bid:
        spread = (ask - bid) / ((ask + bid) / 2)
        if spread > (0.02 if row["prev"] >= 5 else 0.05):
            return f"spread {spread:.1%}"
    return None


# ---------- events ----------

def tdays_between(a, b, tdays):
    """Trading days from session date a to session date b (b after a)."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return sum(1 for d in tdays if a < d <= b)


def merge(events, alerts, now, tdays, prices):
    """Fold alerts into events in place; returns the events created or extended this pass. Two alerts share
    an event when they share a stock, fall within 2 trading days, and share a source item or are both
    fundamental."""
    day = now.tz_convert(ET).normalize().tz_localize(None)
    touched = {}
    for a in alerts:
        group = LANES[a["lane"]][1]
        near = [e for e in events.values() if e["symbol"] == a["symbol"] and tdays_between(e["day"], day, tdays) <= 2]
        match = [e for e in near if a["item"] in e["items"] or (group == "fundamental" and any(
            LANES[x][1] == "fundamental" for x in e["lanes"]))]
        hit = min(match, key=lambda e: e["detected"]) if match else None
        for e in match:   # the alert links events: they become one, kept under the earliest
            if e is not hit:
                absorb(hit, events.pop(e["id"]))
                touched.pop(e["id"], None)
        if hit is None:
            eid = f"{a['symbol']}-{now:%Y%m%d%H%M}-{len(events)}"
            p = prices.get(a["symbol"])
            hit = events[eid] = {"id": eid, "symbol": a["symbol"], "day": f"{day:%Y-%m-%d}", "lanes": [],
                                 "items": [], "src_t": a["src_t"], "detected": now.isoformat(),
                                 "first_price": p, "owner": None, "shown": None, "alerts": []}
        if a["item"] not in hit["items"]:
            hit["items"].append(a["item"])
        if a["lane"] not in hit["lanes"]:
            hit["lanes"].append(a["lane"])
            hit["alerts"].append({k: a[k] for k in ("lane", "item", "src_t", "side_hint", "info")})
        if hit["owner"] is None and LANES[a["lane"]][2]:
            hit["owner"] = a["lane"]
        if a["src_t"] and (not hit["src_t"] or a["src_t"] < hit["src_t"]):
            hit["src_t"] = a["src_t"]
        if hit["first_price"] is None and prices.get(a["symbol"]):
            hit["first_price"] = prices[a["symbol"]]
        touched[hit["id"]] = hit
    return list(touched.values())


def absorb(e, other):
    for k in ("items", "lanes"):
        e[k] += [x for x in other[k] if x not in e[k]]
    e["alerts"] += [x for x in other["alerts"] if x["lane"] not in {y["lane"] for y in e["alerts"]}]
    if e["owner"] is None:
        e["owner"] = other["owner"]
    if other["src_t"] and (not e["src_t"] or other["src_t"] < e["src_t"]):
        e["src_t"] = other["src_t"]
    if other.get("shown") and not e.get("shown"):
        e["shown"] = other["shown"]


def prune(events, day, tdays, keep=12):
    """Drop events older than `keep` trading days (they are already in the pass logs)."""
    for k in [k for k, e in events.items() if tdays_between(e["day"], day, tdays) > keep]:
        events.pop(k)
