"""Builds the dashboard's data file (summary.json) from the forward logs the jobs already write. Read-only: it
places no orders and changes no book. Run by .github/workflows/dashboard.yml, which clones each state branch into
<states>/<branch> and saves the output to the `state-dash` branch, where the page reads it.

- books: every forward book with closed trades against its target, P&L and open positions;
- accounts: each Alpaca paper account's equity and P&L % (read-only API calls, keys from the environment);
- jobs: the latest run of every scheduled workflow;
- nx3: the last few NX3 days, every decided headline with its logged inputs (the page replays them through the
  trees), its decision, price at decision, the settled result, the control, and for skipped stocks the next-day
  move; today's symbols also get a latest price;
- ideas: every row of research/IDEAS.md with a status and the date anything about it last changed, flagged
  "stalled" after 7 days with no change and no live book.

No headline text is read or written. Usage: python -m dashboard.build --states st --out dash
"""
import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from adaptive import accounts as A
from core.alpaca import TRADE

DATA = "https://data.alpaca.markets/v2"
NX3_DAYS, STALE_DAYS = 7, 7
NX3_COST = 0.002          # NX3's round trip (config/risk.yaml news track), for the skipped stocks' moves
SLIM = ["symbol", "t", "decided_at", "e", "fired", "side", "skip", "entry_iex", "prev_c", "dv20", "etb", "inputs"]

# Forward books: where each one's record lives and what it is judged at (PROTOCOL / the idea's prereg).
NX3_ARMS = ["LGBM", "LIN", "RF", "MLP", "ENS", "XGBR", "RFR"]
WATCH = [("PICK-MAIN", "SYS2", "state-watch", "watch/books.json", 30), ("PICK-PENNY", "SYS2", "state-watch",
         "watch/books.json", 30), ("SCOUT-MAIN", "SYS2B", "state-scouts", "scouts/books.json", 30),
         ("SCOUT-PENNY", "SYS2B", "state-scouts", "scouts/books.json", 30)]
AI = [("AI-B", "K2", "state-ai", "ai/books.json", "B"), ("COMBO-B", "K6", "state-ai", "combo/books.json", "B"),
      ("WIDE-C", "K4", "state-wide", "wide/books.json", "C"),
      ("WIDE-C-groq", "K7", "state-wide", "wide/books.json", "C-groq")]
LIVE_IDEAS = {"NX3", "NX1", "P1", "P2", "P4", "M3", "M4", "M6", "M8", "M10", "P3", "R1", "I1F", "A1F", "META3",
              "SYS1", "SYS2", "SYS2B", "K3", "L1", "NXM", "NXR"}


def _json(p):
    try:
        return json.loads(Path(p).read_text().replace("NaN", "null"))
    except (OSError, ValueError):
        return None


def _jsonl(p):
    try:
        return [json.loads(x.replace("NaN", "null")) for x in Path(p).read_text().splitlines() if x.strip()]
    except OSError:
        return []


def _clean(x):
    """NaN and inf to None so the output is valid JSON for the browser."""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _book(name, idea, target, closed, open_, realized=None, unreal=None, size=None, unit="trades", **extra):
    pnl = (realized or 0.0) + (unreal or 0.0) if realized is not None or unreal is not None else None
    return {"name": name, "idea": idea, "target": target, "unit": unit, "closed": closed, "open": open_,
            "realized": realized, "unrealized": unreal, "size": size,
            "pnl_pct": 100 * pnl / size if pnl is not None and size else None, **extra}


# ---------- books ----------

def nx3_books(st):
    root = st / "state-nx3" / "nx3"
    out = []
    settled = [_json(f) for f in sorted((root / "settled").glob("*.json"))]
    counted = [s for s in settled if s and not s.get("excluded")]
    picks = [t for s in counted for t in s.get("trades", []) if t.get("role") == "picks"]
    days = sorted(f.stem for f in (root / "days").glob("*.json"))
    open_n = 0
    if days and (not settled or days[-1] > max(s["day"] for s in settled if s)):
        open_n = len((_json(root / "days" / f"{days[-1]}.json") or {}).get("picks", []))
    out.append(_book("NX3", "NX3", 100, len(picks), open_n, realized=sum(500 * (t["net"] or 0) for t in picks),
                     size=5000.0, account="EL-NX3", first_day="2026-10-29",
                     excluded_days=[s["day"] for s in settled if s and s.get("excluded")]))
    msettled = [_json(f) for f in sorted((root / "nxm" / "settled").glob("*.json"))]
    for a in NX3_ARMS:
        tr = [t for s in msettled if s for t in s.get("arms", {}).get(a, {}).get("trades", []) if t.get("role") == "picks"]
        idea = "NXR" if a in ("XGBR", "RFR") else "NXM"
        out.append(_book(f"{idea}-{a}", idea, 100, len(tr), None, realized=sum(500 * (t["net"] or 0) for t in tr),
                         size=5000.0, first_day="2026-10-30" if idea == "NXR" else "2026-10-29"))
    return out


def nx1_book(st):
    b = _json(st / "state-nx1" / "nx1" / "books.json") or {}
    led = _jsonl(st / "state-nx1" / "nx1" / "ledger.jsonl")
    p = b.get("picks", {})
    cl = p.get("closed", [])
    return [_book("NX1", "NX1", 60, len({x["date"] for x in led if x.get("kind") == "nx1_night"}), len(p.get("pos", [])),
                  realized=sum(x.get("pnl") or 0 for x in cl), size=5000.0, unit="trading days",
                  trades_closed=len(cl))]


def paper_books(st):
    days = sorted((st / "state" / "paper" / "days").glob("*.json"))
    b = (_json(days[-1]) or {}).get("book", {}) if days else {}
    out = []
    for idea in ("P1", "P2"):
        cl = [x for x in b.get("closed", []) if x.get("idea") == idea]
        op = [x for x in b.get("open", []) if x.get("idea") == idea]
        un = sum((x.get("notional") or 0) * ((x["last_price"] / x["entry_price"] - 1) * (1 if x.get("side") == "long" else -1))
                 for x in op if x.get("last_price") and x.get("entry_price"))
        out.append(_book(idea, idea, 100, len(cl), len(op), realized=sum(x.get("pnl") or 0 for x in cl), unreal=un,
                         size=None, account="EL-P1P2", checks="early checks at 50 and 75 trades",
                         positions=[{"symbol": x["symbol"], "side": x.get("side"), "entry": x.get("entry_price"),
                                     "last": x.get("last_price"), "stop": x.get("stop_price")} for x in op]))
    return out


def watch_books(st):
    out = []
    for name, idea, br, path, target in WATCH:
        j = _json(st / br / path) or {}
        for nm in (name, name + "-coin"):
            b = j.get("books", {}).get(nm)
            if b is None:
                continue
            pos = b.get("pos", {})
            un = sum(p.get("side", 1) * p["qty"] * (p["last"] - p["entry"]) for p in pos.values() if p.get("last"))
            out.append(_book(nm, idea, target, len(b.get("trips", [])), len(pos),
                             realized=sum(t.get("net") or 0 for t in b.get("trips", [])), unreal=un,
                             size=b.get("start"), control=nm.endswith("-coin"),
                             note="also needs 4 weeks and to beat its frozen copy"))
    return out


def ai_books(st):
    out = []
    for name, idea, br, path, key in AI:
        b = (_json(st / br / path) or {}).get("books", {}).get(key)
        if b is None:
            continue
        out.append(_book(name, idea, None, len(b.get("trips") or []), len(b.get("pos") or {}),
                         realized=sum(t.get("pnl") or 0 for t in b.get("trips") or []), size=None,
                         note="AI trader thread's book"))
    return out


def log_books(st):
    """Books that log picks on a schedule (weekly or monthly) and are judged later from those logs."""
    out = []
    led = _jsonl(st / "state" / "longterm" / "ledger.jsonl")
    for kind, idea in (("M3_picks", "M3"), ("M4_picks", "M4"), ("M8_picks", "M6"), ("R1_picks", "R1"),
                       ("I1F_picks", "I1F"), ("A1F_picks", "A1F"), ("META3_picks", "META3"),
                       ("M10_watch", "M10"), ("P3_watch", "P3")):
        n = sum(1 for x in led if x.get("kind") == kind)
        last = max((x["date"] for x in led if x.get("kind") == kind), default=None)
        out.append(_book(idea, idea, None, n, None, unit="logged rebalances", last_log=last))
    p4 = _jsonl(st / "state" / "p4" / "ledger.jsonl")
    sig = sum(len((_json(st / "state" / "p4" / x["file"]) or {}).get("signals", [])) for x in p4 if x.get("file"))
    out.append(_book("P4", "P4", None, sig, None, unit="signals", account="EL-P4"))
    k = _jsonl(st / "state-kronos" / "kronos" / "ledger.jsonl")
    out.append(_book("K3", "K3", None, len(k), None, unit="weekly logs"))
    return out


def adaptive_status(st):
    led = _jsonl(st / "state" / "adaptive" / "ledger.jsonl")
    if not led:
        return None
    last = _json(st / "state" / "adaptive" / led[-1]["file"]) or {}
    return {"last_night": led[-1]["date"], "send": last.get("send"),
            "skipped": {k: sum(1 for o in v.get("orders", []) if o.get("kind") == "skip")
                        for k, v in last.get("strategies", {}).items()}}


# ---------- accounts and jobs ----------

def account_rows(get=requests.get):
    rows = []
    for name in A.ACCOUNTS:
        if name == "EL-DATA":
            continue
        h = A.headers(name)
        if h is None:
            rows.append({"name": name, "status": "secrets not set"})
            continue
        try:
            a = get(f"{TRADE}/account", headers=h, timeout=30).json()
            ph = get(f"{TRADE}/account/portfolio/history", params={"period": "1A", "timeframe": "1D"},
                     headers=h, timeout=30).json()
            pos = get(f"{TRADE}/positions", headers=h, timeout=30).json()
            eq = [x for x in (ph.get("equity") or []) if x]
            base = ph.get("base_value") or (eq[0] if eq else None)
            rows.append({"name": name, "status": a.get("status"), "equity": float(a.get("equity", 0)),
                         "last_equity": float(a.get("last_equity", 0)), "base": base,
                         "pnl_pct": 100 * (float(a["equity"]) / base - 1) if base else None,
                         "positions": [{"symbol": p["symbol"], "qty": p["qty"], "side": p.get("side"),
                                        "entry": float(p["avg_entry_price"]), "last": float(p["current_price"]),
                                        "pl_pct": 100 * float(p["unrealized_plpc"])} for p in pos]})
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            rows.append({"name": name, "status": f"no answer ({type(e).__name__})"})
    return rows


def job_rows(repo, token, get=requests.get):
    if not token:
        return []
    try:
        r = get(f"https://api.github.com/repos/{repo}/actions/runs", params={"per_page": 100},
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=30)
        runs = r.json().get("workflow_runs", [])
    except (requests.RequestException, ValueError):
        return []
    seen, out = set(), []
    for x in runs:
        if x["name"] in seen or x.get("event") not in ("schedule", "workflow_dispatch"):
            continue
        seen.add(x["name"])
        out.append({"name": x["name"], "status": x["status"], "conclusion": x.get("conclusion"),
                    "at": x["created_at"], "event": x["event"]})
    return sorted(out, key=lambda x: x["name"])


# ---------- NX3 days ----------

def alpaca_get(path, params):
    h = A.headers("EL-DATA")
    if h is None:
        return None
    try:
        r = requests.get(f"{DATA}{path}", params=params, headers=h, timeout=30)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def daily_closes(symbols, start, end, adjustment):
    """{symbol: {date: close}} from SIP daily bars with the adjustment given."""
    out, token = {}, None
    while symbols:
        p = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": end, "feed": "sip",
             "adjustment": adjustment, "limit": 10000, **({"page_token": token} if token else {})}
        j = alpaca_get("/stocks/bars", p)
        if not j:
            break
        for s, bars in (j.get("bars") or {}).items():
            out.setdefault(s, {}).update({b["t"][:10]: b["c"] for b in bars})
        token = j.get("next_page_token")
        if not token:
            break
    return out


def skipped_moves(day, nxt, rows):
    """Next-close move of each decided stock from its price at decision, in the trade's direction if one was
    taken, else long: raw close on the day times the all-adjusted change to the next close (a split overnight
    stays right)."""
    syms = sorted({r["symbol"] for r in rows if r.get("entry_iex")})
    if not syms:
        return {}
    raw = daily_closes(syms, day, nxt, "raw")
    adj = daily_closes(syms, day, nxt, "all")
    out = {}
    for r in rows:
        s = r["symbol"]
        try:
            c0r, c0a, c1a = raw[s][day], adj[s][day], adj[s][nxt]
        except KeyError:
            continue
        out[s] = c0r * (c1a / c0a) / r["entry_iex"] - 1
    return out


def latest_prices(symbols):
    if not symbols:
        return {}
    out, syms = {}, sorted(symbols)
    for i in range(0, len(syms), 200):
        j = alpaca_get("/stocks/trades/latest", {"symbols": ",".join(syms[i:i + 200]), "feed": "iex"}) or {}
        out.update({s: t.get("p") for s, t in (j.get("trades") or {}).items()})
    return out


def nx3_days(st, cache):
    """The last NX3_DAYS sessions (decisions folder per day), newest first. A settled day's moves are computed
    once and kept in <cache>/nx3_moves_<day>.json."""
    root = st / "state-nx3" / "nx3"
    dirs = sorted([p for p in (root / "days").glob("*") if p.is_dir()])[-NX3_DAYS:]
    out = []
    for i, d in enumerate(dirs):
        day = d.name
        rows = [r for r in _jsonl(d / "decisions.jsonl") if r.get("status") == "decided"]
        allrows = _jsonl(d / "decisions.jsonl")
        settled = _json(root / "settled" / f"{day}.json")
        rec = _json(root / "days" / f"{day}.json") or {}
        res = {}
        if settled:
            for t in settled.get("trades", []):
                res[(t["symbol"], t["role"])] = {"net": t.get("net"), "entry": t.get("entry"), "exit": t.get("exit_ac"),
                                                 "ret": t.get("ret")}
        moves = None
        if settled:
            f = Path(cache) / f"nx3_moves_{day}.json"
            moves = _json(f)
            if moves is None:
                nxt = settled_next(dirs, i, settled)
                moves = skipped_moves(day, nxt, rows) if nxt else {}
                if moves:
                    Path(cache).mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(_clean(moves)))
        ctrl = {c["symbol"]: c.get("side") for c in rec.get("control", [])}
        ev = []
        for r in rows:
            x = {k: r.get(k) for k in SLIM}
            x["result"] = res.get((r["symbol"], "picks"))
            x["next_move"] = None if moves is None else moves.get(r["symbol"])
            ev.append(x)
        out.append({"day": day, "settled": settled is not None, "excluded": (settled or {}).get("excluded"),
                    "counts": {"seen": len(allrows), "decided": len(rows),
                               "missed": sum(1 for r in allrows if r.get("status") == "missed"),
                               "fired": sum(1 for r in rows if r.get("fired")),
                               "picks": sum(1 for r in rows if r.get("side"))},
                    "pick_minus_control": (settled or {}).get("pick_minus_control"),
                    "control": [{"symbol": s, "side": sd, "result": res.get((s, "control"))} for s, sd in ctrl.items()],
                    "events": ev})
    # every stock on every day shown gets its latest price, next to its price at decision
    px = latest_prices({e["symbol"] for d in out for e in d["events"]})
    for d in out:
        for e in d["events"]:
            e["last_price"] = px.get(e["symbol"])
    return out[::-1]


def settled_next(dirs, i, settled):
    """The exit session of a settled day: from its trades (entry_t's next close) or the next logged day."""
    for t in settled.get("trades", []):
        if t.get("days") is not None and t.get("t"):
            return (datetime.fromisoformat(t["t"][:10]) + timedelta(days=int(t["days"]) - 1)).strftime("%Y-%m-%d")
    return dirs[i + 1].name if i + 1 < len(dirs) else None


# ---------- ideas ----------

def ideas(repo_root, books):
    live = {b["idea"] for b in books}
    out = []
    rows = [l for l in (Path(repo_root) / "research" / "IDEAS.md").read_text().splitlines() if l.startswith("| ")]
    now = datetime.now(timezone.utc).date()
    for l in rows:
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if c[0] == "ID" or set(c[0]) <= set("-"):
            continue
        idea_id, outcome = c[0], c[-1]
        prereg = re.findall(r"prereg/([\w-]+)\.md", l)
        ids = [i.strip() for i in idea_id.split(",")]
        base = re.sub(r"-.*", "", ids[0])
        last = last_change(repo_root, ids + [base] + prereg)
        v = [(m.start(), "passed") for m in re.finditer(r"PASS", outcome)] + \
            [(m.start(), "failed") for m in re.finditer(r"FAILED|STOPPED", outcome)]
        if v:
            status = max(v)[1]
        elif not outcome or outcome.startswith("Registered"):
            status = "open"
        else:
            status = "done"     # a finished check with no pass rule (V1-V5) or a stopped book
        is_live = bool({base, *ids} & (LIVE_IDEAS | live))
        if is_live and status != "failed":
            status = "live"
        elif is_live and status == "failed" and base in LIVE_IDEAS:
            status = "live"     # a near-miss or forward book running after a failed screen (PROTOCOL 19)
        age = (now - datetime.fromisoformat(last).date()).days if last else None
        out.append({"id": idea_id, "track": c[1] if len(c) == 7 else "validation", "registered": c[2] if len(c) == 7 else c[1],
                    "idea": c[3] if len(c) == 7 else c[2], "status": status, "outcome": outcome[:400],
                    "last_change": last, "stalled": status == "open" and age is not None and age > STALE_DAYS,
                    "days_since": age})
    return out


def last_change(repo_root, names):
    paths = []
    for n in names:
        for p in (f"research/prereg/{n}.md", f"research/results/{n}"):
            if (Path(repo_root) / p).exists():
                paths.append(p)
    if not paths:
        return None
    r = subprocess.run(["git", "-C", str(repo_root), "log", "-1", "--format=%cs", "--", *paths],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def build(states, out, repo_root=".", repo=None, token=None):
    st = Path(states)
    books = nx3_books(st) + paper_books(st) + nx1_book(st) + watch_books(st) + log_books(st) + ai_books(st)
    res = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "books": books,
           "adaptive": adaptive_status(st), "accounts": account_rows(), "jobs": job_rows(repo, token),
           "nx3": nx3_days(st, Path(out) / "cache"), "ideas": ideas(repo_root, books)}
    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = Path(out) / "summary.json.tmp"
    tmp.write_text(json.dumps(_clean(res), separators=(",", ":"), default=str))
    tmp.replace(Path(out) / "summary.json")
    print(f"books {len(books)}, ideas {len(res['ideas'])}, nx3 days {len(res['nx3'])}, accounts {len(res['accounts'])}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.states, a.out, ".", os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GH_TOKEN"))


if __name__ == "__main__":
    main()
