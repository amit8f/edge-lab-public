"""M4 nightly job: today's ranking, context for each pick, the forward log and the dashboard.

Usage: python -m tracks.main.longterm.live --state <dir> --cache <dir>
Writes <state>/longterm/latest.json and <state>/longterm/dashboard.html. On the last trading day of a
month it also logs the 20 picks, a 20-stock random control and the universe to the hash-chained ledger
(<state>/longterm/ledger.jsonl): that log is M4's forward (confirmatory) test.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import alpaca
from core.ledger import Ledger
from tracks.main import r1_live
from tracks.main.longterm import analyst, bars_batch, explain, news_history, scorecard, sec_data
from tracks.main.longterm.prices import align
from tracks.main.longterm.render import render
from tracks.main.longterm.score import FAMILIES, FAMILY_NAMES, TOP_N, attach_fundamentals, composite, eligible
from tracks.main.shortterm import live_boom

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = ROOT / "research" / "results" / "M4" / "M4_calibration.json"
SCREEN = ROOT / "research" / "results" / "M4" / "M4_screen.json"
RESULTS_DIR = ROOT / "research" / "results"
PLAN_FILE = ROOT / "research" / "plans" / "latest.json"


def _plan():
    """The family buy/sell plan (opinions, not a tested result), or None."""
    return json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else None
SEED = 20261012
LOOKBACK_DAYS = 420
TRACK_DAYS = 252
LOGGED_KINDS = ("M4_picks", "M8_picks", "M10_watch", "P3_watch", "R1_picks")
BOOM_TRACK_DAYS = 30                 # calendar days, about the 21 trading days the boom models predict
M8_WINDOW_DAYS = 91


def news_window(t, fetch=None):
    """Every Benzinga headline created in the 91 days up to t's close (New York)."""
    end = (t + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    start = end - pd.Timedelta(days=M8_WINDOW_DAYS)
    return news_history.fetch_window(start, end, **({"fetch": fetch} if fetch else {}))


def m8_ranking(uni, t, fetch=None, rows=None):
    """M8 (research/prereg/M8.md): M6's analyst-action score on the live universe, from every Benzinga
    headline created in the 91 days up to t's close."""
    rows = news_window(t, fetch) if rows is None else rows
    rows = rows[~rows["symbols"].fillna("").str.contains(",") & (rows["symbols"].fillna("") != "")]
    ev = analyst.events(rows)
    return analyst.score_m6(uni, ev, SEED), len(rows), len(ev)


def last_session(now_utc):
    cal = alpaca.trading_calendar((now_utc - pd.Timedelta(days=30)).date(), now_utc.date())
    et = now_utc.tz_convert("America/New_York")
    done = cal[cal["date"] <= pd.Timestamp(et.date())]
    if pd.Timestamp(et.date()) in set(done["date"]) and et.hour < 17:
        done = done[done["date"] < pd.Timestamp(et.date())]
    t = done["date"].max()
    nxt = alpaca.trading_calendar(t.date(), (t + pd.Timedelta(days=10)).date())
    later = nxt[nxt["date"] > t]["date"]
    month_end = later.empty or later.iloc[0].month != t.month
    return t, bool(month_end)


def week_end(t):
    """True when the next trading day after t falls in a later week."""
    nxt = alpaca.trading_calendar(t.date(), (t + pd.Timedelta(days=10)).date())
    later = nxt[nxt["date"] > t]["date"]
    return later.empty or later.iloc[0].to_period("W-SUN") != t.to_period("W-SUN")


def news_for(symbol, since, limit=6):
    try:
        j = alpaca.get(alpaca.NEWS_URL, {"symbols": symbol, "start": since, "limit": limit, "sort": "desc",
                                         "include_content": "false"}, tries=3)
    except Exception as e:  # context only; a failed lookup never stops the ranking
        print(f"[news] {symbol}: {type(e).__name__}")
        return None
    return [{"time": a.get("created_at"), "headline": a.get("headline"), "source": a.get("source"),
             "url": a.get("url")} for a in j.get("news") or []]


def reddit_mentions(symbol):
    """Posts in the last month whose title names $SYM or the bare ticker, from Reddit's public search.
    Context only; returns None when Reddit refuses the request."""
    try:
        r = requests.get("https://www.reddit.com/search.json",
                         params={"q": f'"${symbol}" OR "{symbol} stock"', "sort": "new", "t": "month", "limit": 100},
                         headers={"User-Agent": "edge-lab-research/0.1"}, timeout=30)
        if r.status_code != 200:
            return None
        posts = [c["data"] for c in r.json().get("data", {}).get("children", [])]
    except Exception:
        return None
    rx = re.compile(rf"(?:\${symbol}\b|\b{symbol}\b)")
    hits = [p for p in posts if rx.search(p.get("title", ""))]
    subs = pd.Series([p.get("subreddit") for p in hits]).value_counts().head(3).to_dict()
    top = sorted(hits, key=lambda p: p.get("score", 0), reverse=True)[:3]
    return {"posts_30d": len(hits), "subreddits": subs,
            "top": [{"title": p.get("title"), "score": p.get("score"),
                     "url": "https://www.reddit.com" + p.get("permalink", "")} for p in top]}


def _ret(fr, sym, entry_day, last_day):
    """All-adjusted return from the first open after entry_day to the last close on or before last_day."""
    d = fr.get(sym)
    if d is None:
        return None
    after = d[d.index > entry_day]
    if after.empty:
        return None
    upto = after[after.index <= last_day]
    if upto.empty:
        return None
    return float(upto["c"].iloc[-1] / after["o"].iloc[0] - 1)


def track_record(ledger, closes, today, kind="M4_picks", cap_days=int(TRACK_DAYS * 365 / 252)):
    """Forward results of every logged date of `kind`: picks, random control, universe and SPY, from
    the next open to today's close (capped at `cap_days` calendar days, 12 months by default)."""
    out = []
    for e in ledger.entries():
        if e["kind"] != kind:
            continue
        rec = json.loads((ledger.root / e["file"]).read_text())
        t = pd.Timestamp(e["date"])
        horizon_end = min(today, t + pd.Timedelta(days=cap_days))
        row = {"date": e["date"], "days": int((horizon_end - t).days)}
        for k, syms in (("picks", [p["symbol"] for p in rec["picks"]]), ("control", rec["control"]),
                        ("universe", rec["universe"])):
            rs = [_ret(closes, s, t, horizon_end) for s in syms]
            rs = [r for r in rs if r is not None]
            row[k] = float(np.mean(rs)) if rs else None
            row[f"{k}_n"] = len(rs)
        row["spy"] = _ret(closes, "SPY", t, horizon_end)
        out.append(row)
    return out


def run(state, cache, now=None, log_days_only=False):
    now = now if now is not None else alpaca.utcnow()
    state, cache = Path(state), Path(cache)
    root = state / "longterm"
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root)
    t, is_month_end = last_session(now)
    is_week_end = week_end(t)
    if log_days_only and not (is_month_end or is_week_end):
        print(f"{t:%Y-%m-%d} is not a logging day (week or month end); the dashboard refreshes on demand")
        return None
    start = (t - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = alpaca.sip_end(t, now)
    print(f"as of {t:%Y-%m-%d} (month end: {is_month_end})")

    assets = bars_batch.assets_all()
    common = assets[(assets["excl_reason"].isna()) & (assets["status"] == "active")]
    names = dict(zip(common["base_symbol"], common["name"]))
    tmap = sec_data.current_ticker_map()
    syms = sorted(set(common["base_symbol"]) & set(tmap["symbol"]))
    logged = set()
    for e in ledger.entries():
        if e["kind"] in LOGGED_KINDS:
            rec = json.loads((root / e["file"]).read_text())
            logged |= {p["symbol"] for p in rec["picks"]} | set(rec["control"]) | set(rec["universe"])
    boom_rows = []
    feat, rtab, closes = bars_batch.run_live(sorted(set(syms) | logged | {"SPY"}), start, end, t, boom=boom_rows)
    feat = feat.merge(tmap[["symbol", "cik"]], on="symbol", how="inner")
    zp = cache / "companyfacts.zip"
    sec_data.download_companyfacts(zp)
    facts = sec_data.facts_for(zp, feat["cik"].unique(), cache / "facts_live")
    uni = eligible(attach_fundamentals(feat, facts, rtab))
    scored = composite(uni)
    print(f"universe {len(scored)} stocks")
    if scored.empty:
        raise RuntimeError("no stock passed the universe filters; check the bar and SEC downloads")

    cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else None
    screen = json.loads(SCREEN.read_text()) if SCREEN.exists() else None
    top = scored.nsmallest(TOP_N, "rank")
    since = (t - pd.Timedelta(days=21)).strftime("%Y-%m-%dT00:00:00Z")
    picks = []
    for _, r in top.iterrows():
        row = r.to_dict()
        good, bad = explain.reasons(row)
        exp = explain.expectation(cal, row["score_pct"], row["price"], row["vol"])
        sym = row["symbol"]
        hist = closes.get(sym)
        spark = [] if hist is None else [round(float(x), 2) for x in hist["c"].iloc[-252:].iloc[::5]]
        picks.append({
            "rank": int(row["rank"]), "symbol": sym, "name": names.get(sym) or "",
            "price": float(row["price"]), "mcap": float(row["mcap"]), "score": float(row["score"]),
            "score_pct": float(row["score_pct"]),
            "families": {FAMILY_NAMES[f]: (None if pd.isna(row[f]) else float(row[f])) for f in FAMILIES},
            "metrics": {k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                        for k in ("mom", "vol", "gpa", "roa", "ep", "bm", "issuance", "assetgrowth")},
            "strengths": good, "watch": bad, "expect": exp,
            "confidence": explain.confidence_label(exp["beat_spy"]),
            "spark": spark, "news": news_for(sym, since), "reddit": reddit_mentions(sym)})

    if is_month_end and not ledger.has("M4_picks", f"{t:%Y-%m-%d}"):
        rng = np.random.default_rng(SEED + t.toordinal())
        ctrl = scored.iloc[rng.choice(len(scored), size=min(TOP_N, len(scored)), replace=False)]["symbol"].tolist()
        ledger.write_json("M4_picks", f"{t:%Y-%m-%d}", f"picks/{t:%Y-%m-%d}.json", {
            "idea": "M4", "as_of": f"{t:%Y-%m-%d}",
            "picks": [{"rank": p["rank"], "symbol": p["symbol"], "score": p["score"], "price": p["price"]} for p in picks],
            "control": ctrl, "control_seed": SEED + t.toordinal(), "universe": scored["symbol"].tolist()})
        print(f"logged M4 picks for {t:%Y-%m-%d}")
    try:
        news_rows = news_window(t)
    except Exception as e:  # without headlines the boom models run with news features missing; M8 skips
        print(f"[news] 91-day window failed: {type(e).__name__}: {e}")
        news_rows = None
    try:
        boom, boom_scored = live_boom.section(boom_rows, uni, tmap, names, news_rows)
    except Exception as e:  # the experimental section must never take the main dashboard down
        print(f"[boom] section skipped: {type(e).__name__}: {e}")
        boom, boom_scored = {}, {}
    if boom_scored and is_week_end:
        for idea, s in boom_scored.items():
            if ledger.has(f"{idea}_watch", f"{t:%Y-%m-%d}"):
                continue
            rng = np.random.default_rng(SEED + t.toordinal())
            ctrl = s.iloc[rng.choice(len(s), size=TOP_N, replace=False)]["symbol"].tolist()
            ledger.write_json(f"{idea}_watch", f"{t:%Y-%m-%d}", f"{idea.lower()}/{t:%Y-%m-%d}.json", {
                "idea": idea, "as_of": f"{t:%Y-%m-%d}", "note": "screen failed; logged for the dashboard only",
                "picks": [{"rank": i + 1, "symbol": r["symbol"], "prob": float(r["prob"]), "price": float(r["price"])}
                          for i, r in s.head(TOP_N).iterrows()],
                "control": ctrl, "control_seed": SEED + t.toordinal(), "universe": s["symbol"].tolist()})
            print(f"logged {idea} watch list for {t:%Y-%m-%d}")
    if is_week_end and not ledger.has("M8_picks", f"{t:%Y-%m-%d}"):
        try:
            if news_rows is None:
                raise RuntimeError("no headlines")
            m8, n_rows, n_ev = m8_ranking(uni, t, rows=news_rows)
        except Exception as e:  # a failed news download skips this month's M8 log; M4 is unaffected
            print(f"[M8] not logged for {t:%Y-%m-%d}: {type(e).__name__}: {e}")
            m8 = None
        if m8 is not None and len(m8) >= TOP_N:
            top8 = m8.nsmallest(TOP_N, "rank")
            rng = np.random.default_rng(SEED + t.toordinal())
            ctrl8 = m8.iloc[rng.choice(len(m8), size=TOP_N, replace=False)]["symbol"].tolist()
            ledger.write_json("M8_picks", f"{t:%Y-%m-%d}", f"m8/{t:%Y-%m-%d}.json", {
                "idea": "M8", "as_of": f"{t:%Y-%m-%d}", "headlines": n_rows, "events": n_ev,
                "picks": [{"rank": int(r["rank"]), "symbol": r["symbol"], "score": float(r["score"]),
                           "up": int(r["an_up"]), "down": int(r["an_down"]), "price": float(r["price"])}
                          for _, r in top8.iterrows()],
                "control": ctrl8, "control_seed": SEED + t.toordinal(), "universe": m8["symbol"].tolist(),
                # every universe stock's inputs, so a whole-cross-section statistic can be pre-registered later
                "universe_scores": [[r["symbol"], float(r["score"]), int(r["an_up"]), int(r["an_down"])]
                                    for _, r in m8.iterrows()]})
            print(f"logged M8 picks for {t:%Y-%m-%d} ({n_ev} analyst actions)")
    if is_week_end and not ledger.has("R1_picks", f"{t:%Y-%m-%d}"):
        try:  # the FWD1 shadow book must never take the M4 log down
            rec = r1_live.record(closes, uni["symbol"].tolist(), t)
            if rec is not None:
                ledger.write_json("R1_picks", f"{t:%Y-%m-%d}", f"r1/{t:%Y-%m-%d}.json", rec)
                print(f"logged R1 picks for {t:%Y-%m-%d}: {len(rec['picks'])} of {len(rec['universe'])}")
        except Exception as e:
            print(f"[R1] not logged for {t:%Y-%m-%d}: {type(e).__name__}: {e}")
    problems = ledger.verify()
    m8_latest = None
    for e in ledger.entries():
        if e["kind"] == "M8_picks":
            m8_latest = json.loads((root / e["file"]).read_text())
    live_tests = {"M8": {"latest": None if m8_latest is None else
                         {"as_of": m8_latest["as_of"], "picks": [dict(p, name=names.get(p["symbol"]) or "")
                                                                 for p in m8_latest["picks"]]},
                         "record": track_record(ledger, closes, t, kind="M8_picks", cap_days=BOOM_TRACK_DAYS)}}
    for idea in boom:
        boom[idea]["record"] = track_record(ledger, closes, t, kind=f"{idea}_watch", cap_days=BOOM_TRACK_DAYS)[-12:]

    data = {"generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "as_of": f"{t:%Y-%m-%d}",
            "universe_size": int(len(scored)), "picks": picks,
            "screen": None if screen is None else {k: screen.get(k) for k in ("primary", "control", "funnel", "phase2_count", "phase1_plus_phase2")},
            "calibration": cal, "track_record": track_record(ledger, closes, t), "live_tests": live_tests, "boom": boom,
            "ledger_ok": not problems, "ledger_entries": len(ledger.entries()),
            "scorecard": scorecard.build(RESULTS_DIR, state),
            "plan": _plan()}
    (root / "latest.json").write_text(json.dumps(data, indent=1, default=str))
    (root / "dashboard.html").write_text(render(data), encoding="utf-8")
    print(f"wrote dashboard: {len(picks)} picks, ledger ok: {not problems}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--log-days-only", action="store_true",
                    help="scheduled runs: only run on the last session of a week or month (forward logs)")
    a = ap.parse_args()
    run(a.state, a.cache, log_days_only=a.log_days_only)


if __name__ == "__main__":
    main()
