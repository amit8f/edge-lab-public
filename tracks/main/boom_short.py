"""BOOM-SHORT forward shadow book (research/prereg/BOOMSHORT.md). Ledger only: no orders, $5,000 book.

Each regular session, before the open:
1. BOOM1's universe at the previous close: every symbol in the frozen list (the stocks BOOM1 was trained on),
   raw close >= $5, 20-day dollar volume >= $1M, chart and market inputs from daily SIP bars (core.chart_features).
2. Pre-open news inputs, as BOOM1's history read them (NX3's events decided at or before the open): Alpaca's
   Benzinga articles since the previous close, scored with NX1's pinned FinBERT, deduplicated over 3 days, made
   into events with nx3_study.session_events; no headline text is kept.
3. The frozen BOOM1 model scores every stock; the day's 20 highest are BOOM1's picks (only if 100+ stocks scored).
4. Pre-market 1-minute SIP bars (raw) up to now - 16 minutes give each pick's pre-market gap and dollar volume;
   the frozen D6 model scores each pick (P(up first) - P(down first)).
5. A pick is shorted when its D6 score <= the frozen cut, Alpaca marks it shortable and easy to borrow, at most
   MAX_SHORTS a day (lowest scores first), $500 each. Control: as many of the day's other 20 picks drawn at
   random (default_rng(SEED + yyyymmdd)), shorted the same way. The decision is logged (kind boomshort_day)
   before 9:30 ET; a day decided later than that is logged as missed, with no trades.
After the close (and on later runs), each logged day is settled from 1-minute SIP bars (raw) exactly as BOOM2's
history: entry = the first regular-session bar's open, exit at the -8% / +8% barrier touched first (filled at the
barrier), else the close; net = short return - NX3's cost model (one day, borrow). Kind boomshort_settle.

Usage: python -m tracks.main.boom_short --state state [--wait-min 75]   (decide today, then settle)
       python -m tracks.main.boom_short --state state --settle-only
       python -m tracks.main.boom_short --check                         (read every source once, no state)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core import chart_features as C
from core.env import alpaca_headers
from core.ledger import Ledger
from tracks.main import boom1_study as B
from tracks.main import boom2_study as B2
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import nxe_study as E

ET = X.ET
MODEL_DIR = Path("models/boom_short")
NX1_META = Path("research/results/NX1/nx1_v1_meta.json")
SEED = 20261109
BOOK, PER_TRADE = 5000.0, 500.0
MAX_SHORTS = 10
HIST_DAYS = 420
DECIDE_AT = pd.Timedelta(hours=9, minutes=28)       # ET; logged before the 9:30 open
LATEST = pd.Timedelta(hours=9, minutes=30)
COVER = 0.10                                         # the bearish tenth (BOOM2 diagnosis, post hoc)
FREEZE_YEAR = 2025                                   # the walk-forward fold whose models are frozen


# ---------- frozen models ----------

def freeze(cache, out=MODEL_DIR):
    """Freezes the 2025 fold of BOOM1's and BOOM2's walk-forward exactly as tested (same rows, seeds and
    settings): BOOM1 trained on stock-days to the purge before 2025, D6 on the 2019-2024 picks; cut = the 10%
    quantile of D6's scores on its out-of-sample 2025 picks. Checks the frozen BOOM1 reproduces the saved picks'
    scores. Writes the two models, the symbol list and manifest.json (sha256 of every file)."""
    from catboost import CatBoostClassifier
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = B.load(cache)
    caldates = pd.DatetimeIndex(X.calendar(cache)["date"])
    start = pd.Timestamp(f"{FREEZE_YEAR}-01-01")
    cut_d = caldates[max(caldates.searchsorted(start) - N.PURGE - 1, 0)]
    tr = df[df["d"] <= cut_d]
    if len(tr) > B.TRAIN_ROWS:
        tr = tr.iloc[np.sort(np.random.default_rng(B.SEED + FREEZE_YEAR).choice(len(tr), B.TRAIN_ROWS, replace=False))]
    m1 = B.classifier().fit(tr[B.INPUTS].to_numpy(np.float32), tr["boom"].to_numpy())
    m1.save_model(str(out / "boom1.cbm"))
    picks = B2.load_all(cache)
    p25 = picks[picks["year"] == FREEZE_YEAR]
    # CatBoost is not bit-identical across runners (thread counts), so the refit is checked on what matters:
    # how many of each 2025 day's 20 saved picks the refit model also ranks top 20 among all that day's stocks.
    te = df[(df["d"] >= start) & (df["d"] < pd.Timestamp(f"{FREEZE_YEAR + 1}-01-01"))]
    te = te.assign(s=m1.predict_proba(te[B.INPUTS].to_numpy(np.float32))[:, 1])
    top = {(g["nxt"].iloc[0], x) for _, g in te.groupby("d") if len(g) >= B.TOP * 5 for x in g.nlargest(B.TOP, "s")["symbol"]}
    overlap = float(np.mean([(d, x) in top for d, x in zip(p25["nxt"], p25["symbol"])]))
    re = m1.predict_proba(p25[B.INPUTS].to_numpy(np.float32))[:, 1]
    drift = {"max_abs": float(np.nanmax(np.abs(re - p25["score"].to_numpy(float)))),
             "mean_abs": float(np.nanmean(np.abs(re - p25["score"].to_numpy(float)))),
             "corr": float(np.corrcoef(re, p25["score"].to_numpy(float))[0, 1]), "top20_overlap": overlap}
    print(f"[boom-short] refit vs saved 2025 scores: {drift}")
    if overlap < 0.8:
        raise SystemExit(f"[boom-short] frozen BOOM1 keeps only {overlap:.0%} of the saved 2025 picks")
    lab = picks[picks["lab"] != 2]
    tr2, te2 = lab[lab["year"] < FREEZE_YEAR], lab[lab["year"] == FREEZE_YEAR]
    m2 = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=B2.SEED,
                            verbose=0, thread_count=-1, allow_writing_files=False, loss_function="MultiClass")
    m2.fit(tr2[d6_cols()].to_numpy(np.float32), tr2["lab"].to_numpy(int))
    m2.save_model(str(out / "d6.cbm"))
    s = d6_score(m2, te2)
    cut = float(np.quantile(s, COVER))
    sel = te2[s <= cut]
    oos = B2.net(-np.ones(len(sel)), sel["open"], sel["exit"], sel["open"])
    syms = sorted(df["symbol"].unique())
    (out / "symbols.json").write_text(json.dumps(syms))
    man = {"frozen": "2026-11-07, before any BOOM-SHORT day was logged (research/prereg/BOOMSHORT.md)",
           "fold": FREEZE_YEAR, "boom1_train_rows": int(len(tr)), "boom1_train_last_day": f"{cut_d:%Y-%m-%d}",
           "boom1_refit_vs_saved_2025": drift, "d6_train_picks": int(len(tr2)), "d6_classes": [int(c) for c in m2.classes_],
           "cut": cut, "cover": COVER, "oos_2025": {"picks": int(len(te2)), "shorts": int(len(sel)),
                                                     "net_per_trade": float(np.nanmean(oos)),
                                                     "down_first": float((sel["lab"] == -1).mean())},
           "inputs_boom1": B.INPUTS, "inputs_d6": d6_cols(), "symbols": len(syms),
           "sha256": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("boom1.cbm", "d6.cbm", "symbols.json")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k not in ("inputs_boom1", "inputs_d6")}, indent=1))
    return man


def d6_cols():
    return [*B.INPUTS, "score", "pm_gap", "pm_dv", "pre_offering"]


def d6_score(m, d):
    pr = m.predict_proba(d[d6_cols()].to_numpy(np.float32))
    k = [int(c) for c in m.classes_]
    return pr[:, k.index(1)] - pr[:, k.index(-1)]


def load_frozen(dirpath=MODEL_DIR):
    from catboost import CatBoostClassifier
    d = Path(dirpath)
    man = json.loads((d / "manifest.json").read_text())
    for f, h in man["sha256"].items():
        if hashlib.sha256((d / f).read_bytes()).hexdigest() != h:
            raise SystemExit(f"[boom-short] {f} does not match the manifest")
    m1, m2 = CatBoostClassifier(), CatBoostClassifier()
    m1.load_model(str(d / "boom1.cbm"))
    m2.load_model(str(d / "d6.cbm"))
    return man, m1, m2, json.loads((d / "symbols.json").read_text())


# ---------- pure helpers (tested) ----------

def universe_rows(daily, spy, prev):
    """daily: X.daily_ohlcv rows for many symbols; spy: SPY's rows. -> one row per symbol at the previous close
    with BOOM1's chart and market inputs, raw close c and dv20, inside BOOM1's universe."""
    mk = C.market_features(spy)
    mk = mk[mk["date"] == prev]
    if mk.empty:
        return pd.DataFrame()
    rows = []
    for s, g in daily.groupby("symbol"):
        f = C.chart_features(g)
        f = f[f["date"] == prev]
        if len(f):
            rows.append(f.assign(symbol=s))
    if not rows:
        return pd.DataFrame()
    u = pd.concat(rows, ignore_index=True)
    u = u[(u["c"] >= B.MIN_PX) & (u["dv20"] >= B.MIN_DV) & u["vol20"].notna()]
    return u.assign(**{k: float(mk[k].iloc[0]) for k in C.MARKET}).reset_index(drop=True)


def news_inputs(ev):
    """Events (nx3_study.session_events) decided at or before the open -> BOOM1's pre-open news inputs and
    BOOM2's offering flag, per symbol (as boom1_study.pre_open_news and boom2_study.offering_flags)."""
    cols = ["symbol", *B.NEWS, "pre_offering"]
    if ev is None or ev.empty:
        return pd.DataFrame(columns=cols)
    ev = ev[ev["t"] <= ev["open_t"]] if "open_t" in ev else ev
    g = ev.groupby("symbol")
    return pd.DataFrame({"pre_news": 1.0, "pre_articles": g["n_articles"].sum(), "pre_fb_neg": g["fb_neg"].max(),
                         "pre_fb_pos": g["fb_pos"].max(), "pre_lm_net": g["lm_net"].mean(),
                         "pre_fb_max_neg": g["fb_max_neg"].max(), "pre_fb_max_pos": g["fb_max_pos"].max(),
                         "pre_offering": g["f_offering"].max().astype(float)}).reset_index()[cols]


def with_news(u, ni):
    u = u.merge(ni, on="symbol", how="left")
    u[["pre_news", "pre_articles", "pre_offering"]] = u[["pre_news", "pre_articles", "pre_offering"]].fillna(0.0)
    return u


def premarket(b, prev_c, dv20, day):
    """Pre-market inputs from 1-minute bars before the open, up to 9:14 (BOOM2's PM_LAST). No pre-market trade:
    gap 0 and dollar volume 0 (history used the open's gap there, which a pre-open decision cannot see)."""
    pm = b[(b["t"] < day + pd.Timedelta(hours=9, minutes=30)) & (b["t"] <= day + B2.PM_LAST)] if b is not None else None
    if pm is None or pm.empty:
        return {"pm_gap": 0.0, "pm_dv": 0.0, "pm_bars": 0}
    last = float(pm["c"].iloc[-1])
    return {"pm_gap": last / prev_c - 1 if prev_c > 0 else 0.0,
            "pm_dv": float((pm["c"] * pm["v"]).sum()) / dv20 if dv20 > 0 else 0.0, "pm_bars": int(len(pm))}


def choose(picks, cut, short_ok, seed):
    """picks: the day's 20 with d6 scores. -> (shorts, control): shorts are the picks with score <= cut that are
    shortable and easy to borrow, lowest first, at most MAX_SHORTS; the control is as many of the other picks,
    drawn at random."""
    c = picks[picks["d6"] <= cut].sort_values(["d6", "symbol"])
    blocked = sorted(c.loc[~c["symbol"].isin(short_ok), "symbol"])
    sh = c[c["symbol"].isin(short_ok)].head(MAX_SHORTS)
    rest = picks[~picks["symbol"].isin(sh["symbol"])].sort_values("symbol")
    k = min(len(sh), len(rest))
    ctl = rest.iloc[np.sort(np.random.default_rng(seed).choice(len(rest), k, replace=False))] if k else rest.iloc[0:0]
    return list(sh["symbol"]), list(ctl["symbol"]), blocked


def settle_rows(bars, info, open_t, close_t):
    """{symbol: 1-minute bars} -> per symbol BOOM2's label, entry, exit and net short return after costs."""
    out = {}
    for s, (prev_c, dv20) in info.items():
        b = bars.get(s)
        r = B2.pick_row(b, prev_c, open_t, close_t, dv20) if b is not None and len(b) else None
        if r is None or r["lab"] == 2:
            out[s] = {"status": "no bars" if r is None else "ambiguous"}
            continue
        net = float(B2.net(-np.ones(1), [r["open"]], [r["exit"]], [r["open"]])[0])
        out[s] = {"status": "settled", "lab": int(r["lab"]), "entry": r["open"], "exit": r["exit"], "net": net}
    return out


def day_seed(d):
    return SEED + int(f"{pd.Timestamp(d):%Y%m%d}")


# ---------- network ----------

class Api:
    def __init__(self):
        self.h = alpaca_headers()

    def calendar(self, start, end):
        return alpaca.trading_calendar(start, end)

    def assets(self):
        return alpaca.active_assets()

    def daily(self, symbols, first, last):
        parts = [X.daily_ohlcv(symbols[i:i + 200], first, last, self.h) for i in range(0, len(symbols), 200)]
        return pd.concat(parts, ignore_index=True)

    def minutes(self, symbols, start, end):
        return E.bars_1min(symbols, start, end, self.h)

    def news(self, start, end):
        from tracks.main.longterm import news_history
        return news_history.fetch_window(start, end)

    def finbert(self, meta_file):
        return N.finbert_scorer(meta_file)


def now_et():
    return alpaca.utcnow().tz_convert(ET).tz_localize(None)


# ---------- the day ----------

def cached_scorer(finbert):
    """Wraps a FinBERT scorer so each distinct text is scored once per run (same scores, less work at 9:28)."""
    memo = {}

    def run(texts):
        new = [t for t in dict.fromkeys(texts) if t not in memo]
        if new:
            memo.update(zip(new, np.asarray(finbert(new), dtype=float).reshape(len(new), 3)))
        return np.array([memo[t] for t in texts], dtype=float).reshape(len(texts), 3)
    return run


def _utc(t):
    return pd.Timestamp(t, tz=ET).tz_convert("UTC")


def warm_news(api, ses, now, finbert):
    """Before the 9:28 wait: fetch the news so far and score it, so 9:28 only fetches and scores the last minutes.
    Returns (news, fetched_until) or None; a failure here only costs speed."""
    try:
        news = api.news(_utc(ses["prev_close_t"]), _utc(now))
        if news is not None and len(news):
            N.headline_rows(news, finbert)
        return news, now
    except Exception as e:                              # noqa: BLE001 - the 9:28 fetch repeats the work
        print(f"[boom-short] news warm-up failed ({type(e).__name__}); 9:28 fetches everything")
        return None


def pre_open_events(api, root, ses, now, finbert, early=None):
    """News since the previous close to now -> pre-open events; the headline hashes of the last 4 days are kept
    (no text) in <root>/heads.parquet for the 3-day dedup. `early` = (news, until) from warm_news: then only
    news from 5 minutes before `until` is fetched again and articles are merged by id."""
    if early is None or early[0] is None:
        news = api.news(_utc(ses["prev_close_t"]), _utc(now))
    else:
        old, until = early
        new = api.news(_utc(until - pd.Timedelta(minutes=5)), _utc(now))
        parts = [x for x in (old, new) if x is not None and len(x)]
        news = pd.concat(parts, ignore_index=True).drop_duplicates("id", keep="last") if parts else new
    hf = Path(root) / "heads.parquet"
    heads = pd.read_parquet(hf) if hf.exists() else pd.DataFrame(columns=["symbol", "hkey", "t"])
    last = {}
    for s, h, t in zip(heads["symbol"], heads["hkey"], heads["t"]):
        last[(s, h)] = max(t, last.get((s, h), t))
    if news is None or not len(news):
        return pd.DataFrame(), 0
    rows = N.headline_rows(news, finbert)
    rows = rows[(rows["t"] > ses["prev_close_t"]) & (rows["t"] <= ses["open_t"])]
    rows = N.dedup(rows, last) if len(rows) else rows
    keep = pd.concat([heads, rows[["symbol", "hkey", "t"]]], ignore_index=True) if len(rows) else heads
    keep = keep[pd.to_datetime(keep["t"]) >= ses["date"] - pd.Timedelta(days=4)]
    tmp = hf.with_suffix(".tmp")
    keep.to_parquet(tmp)
    tmp.replace(hf)
    if rows.empty:
        return pd.DataFrame(), 0
    ev = X.session_events(rows, pd.DataFrame([ses]))
    return ev.assign(open_t=ses["open_t"]), int(len(rows))


def decide(root, api, cal, ses, man, m1, m2, syms, clock, wait_min, sleep):
    led = Ledger(Path(root) / "ledger")
    d = f"{ses['date']:%Y-%m-%d}"
    if led.has("boomshort_day", d):
        print(f"[boom-short] {d} already decided")
        return
    now = clock()
    if now > ses["date"] + LATEST:
        led.write_json("boomshort_day", d, f"days/{d}.json", {"status": "missed", "reason": "run started after 9:30 ET",
                                                             "started": now.isoformat()})
        print(f"[boom-short] {d}: missed (started {now:%H:%M})")
        return
    if now < ses["date"] + DECIDE_AT - pd.Timedelta(minutes=wait_min):
        print(f"[boom-short] {d}: too early ({now:%H:%M}); a later run decides")
        return
    prev = ses["prev"]
    t0 = time.monotonic()
    daily = api.daily(sorted(set(syms) | {"SPY"}), prev - pd.Timedelta(days=HIST_DAYS), prev)
    spy = daily[daily["symbol"] == "SPY"]
    u = universe_rows(daily[daily["symbol"] != "SPY"], spy, prev)
    prevc = daily[daily["date"] == prev].set_index("symbol")["c"]
    print(f"[boom-short] {d}: {len(u)} stocks in BOOM1's universe ({time.monotonic() - t0:.0f}s)")
    finbert = cached_scorer(api.finbert(Path(root) / "finbert_meta.json"))   # pins FinBERT on the first run
    early = warm_news(api, ses, clock(), finbert) if clock() < ses["date"] + DECIDE_AT else None
    while clock() < ses["date"] + DECIDE_AT:
        sleep(min(30.0, max(1.0, (ses["date"] + DECIDE_AT - clock()).total_seconds())))
    now = clock()
    ev, n_heads = pre_open_events(api, root, ses, now, finbert, early)
    u = with_news(u, news_inputs(ev))
    rec = {"status": "decided", "decided_at": now.isoformat(), "prev": f"{prev:%Y-%m-%d}", "universe": int(len(u)),
           "headlines": n_heads, "news_stocks": int(u["pre_news"].sum()), "cut": man["cut"],
           "models_sha256": man["sha256"], "book": BOOK, "per_trade": PER_TRADE}
    if len(u) < B.TOP * 5:
        rec |= {"status": "too few stocks", "picks": [], "shorts": [], "control": []}
    else:
        u["score"] = m1.predict_proba(u[B.INPUTS].to_numpy(np.float32))[:, 1]
        p = u.nlargest(B.TOP, "score").reset_index(drop=True)
        bars = api.minutes(sorted(p["symbol"]), ses["date"] + pd.Timedelta(hours=4), now - pd.Timedelta(minutes=16))
        pm = pd.DataFrame([premarket(bars.get(s), float(prevc.get(s, np.nan)), float(dv), ses["date"])
                           for s, dv in zip(p["symbol"], p["dv20"])])
        p = pd.concat([p, pm], axis=1)
        p["d6"] = d6_score(m2, p)
        a = api.assets()
        short_ok = set(a.loc[a["shortable"].fillna(False).astype(bool) & a["easy_to_borrow"].fillna(False).astype(bool),
                             "symbol"])
        sh, ctl, blocked = choose(p, man["cut"], short_ok, day_seed(ses["date"]))
        rec |= {"picks": [{"symbol": r.symbol, "boom1": float(r.score), "d6": float(r.d6), "pm_gap": float(r.pm_gap),
                           "pm_dv": float(r.pm_dv), "pm_bars": int(r.pm_bars), "pre_news": float(r.pre_news),
                           "prev_c": float(prevc.get(r.symbol, np.nan)), "dv20": float(r.dv20)} for r in p.itertuples()],
                "shorts": sh, "control": ctl, "blocked_not_etb": blocked,
                "inputs_sha256": hashlib.sha256(p[d6_cols()].to_numpy(np.float64).tobytes()).hexdigest()}
    if clock() > ses["date"] + LATEST:
        rec |= {"status": "missed", "reason": "decision finished after 9:30 ET", "shorts": [], "control": []}
    led.write_json("boomshort_day", d, f"days/{d}.json", rec)
    print(f"[boom-short] {d}: {rec['status']}, shorts {rec.get('shorts')}, control {rec.get('control')}")


def settle(root, api, cal):
    led = Ledger(Path(root) / "ledger")
    ses = X.sessions(cal).set_index("date")
    now = None
    for e in led.entries():
        if e["kind"] != "boomshort_day" or led.has("boomshort_settle", e["date"]):
            continue
        day = pd.Timestamp(e["date"])
        rec = json.loads((Path(root) / "ledger" / e["file"]).read_text())
        if day not in ses.index:
            continue
        now = now or now_et()
        if now < ses.loc[day, "close_t"] + pd.Timedelta(minutes=20):
            continue
        names = sorted(set(rec.get("shorts", [])) | set(rec.get("control", [])))
        info = {x["symbol"]: (x["prev_c"], x["dv20"]) for x in rec.get("picks", []) if x["symbol"] in names}
        o_t, c_t = ses.loc[day, "open_t"], ses.loc[day, "close_t"]
        res = settle_rows(api.minutes(names, day + pd.Timedelta(hours=4), c_t), info, o_t, c_t) if names else {}
        bk = [res[s]["net"] for s in rec.get("shorts", []) if res.get(s, {}).get("status") == "settled"]
        cl = [res[s]["net"] for s in rec.get("control", []) if res.get(s, {}).get("status") == "settled"]
        led.write_json("boomshort_settle", e["date"], f"settle/{e['date']}.json", {
            "trades": {s: res[s] for s in rec.get("shorts", []) if s in res},
            "control": {s: res[s] for s in rec.get("control", []) if s in res},
            "book_mean": float(np.mean(bk)) if bk else None, "control_mean": float(np.mean(cl)) if cl else None,
            "book_minus_control": float(np.mean(bk) - np.mean(cl)) if bk and cl else None,
            "pnl_usd": float(np.sum(bk) * PER_TRADE) if bk else 0.0})
        print(f"[boom-short] settled {e['date']}: {len(bk)} trades, book {np.mean(bk) if bk else float('nan'):+.4f}")


def report(root):
    """The forward test so far: per-day mean net of the book's shorts (NW t, 5 lags) and book minus control."""
    led = Ledger(Path(root) / "ledger")
    rows = [json.loads((Path(root) / "ledger" / e["file"]).read_text()) | {"d": e["date"]}
            for e in led.entries() if e["kind"] == "boomshort_settle"]
    n = sum(len([v for v in r["trades"].values() if v.get("status") == "settled"]) for r in rows)
    bm = np.array([r["book_mean"] for r in rows if r["book_mean"] is not None], float)
    bc = np.array([r["book_minus_control"] for r in rows if r["book_minus_control"] is not None], float)
    return {"closed_trades": n, "days_with_trades": int(len(bm)), "book": X.tstat(bm), "book_minus_control": X.tstat(bc),
            "pnl_usd": float(sum(r["pnl_usd"] for r in rows))}


def run(state, wait_min=75, settle_only=False, api=None, clock=now_et, sleep=time.sleep):
    root = Path(state) / "boomshort"
    root.mkdir(parents=True, exist_ok=True)
    api = api or Api()
    now = clock()
    cal = api.calendar((now - pd.Timedelta(days=30)).date(), (now + pd.Timedelta(days=10)).date())
    cal["date"] = pd.to_datetime(cal["date"])
    if not settle_only:
        ses = X.sessions(cal)
        today = ses[ses["date"] == now.normalize()]
        if len(today):
            man, m1, m2, syms = load_frozen()
            decide(root, api, cal, today.iloc[0], man, m1, m2, syms, clock, wait_min, sleep)
        else:
            print("[boom-short] no session today")
    settle(root, api, cal)
    (root / "report.json").write_text(json.dumps(report(root), indent=1, default=float))
    return 0


def check():
    api = Api()
    man, m1, m2, syms = load_frozen()
    print(f"[boom-short] models ok, cut {man['cut']:.4f}, {len(syms)} symbols")
    now = now_et()
    a = api.assets()
    print(f"[boom-short] assets: {len(a)}, shortable+ETB {int((a['shortable'].fillna(False) & a['easy_to_borrow'].fillna(False)).sum())}")
    prev = now.normalize() - pd.Timedelta(days=1)
    daily = api.daily(syms[:200] + ["SPY"], prev - pd.Timedelta(days=HIST_DAYS), prev)
    last = daily["date"].max()
    u = universe_rows(daily[daily["symbol"] != "SPY"], daily[daily["symbol"] == "SPY"], last)
    print(f"[boom-short] daily bars to {last:%Y-%m-%d}: {len(u)} of 200 symbols in the universe")
    if len(u):
        u = with_news(u, news_inputs(None))
        u["score"] = m1.predict_proba(u[B.INPUTS].to_numpy(np.float32))[:, 1]
        print(u.nlargest(5, "score")[["symbol", "c", "score"]].to_string(index=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--wait-min", type=float, default=75)
    ap.add_argument("--settle-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    return check() if a.check else run(a.state, a.wait_min, a.settle_only)


if __name__ == "__main__":
    sys.exit(main())
