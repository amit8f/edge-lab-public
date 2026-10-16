"""K3 (research/prereg/K3.md): Kronos weekly picks, forward test only.

On the last session of each week, the pinned Kronos-small model forecasts the 21-day return of the 500
most traded $5+ common stocks from their last 256 all-adjusted daily SIP bars. The top 20, a same-seed
random 20 and every forecast go to the hash-chained ledger (branch `state-kronos`). Nothing is traded.
`score_weeks` and `summary` are the pre-registered primary test, run once weeks have outcomes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca
from core.bars import fetch_daily_bars, to_frames
from core.ledger import Ledger
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

SEED = 20261016
N_UNIVERSE = 500
TOP = 20
LOOKBACK = 256
PRED_LEN = 21
SAMPLES = 4
CHUNK = 25
MIN_PRICE = 5.0
DV_DAYS = 20
GAP_DAYS = 15
COST = 0.002
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
SETTINGS = {"lookback": LOOKBACK, "pred_len": PRED_LEN, "T": 1.0, "top_k": 0, "top_p": 0.9,
            "sample_count": SAMPLES, "clip": 5, "max_context": 512}


def clean_series(df):
    """Drop zero-volume bars and keep only the part after the last gap of more than 15 calendar days."""
    df = df[df["v"] > 0].sort_index()
    if len(df) < 2:
        return df
    gaps = np.flatnonzero(np.diff(df.index.values).astype("timedelta64[D]").astype(int) > GAP_DAYS)
    return df.iloc[gaps[-1] + 1:] if len(gaps) else df


def universe(raw, t, n=N_UNIVERSE):
    """The n stocks with the highest median raw dollar volume over the 20 sessions up to t, among those
    that traded on t at a raw close of $5 or more. Ties break by symbol."""
    rows = []
    for sym, df in raw.items():
        d = df[df.index <= t].tail(DV_DAYS)
        if d.empty or d.index[-1] != t or d["c"].iloc[-1] < MIN_PRICE:
            continue
        rows.append((sym, float((d["c"] * d["v"]).median()), float(d["c"].iloc[-1])))
    u = pd.DataFrame(rows, columns=["symbol", "dollar_volume", "close"])
    return u.sort_values(["dollar_volume", "symbol"], ascending=[False, True]).head(n).reset_index(drop=True)


def model_inputs(adj, symbols, t):
    """{symbol: 256-bar OHLCV frame ending at t} and the list of symbols without enough clean bars."""
    inputs, dropped = {}, []
    for sym in symbols:
        df = adj.get(sym)
        s = clean_series(df[df.index <= t]) if df is not None else None
        if s is None or len(s) < LOOKBACK or s.index[-1] != t:
            dropped.append(sym)
            continue
        s = s.tail(LOOKBACK)
        inputs[sym] = pd.DataFrame({"open": s["o"], "high": s["h"], "low": s["l"], "close": s["c"],
                                    "volume": s["v"]}).astype(float)
    return inputs, dropped


def future_stamps(t):
    """Timestamps of the 21 forecast steps (only weekday/day/month feed Kronos's time embedding)."""
    return pd.Series(pd.bdate_range(t + pd.Timedelta(days=1), periods=PRED_LEN))


def forecast(inputs, predictor, seed):
    """{symbol: predicted close on step 21 / last close - 1}, in sorted-symbol chunks of 25."""
    import torch
    torch.manual_seed(seed)
    syms = sorted(inputs)
    out = {}
    for i in range(0, len(syms), CHUNK):
        part = syms[i:i + CHUNK]
        dfs = [inputs[s].reset_index(drop=True) for s in part]
        xs = [pd.Series(inputs[s].index) for s in part]
        ys = [future_stamps(inputs[s].index[-1]) for s in part]
        preds = predictor.predict_batch(dfs, xs, ys, pred_len=PRED_LEN, T=SETTINGS["T"], top_k=SETTINGS["top_k"],
                                        top_p=SETTINGS["top_p"], sample_count=SAMPLES, verbose=False)
        for s, p in zip(part, preds):
            out[s] = float(p["close"].iloc[-1] / inputs[s]["close"].iloc[-1] - 1)
        print(f"[K3] forecast {min(i + CHUNK, len(syms))}/{len(syms)}", flush=True)
    return out


def pick(fc, t):
    """Top 20 by forecast (ties by symbol) and the same-seed random 20 from the whole universe."""
    ranked = sorted(fc.items(), key=lambda kv: (-kv[1], kv[0]))
    uni = sorted(fc)
    rng = np.random.default_rng(SEED + t.toordinal())
    control = sorted(rng.choice(uni, size=min(TOP, len(uni)), replace=False).tolist())
    return [s for s, _ in ranked[:TOP]], control


def load_kronos(kronos_dir, pins):
    """KronosPredictor from the pinned code checkout and the pinned HuggingFace revisions. Unpinned repos are
    resolved to their current commit hash; the returned pins are saved and reused every later week."""
    sys.path.insert(0, str(kronos_dir))
    from huggingface_hub import HfApi
    from model import Kronos, KronosPredictor, KronosTokenizer
    pins = dict(pins)
    for repo in (TOKENIZER, MODEL):
        pins.setdefault(repo, HfApi().model_info(repo).sha)
    tok = KronosTokenizer.from_pretrained(TOKENIZER, revision=pins[TOKENIZER]).eval()
    mod = Kronos.from_pretrained(MODEL, revision=pins[MODEL]).eval()
    return KronosPredictor(mod, tok, device="cpu", max_context=SETTINGS["max_context"], clip=SETTINGS["clip"]), pins


def candidates():
    """Alpaca's active US equities that classify() keeps as listed common stock."""
    from core.instruments import classify
    a = classify(pd.DataFrame(alpaca.get(f"{alpaca.TRADE}/assets", {"status": "active", "asset_class": "us_equity"})))
    return sorted(a.loc[a["excl_reason"].isna(), "symbol"].tolist())


def fetch_frames(symbols, start, end, adjustment, size=500):
    from core.env import alpaca_headers
    h, out = alpaca_headers(), {}
    for b in range(0, len(symbols), size):
        out.update(to_frames(fetch_daily_bars(symbols[b:b + size], start, end, adjustment=adjustment, headers=h)))
        print(f"[K3] {adjustment} bars batch {b // size + 1}/{-(-len(symbols) // size)}", flush=True)
    return out


def fetch_all(t, now):
    """Raw bars for the dollar-volume ranking (all candidates) and all-adjusted bars for the universe."""
    end = alpaca.sip_end(t, now)
    syms = candidates()
    raw = fetch_frames(syms, f"{(t - pd.Timedelta(days=45)).date()}", end, "raw")
    u = universe(raw, t)
    adj = fetch_frames(u["symbol"].tolist(), f"{(t - pd.Timedelta(days=550)).date()}", end, "all")
    return u, adj, len(syms)


def week_session(now):
    from tracks.main.longterm.live import last_session, week_end
    t, _ = last_session(now)
    return t, week_end(t)


def run(state_dir, kronos_dir, now=None, fetch=fetch_all, load=load_kronos, session=week_session, dry_run=False):
    now = now or pd.Timestamp.now(tz="UTC")
    root = Path(state_dir) / "kronos"
    led = Ledger(root)
    t, is_week_end = session(now)
    date = f"{t:%Y-%m-%d}"
    if not dry_run and not is_week_end:
        print(f"[K3] {date} is not the last session of its week; nothing to log")
        return None
    if not dry_run and led.has("K3_picks", date):
        print(f"[K3] {date} already logged")
        return None
    t0 = time.time()
    u, adj, n_cand = fetch(t, now)
    inputs, dropped = model_inputs(adj, u["symbol"].tolist(), t)
    pins_path = root / "pins.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    predictor, pins = load(kronos_dir, pins)
    fc = forecast(inputs, predictor, SEED + t.toordinal())
    top, control = pick(fc, t)
    close = dict(zip(u["symbol"], u["close"]))
    rec = {"idea": "K3", "as_of": date, "kronos_commit": KRONOS_COMMIT, "model": MODEL, "tokenizer": TOKENIZER,
           "hf_revisions": pins, "settings": SETTINGS, "torch_seed": SEED + t.toordinal(),
           "candidates": n_cand, "universe_size": int(len(u)), "forecast_n": len(fc), "dropped": sorted(dropped),
           "picks": [{"symbol": s, "forecast": fc[s], "close_raw": close[s]} for s in top],
           "control": control, "control_seed": SEED + t.toordinal(),
           "universe": sorted(fc), "forecasts": [[s, fc[s]] for s in sorted(fc)],
           "seconds": round(time.time() - t0, 1)}
    if dry_run:
        tmp = root / f"dryrun-{date}.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(root / f"dryrun-{date}.json")
        print(f"[K3] dry run {date}: {len(fc)} forecasts in {rec['seconds']}s; not logged")
        return rec
    if not pins_path.exists():
        pins_path.write_text(json.dumps(pins, indent=1))
    led.write_json("K3_picks", date, f"k3/{date}.json", rec)
    print(f"[K3] logged {date}: top {', '.join(top)} ({rec['seconds']}s)")
    return rec


def _window(df, t, sessions):
    """All-adjusted return from the open of the first session after t to the close of the 21st session
    counting that one. A series that ends earlier keeps its last close; None if it never trades after t."""
    after = [s for s in sessions if s > t]
    if len(after) < PRED_LEN:
        return None
    entry, exit_ = after[0], after[PRED_LEN - 1]
    if df is None:
        return None
    d = df[(df.index >= entry) & (df.index <= exit_)]
    if d.empty or d.index[0] != entry:
        return None
    return float(d["c"].iloc[-1] / d["o"].iloc[0] - 1)


def score_weeks(records, frames, sessions):
    """One row per logged week with a full 21-session window: picks, control and universe means, and the
    primary excess (picks - universe - 0.2%)."""
    sessions = sorted(pd.Timestamp(s) for s in sessions)
    rows = []
    for rec in records:
        t = pd.Timestamp(rec["as_of"])
        if len([s for s in sessions if s > t]) < PRED_LEN:
            continue
        row = {"date": rec["as_of"]}
        for k, syms in (("picks", [p["symbol"] for p in rec["picks"]]), ("control", rec["control"]),
                        ("universe", rec["universe"])):
            rs = [r for r in (_window(frames.get(s), t, sessions) for s in syms) if r is not None]
            row[k], row[f"{k}_n"] = (float(np.mean(rs)) if rs else np.nan), len(rs)
        row["excess"] = row["picks"] - row["universe"] - COST
        rows.append(row)
    return pd.DataFrame(rows)


def summary(weekly):
    """Primary statistic: mean excess, Newey-West t (6 lags), moving-block bootstrap CI (blocks of 8)."""
    x = weekly["excess"].to_numpy(float)
    m, se, t = nw_tstat(x, 6)
    lo, hi = block_bootstrap_ci(x, 8, 10_000, SEED) if len(x) >= 8 else (np.nan, np.nan)
    return {"weeks": int(np.isfinite(x).sum()), "mean": m, "se": se, "t": t, "ci": [lo, hi],
            "control_minus_universe": float((weekly["control"] - weekly["universe"]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--kronos", required=True, help="checkout of shiyu-coder/Kronos at KRONOS_COMMIT")
    ap.add_argument("--dry-run", action="store_true", help="forecast the last session but log nothing")
    a = ap.parse_args()
    run(a.state, a.kronos, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
