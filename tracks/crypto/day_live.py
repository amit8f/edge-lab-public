"""CRY-DAY-FWD (research/prereg/CRYDAYFWD.md): the frozen CRY-DAY model watched forward. Ledger only: no orders.

Binance's archive publishes a month after it ends, so each run downloads every finished month since 2026-11,
rebuilds the decision rows with day_study's own code, scores them with the committed models and bar, and logs the
trades not yet logged. Every earlier trade is recomputed and must match the ledger.

    python -m tracks.crypto.day_live --state state
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats as CS
from core.ledger import Ledger, sha256_file
from tracks.crypto import cdata as C
from tracks.crypto import day_study as D
from tracks.main import pick2_study as P2

ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / "research/results/CRYDAY/CRYDAY_fit.json"
START = pd.Timestamp("2026-11-15")            # first decision bar that counts
LAST_BAR = pd.Timestamp("2027-12-31 23:45")
FIRST_MONTH = "2026-11"                       # 2026-11-01..14 only warm up the 7-day inputs
MIN_TRADES, READ_AT, TRIALS = 50, 100, 178
NO_FUND = pd.DataFrame({"t": pd.to_datetime([]), "rate": []})


# ---------- pure helpers (tested) ----------

def quarters_of(uni):
    """{quarter: [symbols]} -> {symbol: {quarters}}."""
    out = {}
    for q, syms in uni.items():
        for s in syms:
            out.setdefault(s, set()).add(q)
    return out


def build_rows(bars, fund, uni):
    """day_study's rows (stage_rows + load) from candles in memory. bars/fund: {symbol: frame}."""
    btc = D.features(D.regular(bars["BTCUSDT"]), NO_FUND)
    btc = btc[["t", "r_1", "r_4", "r_16", "r_96", "vol_16"]].rename(columns=lambda x: x if x == "t" else f"btc_{x}")
    parts = []
    for s, qs in sorted(quarters_of(uni).items()):
        if s not in bars:
            continue
        R = D.coin_rows(bars[s], fund.get(s, NO_FUND), qs).merge(btc, on="t", how="left")
        R["rel_16"] = R["r_16"] - R["btc_r_16"]
        R.insert(0, "symbol", s)
        parts.append(R)
    R = pd.concat(parts, ignore_index=True)
    R["breadth"] = R.groupby("t")["r_16"].transform("mean")
    return R.sort_values(["t", "symbol"]).reset_index(drop=True)


def trades_of(R, models, bar, through):
    """The book's trades with their controls on the decision bars START..through."""
    R = R[(R["t"] >= START) & (R["t"] <= through)].reset_index(drop=True)
    cols = ["t", "symbol", "side", "pred", "net", "gross", "exit", "bars", "ctrl"]
    if not len(R):
        return pd.DataFrame(columns=cols)
    c = D.candidates(R, D.predict_sides(models, R), bar)
    return D.add_control(c.loc[D.take(c)], R, np.random.default_rng(D.SEED))[cols].reset_index(drop=True)


def same(old, new):
    """True when every logged trade is recomputed identically (same bar, coin, side and net)."""
    if len(new) < len(old):
        return False
    return all(o["t"] == str(n.t) and o["symbol"] == n.symbol and o["side"] == n.side and abs(o["net"] - n.net) < 1e-9
               for o, n in zip(old, new.itertuples(), strict=False))


def judge(tk, through, btc):
    days = pd.date_range(START, (through + D.BAR).normalize(), freq="D")
    s = D.summary(tk, days, btc)
    out = {"first": str(START), "through": str(through), "summary": s, "min_trades": MIN_TRADES, "read_at": READ_AT}
    if s["trades"]:
        s["dsr"] = CS.deflated_sharpe(D.daily(tk, "net", days).to_numpy(), TRIALS)
    if s["trades"] >= READ_AT or through >= LAST_BAR - D.H * D.BAR:
        if s["trades"] < MIN_TRADES:
            out["verdict"] = "no conclusion: fewer than 50 trades"
        else:
            mc, al = s["minus_control_daily"], s["alpha_on_btc"]
            out["rules"] = {"mean_net_above_0": bool(s["mean_net"] > 0), "t_3": bool(s["book_daily"]["t_nw"] >= 3),
                            "trades_50": True, "beats_control_t_2": bool(mc["mean"] > 0 and mc["t_nw"] >= 2),
                            "alpha_on_btc_t_2": bool(al["alpha"] > 0 and al["t"] >= 2)}
            out["verdict"] = "PASS" if all(out["rules"].values()) else "FAIL"
    return out


# ---------- network ----------

def months_done(today):
    """Finished months since FIRST_MONTH whose BTC candles the archive already has."""
    out = []
    for m in C.months(FIRST_MONTH, str(pd.Period(today, freq="M") - 1)):
        if C.fetch(C.kline_url("futures/um", "BTCUSDT", "15m", m)) is None:
            break
        out.append(m)
    return out


def universe(root, last_month):
    """The quarterly universe from 2026Q4 to last_month's quarter, ranked as cdata.stage_rank does; kept in state."""
    f = root / "universe.json"
    uni = json.loads(f.read_text()) if f.exists() else {}
    need = [q for q in pd.period_range(pd.Period(FIRST_MONTH, freq="M").asfreq("Q"),
                                       pd.Period(last_month, freq="M").asfreq("Q"), freq="Q").astype(str) if q not in uni]
    syms = C.list_symbols() if need else []
    for q in need:
        m = C.rank_month(q)

        def vol(s, m=m):
            raw = C.fetch(C.kline_url("futures/um", s, "1d", m))
            return s, (float(C.parse_klines(raw)["qv"].sum()) if raw else 0.0)
        with C.ThreadPoolExecutor(C.THREADS) as ex:
            uni[q] = C.top_symbols(dict(ex.map(vol, syms)))
        print(f"[cry-day-fwd] {q} (by {m}): {' '.join(uni[q])}", flush=True)
    if need:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(uni, indent=1))
        tmp.replace(f)
    return uni


def download(uni, months):
    bars, fund = {}, {}
    for s in sorted(set(quarters_of(uni)) | {"BTCUSDT"}):
        k = [C.parse_klines(r) for r in (C.fetch(C.kline_url("futures/um", s, "15m", m)) for m in months) if r]
        if not k:
            print(f"[cry-day-fwd] {s}: no candles", flush=True)
            continue
        bars[s] = pd.concat(k, ignore_index=True).sort_values("t").drop_duplicates("t").reset_index(drop=True)
        fr = [C.parse_funding(r) for r in (C.fetch(f"{C.BASE}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")
                                           for m in months) if r]
        fund[s] = (pd.concat(fr, ignore_index=True).sort_values("t").drop_duplicates("t").reset_index(drop=True)
                   if fr else NO_FUND)
    return bars, fund


def source(root, today):
    months = months_done(today)
    if not months:
        return None
    uni = universe(root, months[-1])
    return (*download(uni, months), uni)


def load_models():
    fit = json.loads(FIT.read_text())
    paths = {s: ROOT / fit["refit"]["models"][s] for s in D.SIDES}
    return ({s: P2.load_model(p) for s, p in paths.items()}, fit["refit"]["bar"],
            {s: sha256_file(p) for s, p in paths.items()})


# ---------- job ----------

def run(state, today=None, source=source, load_models=load_models):
    root = Path(state) / "cryday"
    led = Ledger(root / "ledger")
    got = source(root, today or pd.Timestamp.now("UTC").tz_localize(None))
    if got is None:
        print("[cry-day-fwd] no finished month in the archive yet", flush=True)
        return
    bars, fund, uni = got
    through = min(bars["BTCUSDT"]["t"].max() - (D.H + 1) * D.BAR, LAST_BAR)
    entries = [e for e in led.entries() if e["kind"] == "trades"]
    old = [t for e in entries for t in json.loads((led.root / e["file"]).read_text())["trades"]]
    if entries and str(through) <= entries[-1]["date"]:
        print(f"[cry-day-fwd] nothing new after {entries[-1]['date']}", flush=True)
        return
    if through < START:
        print("[cry-day-fwd] the archive does not reach the first day yet", flush=True)
        return
    models, bar, shas = load_models()
    tk = trades_of(build_rows(bars, fund, uni), models, bar, through)
    if not same(old, tk):
        raise RuntimeError("a logged trade is not reproduced from the archive; nothing written")
    new = tk.iloc[len(old):]
    led.write_json("trades", str(through), f"trades/{through:%Y-%m-%d}.json",
                   {"through": str(through), "bar": bar, "models_sha256": shas, "universe": uni,
                    "trades": json.loads(new.assign(t=new["t"].astype(str)).to_json(orient="records"))})
    probs = led.verify()
    if probs:
        raise RuntimeError(f"ledger problems: {probs[:3]}")
    btc = bars["BTCUSDT"].set_index("t")["c"].resample("D").last().pct_change()
    res = judge(tk, through, btc)
    res["last_trades"] = json.loads(tk.tail(10).assign(t=tk.tail(10)["t"].astype(str)).to_json(orient="records"))
    tmp = root / "summary.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(root / "summary.json")
    print(f"[cry-day-fwd] through {through}: {len(new)} new trades, {len(tk)} in all", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    run(ap.parse_args().state)


if __name__ == "__main__":
    main()
