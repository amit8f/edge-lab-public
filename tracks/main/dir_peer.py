"""DIR-PEER (research/prereg/DIRECTION.md): is a BOOM1 pick's early move its own, or its sector's?

For each BOOM1 pick (main track, $5+) the stock's move since the previous close minus its sector ETF's, and the
mean move of its same 3-digit-SIC peers in BOOM1's universe that day (at least 3, else no call). Rule: when the
peers moved the same way as the stock, follow the sign of the stock-minus-ETF move; otherwise fade it.
Two entries, each its own result: 9:30 (moves to the last pre-market bar at or before 9:14; BOOM2's +-8% barriers
from the open) and 9:51 (moves to the 9:35 bar; +-4% barriers from the 9:51 price). Period 2020-01..2025-09.

Stages (nx3-study workflow): dirpeer_data (SIC codes from EDGAR, ETF daily closes, 1-minute bars 4:00-9:36 of the
picks, their peers and ETFs; one file per month) and dirpeer (the test). Reads BOOM2's picks and minutes.
Writes research/results/DIR/DIRPEER.json (summary numbers only).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import alpaca_headers
from tracks.main import boom1_study as B
from tracks.main import boom2_study as B2
from tracks.main import ch_data
from tracks.news import nx1_study as N
from tracks.news import nx3_study as X
from tracks.news import nxe_study as E
from tracks.news import w1_data as W

IDEA = 3
FIRST, LAST = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-09-30")
MIN_PEERS, MIN_CALLS, PASS_PTS = 3, 1000, 0.03
PM_LAST = pd.Timedelta(hours=9, minutes=14)
T935 = pd.Timedelta(hours=9, minutes=35)
BOOT, BOOT_SEED = 2000, 0
SECTOR_ETFS = sorted({e for e, _ in ch_data.SECTORS})


def pdir(cache):
    return X.nx3dir(cache) / "dirpeer"


# ---------- pure helpers (tested) ----------

def moves_at(b, open_t, prevc):
    """1-minute bars (t, c) of one symbol's morning -> (move to the last pre-market bar <= 9:14, move to the last
    bar <= 9:35) versus prevc, the previous close in today's raw price terms. NaN where no such bar."""
    day = open_t.normalize()
    if b is None or not len(b) or not np.isfinite(prevc) or prevc <= 0:
        return np.nan, np.nan
    pm = b[(b["t"] < open_t) & (b["t"] <= day + PM_LAST)]
    m35 = b[b["t"] <= day + T935]
    return (float(pm["c"].iloc[-1]) / prevc - 1 if len(pm) else np.nan,
            float(m35["c"].iloc[-1]) / prevc - 1 if len(m35) else np.nan)


def peer_side(stock, etf, peers, n_peers):
    """The registered rule: follow sign(stock - etf) when the peers' mean moved the same way as the stock, else fade
    it; 0 (no call) with fewer than 3 peers or any input missing or flat."""
    stock, etf, peers = (np.asarray(x, float) for x in (stock, etf, peers))
    x = np.sign(stock - etf)
    agree = np.sign(peers) == np.sign(stock)
    side = np.where(agree, x, -x)
    ok = (np.asarray(n_peers) >= MIN_PEERS) & np.isfinite(stock) & np.isfinite(etf) & np.isfinite(peers) \
        & (np.sign(stock) != 0) & (np.sign(peers) != 0)
    return np.where(ok, side, 0.0)


def balanced_accuracy(side, lab):
    """Mean of the hit rate on up-first and on down-first picks (picks that touched neither barrier left out)."""
    side, lab = np.asarray(side, float), np.asarray(lab, float)
    m = side != 0
    r = [float((side[m & (lab == k)] == k).mean()) for k in (1, -1) if (m & (lab == k)).any()]
    return float(np.mean(r)) if len(r) == 2 else float("nan")


def random_sides(days, idea=IDEA):
    """Control: a random side per call, seeded yyyymmdd + idea number for each day."""
    days = pd.DatetimeIndex(days)
    out = np.empty(len(days))
    for d in np.unique(days):
        ix = np.flatnonzero(days == d)
        out[ix] = np.random.default_rng(int(pd.Timestamp(d).strftime("%Y%m%d")) + idea).choice([-1.0, 1.0], len(ix))
    return out


def boot_days(x, n=BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "days": len(x)}
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)),
            "days": len(x)}


def entry_result(df, side, gap, entry, exitp, lab):
    """One entry's result. df has d and open (the price tier for costs)."""
    side, gap, lab = np.asarray(side, float), np.asarray(gap, float), np.asarray(lab, float)
    m = (side != 0) & np.isfinite(np.asarray(exitp, float)) & np.isfinite(np.asarray(entry, float)) & (lab != 2)
    x = df.loc[m, ["d", "open"]].assign(side=side[m], lab=lab[m], gap=gap[m],
                                        entry=np.asarray(entry, float)[m], exitp=np.asarray(exitp, float)[m])
    if x.empty:
        return {"calls": 0, "verdict": "too few"}
    x["ctrl"] = random_sides(x["d"])
    price = x["open"].to_numpy(float)
    x["net"] = B2.net(x["side"], x["entry"], x["exitp"], price)
    x["ctrl_net"] = B2.net(x["ctrl"], x["entry"], x["exitp"], price)
    gross = x["side"] * (x["exitp"] / x["entry"] - 1)
    x["net2x"] = gross - 2 * (gross - x["net"])
    g = np.sign(x["gap"].to_numpy(float))
    ba = balanced_accuracy(x["side"], x["lab"])
    ba_follow, ba_fade = balanced_accuracy(g, x["lab"]), balanced_accuracy(-g, x["lab"])
    pmc = boot_days(x.assign(_d=x["net"] - x["ctrl_net"]).groupby("d")["_d"].mean().to_numpy())
    n = len(x)
    best = np.nanmax([ba_follow, ba_fade])
    passed = (n >= MIN_CALLS and ba >= best + PASS_PTS and pmc["lo"] is not None and pmc["lo"] > 0
              and x["net"].mean() > 0)
    touched = x["lab"] != 0
    return {"verdict": "too few" if n < MIN_CALLS else "passed" if passed else "failed", "calls": n,
            "balanced_accuracy": ba, "ba_follow_gap": ba_follow, "ba_fade_gap": ba_fade, "ba_needed": best + PASS_PTS,
            "hit_rate_touched": float((x.loc[touched, "side"] == x.loc[touched, "lab"]).mean()) if touched.any() else None,
            "touched_share": float(touched.mean()),
            "mean_net": float(x["net"].mean()), "mean_net_2x_cost": float(x["net2x"].mean()),
            "control_mean_net": float(x["ctrl_net"].mean()), "arm_minus_control_per_day": pmc,
            "per_side": {s: {"n": int((x["side"] == v).sum()), "mean_net": float(x.loc[x["side"] == v, "net"].mean())
                             if (x["side"] == v).any() else None} for s, v in (("long", 1), ("short", -1))},
            "follow_share": float((x["side"] == g).mean()),
            "per_year": {int(y): {"n": len(s), "mean_net": float(s["net"].mean())}
                         for y, s in x.groupby(pd.DatetimeIndex(x["d"]).year)}}


# ---------- stages (network / disk) ----------

def universe(cache):
    """BOOM1's universe rows (symbol, d, nxt) in the period, with CIK and the previous close in nxt's raw terms."""
    d = X.nx3dir(cache) / X.CHART
    parts = []
    for f in sorted(d.glob("c_*.parquet")):
        x = pd.read_parquet(f, columns=["symbol", "date", "c", "dv20", "vol20", "nxt", "ac"])
        parts.append(x[(x["c"] >= B.MIN_PX) & (x["dv20"] >= B.MIN_DV) & x["vol20"].notna() & x["nxt"].notna()
                       & (x["nxt"] >= FIRST) & (x["nxt"] <= LAST)][["symbol", "date", "nxt", "ac"]])
    u = pd.concat(parts, ignore_index=True)
    w = W.read_rows(cache, columns=["symbol", "date", "o", "ao", "cik"],
                    filters=[("date", ">=", FIRST - pd.Timedelta(days=10)), ("date", "<=", LAST)])
    w = w.drop_duplicates(["symbol", "date"])
    u = u.merge(w.rename(columns={"date": "nxt"}), on=["symbol", "nxt"], how="inner")
    u["prevc"] = u["ac"] * u["o"] / u["ao"]          # previous close, split-adjusted to nxt's raw prices
    return u[["symbol", "nxt", "cik", "prevc"]]


def stage_data(cache, deadline):
    from core import industry
    from core.bars import fetch_daily_bars, to_frames
    root = pdir(cache)
    root.mkdir(parents=True, exist_ok=True)
    picks = B2.load_all(cache)
    picks = picks[(picks["nxt"] >= FIRST) & (picks["nxt"] <= LAST) & (picks["open"] >= B.MIN_PX)]
    uf = root / "universe.parquet"
    if not uf.exists():
        N.atomic_parquet(universe(cache), uf)
    u = pd.read_parquet(uf)
    sic = industry.sic_codes(set(u["cik"].dropna().astype(int)), root)
    h = alpaca_headers()
    ef = root / "etf_daily.parquet"
    if not ef.exists():
        fr = to_frames(fetch_daily_bars(SECTOR_ETFS, "2019-12-01", "2025-10-01", adjustment="raw", headers=h))
        N.atomic_parquet(pd.concat([x[["c"]].assign(symbol=k) for k, x in fr.items()]).reset_index(names="date"), ef)
    etf = pd.read_parquet(ef)
    u["sic"] = [sic.get(int(c)) if pd.notna(c) else None for c in u["cik"]]
    u["sic3"] = [str(s)[:3] if s else None for s in u["sic"]]
    u["etf"] = [ch_data.sector_etf(s) for s in u["sic"]]
    ses = X.sessions(X.calendar(cache)).set_index("date")
    for m, gm in picks.groupby(picks["nxt"].dt.to_period("M"), sort=True):
        f = root / "months" / f"p_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            return False
        out = []
        for day, g in gm.groupby("nxt"):
            if day not in ses.index:
                continue
            ud = u[u["nxt"] == day]
            pk = g[["symbol"]].merge(ud, on="symbol", how="left")
            peers = ud[ud["sic3"].isin(set(pk["sic3"].dropna()))]
            etfs = sorted(set(pk["etf"].dropna()))
            syms = sorted(set(peers["symbol"]) | set(g["symbol"]) | set(etfs))
            o_t, prev = ses.loc[day, "open_t"], ses.loc[day, "prev"]
            bars = E.bars_1min(syms, day + pd.Timedelta(hours=4), day + T935 + pd.Timedelta(minutes=1), h)
            mv = {s: moves_at(bars.get(s), o_t, pc) for s, pc in zip(ud["symbol"], ud["prevc"])}
            ec = etf[etf["date"] == prev].set_index("symbol")["c"]
            emv = {e: moves_at(bars.get(e), o_t, float(ec.get(e, np.nan))) for e in etfs}
            for r in pk.itertuples(index=False):
                sm = mv.get(r.symbol, (np.nan, np.nan))
                em = emv.get(r.etf, (np.nan, np.nan))
                ok = isinstance(r.sic3, str)
                pr = peers[(peers["sic3"] == r.sic3) & (peers["symbol"] != r.symbol)] if ok else peers.iloc[0:0]
                pm = np.array([mv.get(s, (np.nan, np.nan)) for s in pr["symbol"]], dtype=float).reshape(-1, 2)
                out.append({"symbol": r.symbol, "nxt": day, "sic3": r.sic3 if ok else None, "etf": r.etf,
                            "s_pm": sm[0], "s_935": sm[1], "e_pm": em[0], "e_935": em[1],
                            "p_pm": float(np.nanmean(pm[:, 0])) if np.isfinite(pm[:, 0]).any() else np.nan,
                            "p_935": float(np.nanmean(pm[:, 1])) if np.isfinite(pm[:, 1]).any() else np.nan,
                            "n_pm": int(np.isfinite(pm[:, 0]).sum()), "n_935": int(np.isfinite(pm[:, 1]).sum())})
        N.atomic_parquet(pd.DataFrame(out), f)
        print(f"[dirpeer data] {m}: {len(gm)} picks", flush=True)
    return True


def run(cache, out):
    picks = B2.load_all(cache)
    picks = picks[(picks["nxt"] >= FIRST) & (picks["nxt"] <= LAST) & (picks["open"] >= B.MIN_PX)]
    mv = pd.concat([pd.read_parquet(f) for f in sorted((pdir(cache) / "months").glob("p_*.parquet"))], ignore_index=True)
    df = picks.merge(mv, on=["symbol", "nxt"], how="inner").reset_index(drop=True)
    rep = {"idea_count": "145 (160 with phase 1)", "picks": len(picks), "picks_with_moves": len(df),
           "sector_map": "tracks/main/ch_data.py SECTORS (SIC ranges to the 11 SPDR sector ETFs, committed before "
                         "this test; SPY where no range matches)",
           "gap_rules_note": "follow-gap / fade-gap use the sign of the stock's own move at the same observation time",
           "entries": {}}
    s30 = peer_side(df["s_pm"], df["e_pm"], df["p_pm"], df["n_pm"])
    rep["entries"]["9:30"] = entry_result(df, s30, df["s_pm"], df["open"], df["exit"], df["lab"])
    s51 = peer_side(df["s_935"], df["e_935"], df["p_935"], df["n_935"])
    rep["entries"]["9:51"] = entry_result(df, s51, df["s_935"], df["e951"], df["exit951"], df["lab951"])
    v = [e["verdict"] for e in rep["entries"].values()]
    rep["verdict"] = "passed" if "passed" in v else "too few" if all(x == "too few" for x in v) else "failed"
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "DIRPEER.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    return rep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dirpeer_data", "dirpeer"], required=True)
    ap.add_argument("--cache", default="cache/nx1")
    ap.add_argument("--out", default="research/results/DIR")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    if a.stage == "dirpeer_data":
        done = stage_data(a.cache, time.monotonic() + a.budget_min * 60)
        print("[dirpeer data] done" if done else "[dirpeer data] budget used; rerun to continue")
    else:
        run(a.cache, a.out)


if __name__ == "__main__":
    main()
