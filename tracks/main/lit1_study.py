"""LIT1 (research/prereg/LIT1.md, Amendments 1-2): setups of 1-3 conditions that light up together.

Stages (cache under <cache>, results under <out>):
  data    insider trades (SEC DERA data sets), cross-asset daily bars, and one feature file per year 2019-2025
          (universe stock-days with every sign, the fixed-bracket labels and the next-10-day path)
  search  2019-2021 only: cut points, then singles, pairs and triples (long and short) -> <cache>/candidates.json
  choose  2022-2023 only: up to 10 families -> <out>/LIT1_families.json
  test    one look, 2024-01 .. 2025-09: three-checks-a-day exit on 5-minute bars, random control on the once-a-day
          exit, FF5+momentum alpha -> <out>/LIT1_test.json

    python -m tracks.main.lit1_study --stage data|search|choose|test --w1 cache/w1 --cache cache/lit1
"""
import argparse
import io
import json
import math
import time
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import industry, sec
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from core.stats import deflated_sharpe, factor_alpha
from tracks.main import sector2_study as S2
from tracks.main import sector3_study as S3
from tracks.main import sector4_study as S4
from tracks.main import sector5_study as S5
from tracks.main.longterm.sec_data import DERA
from tracks.news import w1_data as W
from validation import french

ET = "America/New_York"
YEARS = range(2019, 2026)
DATA_END = pd.Timestamp("2025-09-30")
TRAIN_END, CHOOSE_END, TEST_START = pd.Timestamp("2021-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01")
HOLD, COST, MIN_PX, MIN_DV = 10, 0.002, 5.0, 5e6
CROSS = ["VIXY", "USO", "TLT", "UUP"]
QS = (0.10, 0.25, 0.75, 0.90)
CONT = ["r1", "r5", "r20", "r60", "hi20", "lo20", "hi250", "vol20", "rng", "vr", "gap", "oc", "clv", "streak",
        "logp", "logdv", "news_n", "news_n5", "tone", "days_since_news", "mkt1", "mkt5", "mktvol20", "breadth",
        "sec1", "sec5", "own_minus_sec", "ins_net20"] + [f"{s}_{k}" for s in CROSS for k in ("r1", "r5")]
TYPE_FLAGS = [f"t_{k}" for k in S2.TYPES]
FLAGS = TYPE_FLAGS + ["earn", "k8_other", "f_offer", "ins_buy5", "ins_bigbuy20", "ins_sell5", "c_mon", "c_fri",
                      "c_tom", "c_preholiday", "c_opex", "c_fomc", "c_fomc_eve", "c_cpi", "c_jobs"]
MIN_TRAIN, MIN_CHOOSE, MIN_WIN, TOP_PAIRS, TOP_CHOOSE, MAX_FAM = 150, 60, 0.60, 300, 500, 10
MIN_DAYS_TRAIN, MIN_DAYS_CHOOSE, MAX_DAY_SHARE = 40, 20, 0.10   # Amendment 3
CHECKS = (690, 840, 955)          # 11:30, 14:00, 15:55 ET (minute of day at which the price is read)
SEED, DRAWS = 192, 200


# ---------- pure helpers (tested) ----------

def wilson_lo(wins, n, z=1.96):
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    return float((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


def bracket(O, H, L, C, sigma, side):
    """Fixed bracket (search labels): entry O[:,0]; stop and take-profit one sigma away; gaps fill at the open,
    the stop first when both are touched; out at the close of the last column. Rows with any NaN -> NaN."""
    n, k = O.shape
    E = O[:, 0]
    sd = float(side)
    S, T = E * (1 - sd * sigma), E * (1 + sd * sigma)
    X = np.full(n, np.nan)
    live = np.isfinite(E) & np.isfinite(sigma) & (sigma > 0)
    for j in range(k):
        o, h, lo, c = O[:, j], H[:, j], L[:, j], C[:, j]
        adv = lo if sd > 0 else h
        fav = h if sd > 0 else lo
        bad = live & ~np.isfinite(c)
        X[bad], live = np.nan, live & ~bad
        g_stop = live & (j > 0) & (sd * (o - S) <= 0)
        X[g_stop] = o[g_stop]
        live &= ~g_stop
        t_stop = live & (sd * (adv - S) <= 0)
        X[t_stop] = S[t_stop]
        live &= ~t_stop
        g_tp = live & (j > 0) & (sd * (o - T) >= 0)
        X[g_tp] = o[g_tp]
        live &= ~g_tp
        t_tp = live & (sd * (fav - T) >= 0)
        X[t_tp] = T[t_tp]
        live &= ~t_tp
        if j == k - 1:
            X[live] = c[live]
    return sd * (X / E - 1) - COST


def ratchet_daily(O, H, L, C, ON, sigma, side):
    """Once-a-day version of the trading rule: stop live (daily low/high, gaps at the open); at each close beyond
    the check level, ratchet when the setup is on (ON[:, j]) else exit; out at the last close. Returns (net, days)."""
    n, k = O.shape
    E = O[:, 0]
    sd = float(side)
    S, T = E * (1 - sd * sigma), E * (1 + sd * sigma)
    X, D = np.full(n, np.nan), np.full(n, k)
    live = np.isfinite(E) & np.isfinite(sigma) & (sigma > 0)
    for j in range(k):
        o, h, lo, c = O[:, j], H[:, j], L[:, j], C[:, j]
        adv = lo if sd > 0 else h
        bad = live & ~np.isfinite(c)
        X[bad], live = np.nan, live & ~bad
        g = live & (j > 0) & (sd * (o - S) <= 0)
        X[g], D[g] = o[g], j + 1
        live &= ~g
        t = live & (sd * (adv - S) <= 0)
        X[t], D[t] = S[t], j + 1
        live &= ~t
        beyond = live & (sd * (c - T) >= 0)
        off = beyond & ~ON[:, j]
        X[off], D[off] = c[off], j + 1
        live &= ~off
        up = beyond & ON[:, j]
        T = np.where(up, c * (1 + sd * sigma), T)
        S = np.where(up, np.where(sd > 0, np.maximum(S, c * (1 - sigma)), np.minimum(S, c * (1 + sigma))), S)
        if j == k - 1:
            X[live], D[live] = c[live], k
    return sd * (X / E - 1) - COST, D


def ratchet_intraday(days, on_prev, on_close, sigma, side):
    """Three-checks-a-day rule on 5-minute bars. days: list (one per trading day, entry day first) of frames with
    hm (minute the bar ends), o, h, l, c; on_prev/on_close: the setup's state at the previous close / this close.
    Returns (net, exit day index) or (nan, None) when the entry day has no bars."""
    if not len(days) or days[0] is None or not len(days[0]):
        return np.nan, None
    sd = float(side)
    E = float(days[0]["o"].iloc[0])
    S, T = E * (1 - sd * sigma), E * (1 + sd * sigma)
    for j, b in enumerate(days):
        if b is None or not len(b):
            continue
        for _, r in b.iterrows():
            adv = r["l"] if sd > 0 else r["h"]
            if sd * (r["o"] - S) <= 0 and not (j == 0 and r["hm"] == b["hm"].iloc[0]):
                return sd * (r["o"] / E - 1) - COST, j
            if sd * (adv - S) <= 0:
                return sd * (S / E - 1) - COST, j
            if r["hm"] in CHECKS:
                p = r["c"]
                last = r["hm"] == CHECKS[-1]
                if sd * (p - T) >= 0:
                    on = on_close[j] if last else on_prev[j]
                    if not on:
                        return sd * (p / E - 1) - COST, j
                    T = p * (1 + sd * sigma)
                    S = max(S, p * (1 - sigma)) if sd > 0 else min(S, p * (1 + sigma))
                if last and j == len(days) - 1:
                    return sd * (p / E - 1) - COST, j
    b = next((x for x in reversed(days) if x is not None and len(x)), None)
    return sd * (float(b["c"].iloc[-1]) / E - 1) - COST, len(days) - 1


def streak(ret, key):
    """Signed count of consecutive up (+) or down (-) days, restarting on a new series."""
    s = np.sign(ret.fillna(0)).to_numpy()
    k = key.to_numpy()
    out = np.zeros(len(s))
    for i in range(len(s)):
        if i and k[i] == k[i - 1] and s[i] != 0 and s[i] == np.sign(out[i - 1]):
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return out


def cut_conditions(df, cuts):
    """Feature frame + {feature: [q10, q25, q75, q90]} -> {name: bool array}."""
    out = {}
    for f, (q10, q25, q75, q90) in cuts.items():
        x = df[f].to_numpy(float)
        out[f"{f}<=p10"], out[f"{f}<=p25"] = x <= q10, x <= q25
        out[f"{f}>=p75"], out[f"{f}>=p90"] = x >= q75, x >= q90
    for f in FLAGS:
        out[f] = df[f].fillna(False).astype(bool).to_numpy()
    return out


def prev_index(df):
    """Row index of the previous row of the same series (-1 when none)."""
    key = df["symbol"].astype(str) + "|" + df["seg"].astype(str)
    same = key.to_numpy()[1:] == key.to_numpy()[:-1]
    p = np.r_[-1, np.arange(len(df) - 1)]
    p[1:][~same] = -1
    return p


def light(on, prev):
    """Light-ups: on today and not on at the previous row of the series."""
    was = np.where(prev >= 0, on[np.maximum(prev, 0)], False)
    return on & ~was


def setup_stats(lu, net, dcode=None):
    """Trade stats of the light-ups; with day codes also distinct days, the busiest day's share and the Wilson lower
    bound on the day count (Amendment 3)."""
    ok = lu & np.isfinite(net)
    x = net[ok]
    n, w = len(x), int((x > 0).sum())
    s = {"n": n, "win": w / n if n else 0.0, "net": float(x.mean()) if n else 0.0, "lo": wilson_lo(w, n)}
    if dcode is not None:
        c = np.bincount(dcode[ok]) if n else np.zeros(1, int)
        days = int((c > 0).sum())
        s.update(days=days, top_share=float(c.max() / n) if n else 1.0,
                 lo=wilson_lo(s["win"] * days, days))
    return s


def spread_ok(s, min_days):
    return s["days"] >= min_days and s["top_share"] <= MAX_DAY_SHARE


# ---------- data stage ----------

def parse_trades(sub_b, own_b, tr_b):
    """Officer/director open-market purchases (P, acquired) and sales (S, disposed) on original Form 4s:
    (cik, filed, code, value)."""
    rd = lambda b, cols: pd.read_csv(io.BytesIO(b), sep="\t", dtype=str, usecols=cols)
    sub = rd(sub_b, ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"]
    own = rd(own_b, ["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP"])
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    acc = set(own.loc[rel.str.contains("OFFICER") | rel.str.contains("DIRECTOR"), "ACCESSION_NUMBER"])
    tr = rd(tr_b, ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                   "TRANS_ACQUIRED_DISP_CD"])
    code, ad = tr["TRANS_CODE"].str.strip(), tr["TRANS_ACQUIRED_DISP_CD"].str.strip()
    tr = tr[((code == "P") & (ad == "A")) | ((code == "S") & (ad == "D"))].copy()
    tr["code"] = tr["TRANS_CODE"].str.strip()
    tr["value"] = pd.to_numeric(tr["TRANS_SHARES"], errors="coerce") * pd.to_numeric(tr["TRANS_PRICEPERSHARE"],
                                                                                       errors="coerce")
    tr = tr[tr["ACCESSION_NUMBER"].isin(acc) & (tr["value"] > 0)].drop_duplicates(["ACCESSION_NUMBER",
                                                                                     "NONDERIV_TRANS_SK"])
    m = tr.merge(sub, on="ACCESSION_NUMBER")
    out = pd.DataFrame({"cik": pd.to_numeric(m["ISSUERCIK"], errors="coerce"),
                        "filed": pd.to_datetime(m["FILING_DATE"], errors="coerce", format="mixed"),
                        "code": m["code"].to_numpy(), "value": m["value"].to_numpy()})
    return out.dropna(subset=["cik", "filed"]).astype({"cik": "int64"})


def insiders(cache):
    d = Path(cache) / "dera"
    d.mkdir(parents=True, exist_ok=True)
    parts = []
    for per in pd.period_range("2018Q4", "2025Q3", freq="Q"):
        f = d / f"trades_{per.year}q{per.quarter}.parquet"
        if not f.exists():
            r = sec.get(DERA.format(y=per.year, q=per.quarter))
            if r is None:
                print(f"[lit1] DERA {per} missing", flush=True)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                pick = lambda stem, z=z: z.read(next(n for n in z.namelist() if n.upper().endswith(stem)))
                x = parse_trades(pick("SUBMISSION.TSV"), pick("REPORTINGOWNER.TSV"), pick("NONDERIV_TRANS.TSV"))
            tmp = f.with_suffix(".tmp")
            x.to_parquet(tmp)
            tmp.replace(f)
        parts.append(pd.read_parquet(f))
    t = pd.concat(parts, ignore_index=True)
    return t[t["filed"] <= DATA_END]


def cross_asset(cache):
    f = Path(cache) / "cross.parquet"
    if not f.exists():
        raw = fetch_daily_bars(CROSS, "2018-01-01", str(DATA_END.date()), "all", alpaca_headers())
        fr = to_frames(raw)
        out = pd.DataFrame({s: fr[s]["c"] for s in CROSS})
        out.to_parquet(f)
    c = pd.read_parquet(f)
    c = c[c.index <= DATA_END]
    out = pd.DataFrame(index=c.index)
    for s in CROSS:
        out[f"{s}_r1"] = c[s].pct_change()
        out[f"{s}_r5"] = c[s].pct_change(5)
    return out


def macro():
    root = Path(__file__).resolve().parents[2] / "research"
    return pd.concat([pd.read_csv(root / "macro_dates_2019_2023.csv"), pd.read_csv(root / "macro_dates_2024_2025.csv")])


def news_frame(w1, cal, first, last):
    rows = []
    for per in pd.period_range(first, last, freq="M"):
        n = W.news_month(w1, per)
        n = n[n["symbols"].fillna("") != ""].copy()
        n["date"] = S2.to_session(W.et_time(n["created_at"]), cal)
        n = n.dropna(subset=["date"])
        h = n["headline"].fillna("").str.lower()
        n["tone"] = S5.tone(n["headline"]).to_numpy()
        for k, pat in S2.TYPES.items():
            n[f"t_{k}"] = h.str.contains(pat, regex=True)
        n["symbol"] = n["symbols"].str.split(",")
        rows.append(n[["symbol", "date", "tone"] + TYPE_FLAGS].explode("symbol"))
    x = pd.concat(rows, ignore_index=True)
    return x.groupby(["symbol", "date"]).agg(news_n=("tone", "size"), tone=("tone", "mean"),
                                              **{c: (c, "max") for c in TYPE_FLAGS}).reset_index()


def roll(d, col, w, fn="sum", mp=1):
    g = d.groupby(["symbol", "seg"], sort=False)[col]
    r = getattr(g.rolling(w, min_periods=mp), fn)()
    return r.reset_index(level=[0, 1], drop=True)


def year_features(w1, y, ctx):
    start = pd.Timestamp(f"{y - 1}-01-01")
    end = min(pd.Timestamp(f"{y + 1}-01-31"), DATA_END)
    d = W.read_rows(w1, ["symbol", "date", "o", "h", "l", "c", "v", "ao", "ac", "dv20", "seg", "cik"],
                    [("date", ">=", start), ("date", "<=", end)])
    d = d[d["v"] > 0].drop_duplicates(["symbol", "date"]).sort_values(["symbol", "seg", "date"]).reset_index(drop=True)
    g = d.groupby(["symbol", "seg"], sort=False)
    inside = (g["c"].shift() >= MIN_PX) & (g["dv20"].shift() >= MIN_DV) & (d["date"].dt.year == y)
    d = d[d["symbol"].isin(set(d.loc[inside, "symbol"]))].reset_index(drop=True)   # stocks in the universe in y
    g = d.groupby(["symbol", "seg"], sort=False)
    f = d["ac"] / d["c"]
    d["ah"], d["al"] = d["h"] * f, d["l"] * f
    d["pc"], d["pdv"], d["pac"] = g["c"].shift(), g["dv20"].shift(), g["ac"].shift()
    d["r1"] = d["ac"] / d["pac"] - 1
    for k in (5, 20, 60):
        d[f"r{k}"] = d["ac"] / g["ac"].shift(k) - 1
    d["hi20"] = d["ac"] / roll(d, "ac", 20, "max", 20) - 1
    d["lo20"] = d["ac"] / roll(d, "ac", 20, "min", 20) - 1
    d["hi250"] = d["ac"] / roll(d, "ac", 250, "max", 120) - 1
    d["vol20"] = roll(d, "r1", 20, "std", 15)
    d["rng"] = (d["h"] - d["l"]) / d["c"]
    med = roll(d, "v", 20, "median", 10)
    d["vr"] = d["v"] / med.groupby([d["symbol"], d["seg"]]).shift()
    d["gap"], d["oc"] = d["ao"] / d["pac"] - 1, d["ac"] / d["ao"] - 1
    d["clv"] = ((d["c"] - d["l"]) / (d["h"] - d["l"])).where(d["h"] > d["l"])
    d["streak"] = streak(d["r1"], d["symbol"].astype(str) + "|" + d["seg"].astype(str))
    d["logp"], d["logdv"] = np.log(d["c"]), np.log(d["dv20"].where(d["dv20"] > 0))
    # next-10-day path (adjusted) and the window's last date
    for j in range(1, HOLD + 1):
        for c, src in (("o", "ao"), ("h", "ah"), ("l", "al"), ("c", "ac")):
            d[f"{c}{j}"] = g[src].shift(-j)
    d["exit_date"] = g["date"].shift(-HOLD)
    d["uni"] = (d["pc"] >= MIN_PX) & (d["pdv"] >= MIN_DV) & (d["pac"] > 0)
    cal = pd.DatetimeIndex(sorted(d["date"].unique()))
    # market and sector (universe rows)
    u = d[d["uni"]]
    m = u.groupby("date")["r1"].agg(mkt1="mean", breadth=lambda s: (s > 0).mean())
    m["mkt5"] = m["mkt1"].rolling(5).sum()
    m["mktvol20"] = m["mkt1"].rolling(20).std()
    d = d.join(m, on="date")
    d["sector"] = [ctx["ff"].get(int(c), 12) if pd.notna(c) else 12 for c in d["cik"]]
    sg = u.assign(sector=d.loc[u.index, "sector"]).groupby(["date", "sector"])["r1"].agg(["sum", "count"])
    sm = (sg["sum"] / sg["count"]).unstack().sort_index()
    s5 = sm.rolling(5, min_periods=3).sum().stack().rename("sec5")
    d = d.join(sg, on=["date", "sector"]).join(s5, on=["date", "sector"])
    own = d["uni"] & d["r1"].notna()
    d["sec1"] = ((d["sum"] - np.where(own, d["r1"], 0)) / (d["count"] - own.astype(int))).where(d["count"] - own > 0)
    d["own_minus_sec"] = d["r1"] - d["sec1"]
    d = d.drop(columns=["sum", "count"])
    # news
    nw = news_frame(w1, cal, f"{y - 1}-11", f"{min(y, 2025)}-{'09' if y == 2025 else '12'}")
    d = d.merge(nw, on=["symbol", "date"], how="left")
    d["news_n"] = d["news_n"].fillna(0)
    d = d.sort_values(["symbol", "seg", "date"]).reset_index(drop=True)
    d["news_n5"] = roll(d, "news_n", 5)
    has = d["news_n"] > 0
    grp = (has | (d["symbol"] != d["symbol"].shift())).cumsum()
    d["days_since_news"] = d.groupby(grp).cumcount().where(~has, 0).clip(upper=60)
    # filings and insiders
    fl = ctx["filings"]
    fl = fl[(fl["filed"] >= cal[0] - pd.Timedelta(days=7)) & (fl["filed"] <= cal[-1])]
    e = S2.earn_days(fl, cal).assign(earn=True)
    ff = S3.filing_flags(fl, cal)[["cik", "date", "k8_other", "f_offer"]]
    d["cik_i"] = d["cik"].astype("Int64")
    d = d.merge(e.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
    d = d.merge(ff.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
    it = ctx["ins"]
    it = it[(it["filed"] >= cal[0]) & (it["filed"] <= cal[-1])].copy()
    it["date"] = cal.values[np.searchsorted(cal.values, it["filed"].values)]
    it["buy"] = np.where(it["code"] == "P", it["value"], 0.0)
    it["sell"] = np.where(it["code"] == "S", it["value"], 0.0)
    ia = it.groupby(["cik", "date"]).agg(buy=("buy", "sum"), sell=("sell", "sum")).reset_index()
    d = d.merge(ia.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
    d[["buy", "sell"]] = d[["buy", "sell"]].fillna(0.0)
    d = d.sort_values(["symbol", "seg", "date"]).reset_index(drop=True)
    d["_b"], d["_s"], d["_net"] = (d["buy"] > 0).astype(float), (d["sell"] > 0).astype(float), d["buy"] - d["sell"]
    d["ins_buy5"] = roll(d, "_b", 5) > 0
    d["ins_bigbuy20"] = roll(d, "buy", 20) >= 1e5
    d["ins_sell5"] = roll(d, "_s", 5) > 0
    d["ins_net20"] = roll(d, "_net", 20)
    # cross-asset and calendar (about day t+1)
    d = d.join(ctx["cross"], on="date")
    cf = S5.calendar_flags(cal, ctx["macro"]).shift(-1).fillna(False).astype(bool)
    d = d.join(cf, on="date")
    for c in FLAGS:
        d[c] = d[c].fillna(False).astype(bool) if c in d else False
    # labels
    O = d[[f"o{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    H = d[[f"h{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    L = d[[f"l{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    C = d[[f"c{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    sig = d["vol20"].to_numpy(float)
    d["net_long"], d["net_short"] = bracket(O, H, L, C, sig, 1), bracket(O, H, L, C, sig, -1)
    d = d[(d["date"].dt.year == y) & d["uni"]]
    keep = (["symbol", "seg", "date", "cik", "pdv", "exit_date", "net_long", "net_short"] + CONT + FLAGS
            + [f"{c}{j}" for c in "ohlc" for j in range(1, HOLD + 1)])
    return d[keep].reset_index(drop=True)


def stage_data(w1, cache, sic_cache):
    out = Path(cache) / "features"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    S2.ensure_inputs(w1, cache, t0 + 3600)
    ciks = set()
    for p in W.batch_files(w1):
        ciks |= {int(c) for c in pd.read_parquet(p, columns=["cik"])["cik"].dropna().unique()}
    ctx = {"ff": industry.industries(ciks, sic_cache).to_dict(), "filings": W.load_filings(w1, ["cik", "form", "filed", "items"]),
           "ins": insiders(cache), "cross": cross_asset(cache), "macro": macro()}
    print(f"[lit1] {len(ctx['ins'])} insider trades, {len(ctx['cross'])} cross-asset days", flush=True)
    for y in YEARS:
        f = out / f"f_{y}.parquet"
        if f.exists():
            continue
        x = year_features(w1, y, ctx)
        tmp = f.with_suffix(".tmp")
        x.astype({c: "float32" for c in x.columns if x[c].dtype == "float64"}).to_parquet(tmp)
        tmp.replace(f)
        print(f"[lit1] {y}: {len(x)} stock-days, {time.monotonic() - t0:.0f} s", flush=True)


# ---------- search / choose ----------

def load(cache, first, last, cols=None, whole_window=True):
    parts = []
    for y in range(first.year, last.year + 1):
        x = pd.read_parquet(Path(cache) / "features" / f"f_{y}.parquet", columns=cols)
        x = x[(x["date"] >= first) & (x["date"] <= last)]
        parts.append(x[x["exit_date"] <= last] if whole_window else x)
    d = pd.concat(parts, ignore_index=True).sort_values(["symbol", "seg", "date"]).reset_index(drop=True)
    return d


BASE_COLS = ["symbol", "seg", "date", "exit_date", "net_long", "net_short"]


def stage_search(cache):
    d = load(cache, pd.Timestamp("2019-01-01"), TRAIN_END, BASE_COLS + CONT + FLAGS)
    assert d["exit_date"].max() <= TRAIN_END
    cuts = {f: [float(np.nanquantile(d[f], q)) for q in QS] for f in CONT}
    cond = cut_conditions(d, cuts)
    cond = {k: v for k, v in cond.items() if v.mean() >= 0.01}
    names = sorted(cond)
    A = np.asfortranarray(np.column_stack([cond[k] for k in names]))
    prev = prev_index(d)
    P = np.asfortranarray(np.where((prev >= 0)[:, None], A[np.maximum(prev, 0)], False))
    nets = {1: d["net_long"].to_numpy(float), -1: d["net_short"].to_numpy(float)}
    dcode = pd.factorize(d["date"])[0]
    print(f"[lit1] search: {len(d)} rows, {len(names)} conditions", flush=True)
    res = []

    def score(idx):
        on = A[:, idx[0]].copy()
        was = P[:, idx[0]].copy()
        for i in idx[1:]:
            on &= A[:, i]
            was &= P[:, i]
        lu = on & ~was
        for side, net in nets.items():
            s = setup_stats(lu, net, dcode)
            res.append({"conds": [names[i] for i in idx], "side": side, **s})
        return res[-2:]
    t0 = time.monotonic()
    pair_scores = {1: [], -1: []}
    for i in range(len(names)):
        score((i,))
    for i, j in combinations(range(len(names)), 2):
        for r in score((i, j)):
            if r["n"] >= MIN_TRAIN and spread_ok(r, MIN_DAYS_TRAIN):
                pair_scores[r["side"]].append((r["lo"], i, j))
    print(f"[lit1] pairs done {time.monotonic() - t0:.0f} s", flush=True)
    for side in (1, -1):
        top = sorted(pair_scores[side], reverse=True)[:TOP_PAIRS]
        for _, i, j in top:
            for k in range(len(names)):
                if k in (i, j):
                    continue
                on = A[:, i] & A[:, j] & A[:, k]
                was = P[:, i] & P[:, j] & P[:, k]
                s = setup_stats(on & ~was, nets[side], dcode)
                res.append({"conds": sorted([names[i], names[j], names[k]]), "side": side, **s})
    print(f"[lit1] triples done {time.monotonic() - t0:.0f} s, {len(res)} scored", flush=True)
    keep = [r for r in res if r["n"] >= MIN_TRAIN and r["win"] >= MIN_WIN and r["net"] > 0
            and spread_ok(r, MIN_DAYS_TRAIN)]
    uniq = {}
    for r in keep:
        uniq.setdefault((tuple(sorted(r["conds"])), r["side"]), r)
    keep = sorted(uniq.values(), key=lambda r: -r["lo"])
    summary = {"rows": len(d), "conditions": len(names), "scored": len(res), "kept": len(keep),
               "best_win": max((r["win"] for r in res if r["n"] >= MIN_TRAIN), default=None),
               "share_kept": len(keep) / max(len(res), 1)}
    (Path(cache) / "candidates.json").write_text(json.dumps({"cuts": cuts, "names": names, "candidates": keep[:5000],
                                                              "summary": summary}, default=float))
    print(f"[lit1] search summary {summary}", flush=True)
    # QA (Amendment 3 check): base rates and how many distinct days the leading candidates light up on
    for side, net in nets.items():
        x = net[np.isfinite(net)]
        print(f"[lit1] qa base side {side}: n {len(x)} win {(x > 0).mean():.3f} net {x.mean():.4f}", flush=True)
    dates = d["date"].to_numpy()
    idx = {k: i for i, k in enumerate(names)}
    for r in keep[:25]:
        ii = [idx[c] for c in r["conds"]]
        on, was = A[:, ii].all(axis=1), P[:, ii].all(axis=1)
        lu = (on & ~was) & np.isfinite(nets[r["side"]])
        days = pd.Series(dates[lu]).value_counts()
        print(f"[lit1] qa top {r['conds']} side {r['side']} n {r['n']} win {r['win']:.3f} net {r['net']:.4f} "
              f"days {len(days)} biggest-day share {days.iloc[0] / days.sum():.2f} lo {r['lo']:.3f}", flush=True)
    return summary


def setup_on(df, cuts, conds):
    c = cut_conditions(df, {k: v for k, v in cuts.items() if any(x.startswith((k + "<", k + ">"))
                                                                 for x in conds)})
    on = np.ones(len(df), bool)
    for k in conds:
        on &= c[k]
    return on


def stage_choose(cache, out):
    cand = json.loads((Path(cache) / "candidates.json").read_text())
    d = load(cache, pd.Timestamp("2022-01-01"), CHOOSE_END, BASE_COLS + CONT + FLAGS)
    assert d["date"].min() >= pd.Timestamp("2022-01-01") and d["exit_date"].max() <= CHOOSE_END
    prev = prev_index(d)
    dcode = pd.factorize(d["date"])[0]
    rows = []
    for r in cand["candidates"][:TOP_CHOOSE]:
        on = setup_on(d, cand["cuts"], r["conds"])
        lu = light(on, prev)
        s = setup_stats(lu, d["net_long" if r["side"] == 1 else "net_short"].to_numpy(float), dcode)
        rows.append({**r, "choose": s, "_lu": np.flatnonzero(lu)})
    ok = [r for r in rows if r["choose"]["n"] >= MIN_CHOOSE and r["choose"]["win"] >= MIN_WIN
          and r["choose"]["net"] > 0 and spread_ok(r["choose"], MIN_DAYS_CHOOSE)]
    ok.sort(key=lambda r: -r["choose"]["lo"])
    picked = []
    for r in ok:
        s = set(r["_lu"])
        if all(len(s & p["_set"]) / max(len(s | p["_set"]), 1) <= 0.5 for p in picked):
            picked.append({**r, "_set": s})
        if len(picked) >= MAX_FAM:
            break
    fams = [{k: v for k, v in p.items() if not k.startswith("_")} for p in picked]
    res = {"search": cand["summary"], "evaluated": len(rows), "passed_choose": len(ok), "families": fams,
           "cuts": cand["cuts"]}
    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = Path(out) / "LIT1_families.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(Path(out) / "LIT1_families.json")
    print(f"[lit1] choose: {len(ok)} of {len(rows)} passed, {len(fams)} families", flush=True)
    for f in fams:
        print(f"  {f['side']:+d} {f['conds']} train {f['win']:.2f}/{f['n']} choose {f['choose']['win']:.2f}/"
              f"{f['choose']['n']} net {f['choose']['net']:.4f}", flush=True)


# ---------- test ----------

def bars5(sym, start, end, h):
    rows, tok = [], None
    while True:
        prm = {"symbols": sym, "timeframe": "5Min", "start": start, "end": end, "feed": "sip", "adjustment": "raw",
               "limit": 10000}
        if tok:
            prm["page_token"] = tok
        for k in range(6):
            r = requests.get(S4.URL, headers=h, params=prm, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * (k + 1))
                continue
            r.raise_for_status()
            break
        js = r.json()
        rows += js.get("bars", {}).get(sym, []) or []
        tok = js.get("next_page_token")
        if not tok:
            break
    b = pd.DataFrame(rows)
    if not len(b):
        return {}
    b = b[b["v"] > 0]
    et = pd.to_datetime(b["t"], utc=True).dt.tz_convert(ET)
    b["date"] = et.dt.tz_localize(None).dt.normalize()
    b["hm"] = et.dt.hour * 60 + et.dt.minute + 5
    b = b[(b["hm"] > 570) & (b["hm"] <= 960)]
    return {d: g.sort_values("hm")[["hm", "o", "h", "l", "c"]].reset_index(drop=True) for d, g in b.groupby("date")}


def stage_test(cache, out):
    fam = json.loads((Path(out) / "LIT1_families.json").read_text())
    if not fam["families"]:
        res = {"verdict": "FAIL", "reason": "no family passed the 2022-2023 choice; test period not opened"}
        (Path(out) / "LIT1_test.json").write_text(json.dumps(res, indent=1))
        print("[lit1] no families", flush=True)
        return
    d = load(cache, TEST_START, DATA_END, whole_window=False)
    assert d["date"].min() >= TEST_START and d["date"].max() <= DATA_END
    prev = prev_index(d)
    cal = pd.DatetimeIndex(sorted(d["date"].unique()))
    pos = {dt: i for i, dt in enumerate(cal)}
    ons = [setup_on(d, fam["cuts"], f["conds"]) for f in fam["families"]]
    whole = (d["exit_date"] <= DATA_END).to_numpy()       # the trade's 10-day window ends inside the period
    ev = []
    for k, (f, on) in enumerate(zip(fam["families"], ons, strict=True)):
        for i in np.flatnonzero(light(on, prev) & whole):
            if pos[d.at[i, "date"]] + HOLD < len(cal):
                ev.append((d.at[i, "date"], k, i, f["side"]))
    ev.sort()
    key = d["symbol"].astype(str).to_numpy()
    rowpos = {(key[i], d.at[i, "date"]): i for i in range(len(d))}
    h = alpaca_headers()
    busy, trades = {}, []
    O = d[[f"o{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    H = d[[f"h{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    L = d[[f"l{j}" for j in range(1, HOLD + 1)]].to_numpy(float)
    C = d[[f"c{j}" for j in range(1, HOLD + 1)]].to_numpy(float)

    def on_path(k, i):
        """The family's state at the close of t+1 .. t+10 (False where the stock has no row)."""
        t = pos[d.at[i, "date"]]
        return np.array([bool(ons[k][j]) if (j := rowpos.get((key[i], cal[t + m]))) is not None else False
                         for m in range(1, HOLD + 1)])
    t0 = time.monotonic()
    for dt, k, i, side in ev:
        if busy.get(key[i], pd.Timestamp.min) >= dt:
            continue
        t = pos[dt]
        days = list(cal[t + 1:t + 1 + HOLD])
        st = on_path(k, i)
        b = bars5(key[i], f"{days[0].date()}T13:00:00Z", f"{days[-1].date()}T21:00:00Z", h)
        dayb = [b.get(x) for x in days]
        on_prev = np.r_[True, st[:-1]]
        net, j = ratchet_intraday(dayb, on_prev, st, float(d.at[i, "vol20"]), side)
        if j is None:
            continue
        nd, dd = ratchet_daily(O[i:i + 1], H[i:i + 1], L[i:i + 1], C[i:i + 1], st[None, :], d["vol20"].to_numpy(float)[i:i + 1], side)
        busy[key[i]] = days[j]
        trades.append({"date": dt, "exit": days[j], "symbol": key[i], "family": k, "side": side, "net": net,
                       "net_daily": float(nd[0]), "net_fixed": float(d.at[i, "net_long" if side == 1 else "net_short"])})
    tr = pd.DataFrame(trades)
    print(f"[lit1] test: {len(tr)} trades, {time.monotonic() - t0:.0f} s", flush=True)
    # random control on the once-a-day exit
    rng = np.random.default_rng(SEED)
    dd = d["date"].to_numpy()
    by_day = {dt: np.flatnonzero((dd == np.datetime64(dt)) & whole) for dt in tr["date"].unique()}
    sig = d["vol20"].to_numpy(float)
    draws = []
    for _ in range(DRAWS):
        nets = []
        for (dt, side, k), grp in tr.groupby(["date", "side", "family"]):
            pool = by_day[dt]
            pick = rng.choice(pool, size=len(grp), replace=len(pool) < len(grp))
            st = np.array([on_path(k, i) for i in pick])
            nd, _ = ratchet_daily(O[pick], H[pick], L[pick], C[pick], st, sig[pick], side)
            nets.append(nd)
        draws.append(float(np.nanmean(np.concatenate(nets))))
    # stats
    daily = tr.groupby("date")["net"].mean()
    fac = french.parse(french.fetch("ff5_daily", cache)[0], "D").join(
        french.parse(french.fetch("mom_daily", cache)[0], "D"), how="inner")
    fm = fac.groupby(fac.index.to_period("M")).sum()
    fm.columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"][:len(fm.columns)]
    ff5 = fm[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    mom = fm[["MOM"]]
    tr["exit_m"] = pd.to_datetime(tr["exit"]).dt.to_period("M")
    mon = tr.groupby(tr["exit_m"])["net"].sum() / tr.groupby(tr["exit_m"])["net"].size().clip(lower=1)
    alpha = factor_alpha(mon.to_numpy(), mon.index, ff5, mom, 3) if len(mon) >= 6 else None

    def summ(x):
        return {"trades": len(x), "win": float((x["net"] > 0).mean()) if len(x) else None,
                "net": float(x["net"].mean()) if len(x) else None,
                "t_day": S4.nw_t(x.groupby("date")["net"].mean().to_numpy()) if len(x) else None,
                "win_daily_exit": float((x["net_daily"] > 0).mean()) if len(x) else None,
                "net_daily_exit": float(x["net_daily"].mean()) if len(x) else None,
                "net_fixed_exit": float(x["net_fixed"].mean()) if len(x) else None}
    allp = summ(tr)
    rand95 = float(np.percentile(draws, 95))
    checks = {"win_ge_60": (allp["win"] or 0) >= 0.60, "net_t_ge_3": (allp["net"] or 0) > 0 and (allp["t_day"] or 0) >= 3,
              "trades_ge_100": len(tr) >= 100, "beats_random": (allp["net_daily_exit"] or -1) > rand95,
              "alpha": bool(alpha and alpha["alpha"] > 0 and alpha["t"] >= 2)}
    res = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "all": allp,
           "trades_per_day": len(tr) / len(cal), "random_daily_exit_mean": float(np.mean(draws)), "random_p95": rand95,
           "alpha": alpha, "dsr": deflated_sharpe(daily.to_numpy(), 207) if len(daily) > 20 else None,
           "families": {str(k): summ(tr[tr["family"] == k]) for k in range(len(fam["families"]))},
           "longs": summ(tr[tr["side"] == 1]), "shorts": summ(tr[tr["side"] == -1]),
           "years": {str(y): summ(tr[pd.to_datetime(tr["date"]).dt.year == y]) for y in (2024, 2025)},
           "reached_70": (allp["win"] or 0) >= 0.70}
    tmp = Path(out) / "LIT1_test.tmp"
    tmp.write_text(json.dumps(res, indent=1, default=str))
    tmp.replace(Path(out) / "LIT1_test.json")
    tr.drop(columns=["exit_m"]).to_csv(Path(out) / "LIT1_trades.csv", index=False)
    print(json.dumps({k: res[k] for k in ("verdict", "checks", "all", "trades_per_day", "random_p95")}, default=str),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "search", "choose", "test"])
    ap.add_argument("--w1", default="cache/w1")
    ap.add_argument("--cache", default="cache/lit1")
    ap.add_argument("--out", default="research/results/LIT1")
    ap.add_argument("--sic", default="cache/sector1", help="SECTOR1's cache, which holds the SIC codes")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.w1, a.cache, a.sic)
    elif a.stage == "search":
        stage_search(a.cache)
    elif a.stage == "choose":
        stage_choose(a.cache, a.out)
    else:
        stage_test(a.cache, a.out)


if __name__ == "__main__":
    main()
