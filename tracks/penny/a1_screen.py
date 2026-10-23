"""A1 historical screen (research/prereg/A1.md): short penny stocks whose new 10-K gives a Beneish M-score
above -1.78, hold 63 trading days with a 35% stop, against a random penny short control.

Each 10-K is scored from its own XBRL facts (this year and the prior-year comparatives in the same filing),
read from the SEC companyfacts bulk zip in 500-company batches. Bars are fetched in ~500-symbol batches and
reduced at once to month-end rows with the short trade's outcome. Both steps are cached per batch.

Usage: python -m tracks.penny.a1_screen --cache <dir> --out <dir>
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core import alpaca, stats
from core.bars import fetch_daily_bars, to_frames
from core.env import alpaca_headers
from tracks.main import weekly_panel
from tracks.main.longterm import bars_batch, sec_data
from tracks.main.longterm.prices import segment_ids
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

BARS_START, BARS_END = "2016-01-01", "2025-09-30"
FIRST_T, LABEL_END = pd.Timestamp("2016-12-30"), pd.Timestamp("2025-10-01")
COST, BORROW, STOP, HOLD = 0.02, 0.20, 0.35, 63
M_FLAG, MIN_SALES, SEED, N_BOOT, LAGS, BLOCK = -1.78, 1e6, 20261025, 10_000, 3, 3
ANNUAL_FORMS = ("10-K", "10-KT")

FLOW = {
    "sales": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
              "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gp": ["GrossProfit"],
    "dep": ["Depreciation", "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "inc": ["IncomeLossFromContinuingOperations", "NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
INSTANT = {
    "rec": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "ca": ["AssetsCurrent"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "ta": ["Assets"],
    "cl": ["LiabilitiesCurrent"],
    "ltd": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
WEIGHTS = {"dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115, "sgai": -0.172,
           "tata": 4.679, "lvgi": -0.327}


def filing_facts(cf):
    """companyfacts JSON -> rows (cik, accn, filed, key, end, days, val) from 10-K/10-KT filings for the
    concepts above; one concept per key (the first in its list that the filing reports)."""
    cik = int(cf.get("cik", 0))
    gaap = (cf.get("facts") or {}).get("us-gaap") or {}
    rows = []
    for key, concepts in {**FLOW, **INSTANT}.items():
        for prio, c in enumerate(concepts):
            for f in (gaap.get(c) or {}).get("units", {}).get("USD", []):
                if f.get("form") in ANNUAL_FORMS:
                    rows.append((cik, f.get("accn"), f["filed"], key, prio, f.get("start"), f["end"], f["val"]))
    df = pd.DataFrame(rows, columns=["cik", "accn", "filed", "key", "prio", "start", "end", "val"])
    for c in ("filed", "start", "end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["days"] = (df["end"] - df["start"]).dt.days
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype(float)
    return df


def _ratio(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def _parts(d):
    """Per-year ratios the indexes compare: gross margin, asset quality, depreciation rate, leverage."""
    g = {k: d.get(k, np.nan) for k in ("sales", "cogs", "gp", "ca", "ppe", "ta", "dep", "cl", "ltd")}
    gp = g["gp"] if np.isfinite(g["gp"]) else g["sales"] - g["cogs"]
    return {"gm": _ratio(gp, g["sales"]), "aq": 1 - _ratio(g["ca"] + g["ppe"], g["ta"]),
            "dp": _ratio(g["dep"], g["dep"] + g["ppe"]), "lv": _ratio(g["cl"] + g["ltd"], g["ta"])}


def mscore(cur, pri):
    """Beneish 8-variable M-score from this year's (cur) and last year's (pri) values (dicts; missing =
    NaN). Returns (M or NaN, indexes, names of indexes set to 1). DSRI, SGI and TATA are required."""
    c, p = {k: cur.get(k, np.nan) for k in ("sales", "rec", "sga", "inc", "cfo", "ta")}, \
        {k: pri.get(k, np.nan) for k in ("sales", "rec", "sga")}
    pc, pp = _parts(cur), _parts(pri)
    ix = {"dsri": _ratio(_ratio(c["rec"], c["sales"]), _ratio(p["rec"], p["sales"])),
          "gmi": _ratio(pp["gm"], pc["gm"]),
          "aqi": _ratio(pc["aq"], pp["aq"]),
          "sgi": _ratio(c["sales"], p["sales"]),
          "depi": _ratio(pp["dp"], pc["dp"]),
          "sgai": _ratio(_ratio(c["sga"], c["sales"]), _ratio(p["sga"], p["sales"])),
          "tata": _ratio(c["inc"] - c["cfo"], c["ta"]),
          "lvgi": _ratio(pc["lv"], pp["lv"])}
    if not all(np.isfinite(ix[k]) for k in ("dsri", "sgi", "tata")):
        return np.nan, ix, []
    neutral = [k for k, v in ix.items() if not np.isfinite(v)]
    for k in neutral:
        ix[k] = 1.0
    return float(-4.84 + sum(WEIGHTS[k] * v for k, v in ix.items())), ix, neutral


def score_filings(df):
    """One row per 10-K: cik, accn, filed, fy_end, sales, M, the 8 indexes and which were set to 1."""
    out = []
    for (cik, accn), g in df.dropna(subset=["accn", "end"]).groupby(["cik", "accn"]):
        ta = g[(g["key"] == "ta")]
        if ta.empty:
            continue
        E = ta["end"].max()
        P = E - pd.Timedelta(days=365)
        vals = {}
        for when, target in (("cur", E), ("pri", P)):
            near = (g["end"] - target).abs() <= pd.Timedelta(days=20)
            flow = g["key"].isin(FLOW) & g["days"].between(330, 400)
            inst = g["key"].isin(INSTANT)
            h = g[near & (flow | inst)].sort_values(["key", "prio"])
            vals[when] = h.groupby("key")["val"].first().to_dict()
        s1, s0 = vals["cur"].get("sales", np.nan), vals["pri"].get("sales", np.nan)
        row = {"cik": int(cik), "accn": accn, "filed": g["filed"].min(), "fy_end": E, "sales": s1}
        if not (np.isfinite(s1) and np.isfinite(s0)) or min(s1, s0) < MIN_SALES:
            out.append({**row, "m": np.nan, "why": "sales missing or below $1M"})
            continue
        m, ix, neutral = mscore(vals["cur"], vals["pri"])
        out.append({**row, "m": m, **ix, "neutral": ",".join(neutral),
                    "why": "" if np.isfinite(m) else "DSRI, SGI or TATA missing"})
    return pd.DataFrame(out)


def scores(zip_path, ciks, cache_dir):
    """Scored 10-Ks for `ciks`, 500 companies per cached batch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ciks = sorted({int(c) for c in ciks})
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for b in range(0, len(ciks), 500):
            f = cache_dir / f"m_{b // 500:04d}.parquet"
            if not f.exists():
                rows = [score_filings(filing_facts(json.loads(z.read(n)))) for c in ciks[b:b + 500]
                        if (n := f"CIK{c:010d}.json") in names]
                part = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) \
                    else pd.DataFrame(columns=["cik", "accn", "filed", "m"])
                tmp = f.with_suffix(".tmp")
                part.to_parquet(tmp)
                tmp.replace(f)
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def short_trade(d, seg, i, cal, k):
    """Short at the adjusted open of bar i+1, out at the first close >= STOP above entry or the close of
    the HOLD-th trading day from entry (cal[k + HOLD]), or the series' last close. Returns (gross short
    return, trading days held, stopped)."""
    if i + 1 >= len(d) or seg[i + 1] != seg[i]:
        return np.nan, 0, False
    entry = d["o"].iloc[i + 1]
    end = cal[k + HOLD]
    x = int(d.index.searchsorted(end, side="right"))
    path = d.iloc[i + 1:x]
    path = path[seg[i + 1:x] == seg[i]]
    hit = np.flatnonzero(path["c"].to_numpy() >= entry * (1 + STOP))
    j = hit[0] if len(hit) else len(path) - 1
    held = int(cal.searchsorted(path.index[j]) - k)
    return float(1 - path["c"].iloc[j] / entry), held, bool(len(hit))


def symbol_rows(sym, raw, adj, cal, dates):
    """Month-end rows (P1 penny universe fields) with the short trade's outcome."""
    d = pd.DataFrame({"o": adj["o"], "c": adj["c"]}).join(raw[["c", "v"]].rename(columns={"c": "rc"}), how="inner")
    d = d[(d["v"] > 0) & (d["c"] > 0) & (d["rc"] > 0)].sort_index()
    if d.empty:
        return []
    seg = segment_ids(d.index)
    nb = pd.Series(seg, index=d.index).groupby(seg).cumcount().to_numpy() + 1
    dv = d["rc"].to_numpy() * d["v"].to_numpy()
    pos = {t: n for n, t in enumerate(d.index)}
    out = []
    for t in dates:
        i = pos.get(t)
        if i is None or nb[i] < 60:
            continue
        price, dv20 = float(d["rc"].iloc[i]), float(np.median(dv[i - 19:i + 1]))
        if not (1 <= price < 5 and dv20 >= 5e5):
            continue
        gross, held, stopped = short_trade(d, seg, i, cal, int(cal.searchsorted(t)))
        out.append({"symbol": sym, "date": t, "price": price, "dv20": dv20, "gross": gross, "held": held,
                    "stopped": stopped})
    return out


def month_ends(cal):
    cal = pd.DatetimeIndex(cal)
    s = pd.Series(cal, index=cal)
    me = s.groupby(s.index.to_period("M")).max()
    keep = []
    for t in me:
        k = cal.searchsorted(t)
        if t >= FIRST_T and k + HOLD < len(cal) and cal[k + HOLD] < LABEL_END:
            keep.append(t)
    return keep


def net(df):
    return df["gross"] - COST - BORROW * df["held"] / 252


def pick(rows, flags, dates):
    """Per month: flagged universe names (a 10-K filed in the 3 months up to t with M > M_FLAG, entering
    at the first month-end it is in the universe) and a control from the rest, default_rng(SEED + month)."""
    f = flags[flags["m"] > M_FLAG][["cik", "accn", "filed", "m"]]
    u = rows.dropna(subset=["cik"]).astype({"cik": "int64"})
    c = u.merge(f, on="cik")
    c = c[(c["filed"] <= c["date"]) & (c["filed"] > c["date"] - pd.DateOffset(months=3))]
    c = c.sort_values(["date", "symbol"]).drop_duplicates("accn").drop_duplicates(["date", "cik"])
    out = []
    for w, t in enumerate(dates):
        fl = c[c["date"] == t]
        if fl.empty:
            continue
        rest = rows[(rows["date"] == t) & ~rows["symbol"].isin(fl["symbol"])].sort_values("symbol").reset_index(drop=True)
        k = np.random.default_rng(SEED + w).choice(len(rest), size=min(len(fl), len(rest)), replace=False)
        out.append(fl.assign(role="flag", month=w))
        out.append(rest.iloc[np.sort(k)].assign(role="control", month=w))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def monthly(sel, with_costs=True):
    s = sel.dropna(subset=["gross"]).assign(net=lambda d: net(d) if with_costs else d["gross"])
    t = s.pivot_table(index="date", columns="role", values="net", aggfunc="mean")
    t["n_flags"] = s[s["role"] == "flag"].groupby("date").size()
    t["diff"] = t["flag"] - t["control"]
    return t.dropna(subset=["diff"])


def summary(t):
    m, se, tt = nw_tstat(t["diff"], LAGS)
    lo, hi = block_bootstrap_ci(t["diff"], BLOCK, N_BOOT, SEED)
    return {"months": int(len(t)), "mean": m, "se": se, "t_hac": tt, "ci95": [lo, hi],
            "mean_n_flags": float(t["n_flags"].mean())}


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    h = alpaca_headers()
    spy = to_frames(fetch_daily_bars(["SPY"], BARS_START, alpaca.sip_end(BARS_END), adjustment="all", headers=h))["SPY"]
    cal = pd.DatetimeIndex(spy.index)
    dates = month_ends(cal)
    assets = bars_batch.assets_all()
    syms = sorted(set(assets.loc[assets["excl_reason"].isna(), "base_symbol"]))
    (cache / "bars").mkdir(exist_ok=True)
    for b in range(0, len(syms), 500):
        f = cache / "bars" / f"a1_{b // 500:04d}.parquet"
        if f.exists():
            continue
        batch, end = syms[b:b + 500], alpaca.sip_end(BARS_END)
        raw = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="raw", headers=h))
        adj = to_frames(fetch_daily_bars(batch, BARS_START, end, adjustment="all", headers=h))
        rows = [r for s in batch if s in raw and s in adj for r in symbol_rows(s, raw[s], adj[s], cal, dates)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp)
        tmp.replace(f)
        print(f"[a1] bars batch {b // 500 + 1}/{-(-len(syms) // 500)}: {len(rows)} rows", flush=True)
    rows = pd.concat([pd.read_parquet(p) for p in sorted((cache / "bars").glob("a1_*.parquet"))], ignore_index=True)
    tmap = sec_data.dera_ticker_map(2015, pd.Period(dates[-1], "Q"), cache / "dera")
    rows["cik"] = sec_data.map_ciks(rows[["symbol", "date"]], tmap)
    funnel = {"penny_month_rows": int(len(rows)), "with_cik": int(rows["cik"].notna().sum())}
    zp = cache / "companyfacts.zip"
    if not zp.exists():
        sec_data.download_companyfacts(zp)
    sc = scores(zp, rows["cik"].dropna().unique(), cache / "mscore")
    funnel.update({"10ks": int(len(sc)), "scored": int(sc["m"].notna().sum()),
                   "flagged": int((sc["m"] > M_FLAG).sum()),
                   "not_scored": sc.loc[sc["m"].isna(), "why"].value_counts().to_dict()})
    print("FUNNEL", funnel, flush=True)
    fac, sha5, shau = weekly_panel.factors()
    p2, both = stats.idea_counts()
    sel = pick(rows, sc, dates)
    t = monthly(sel)
    prim = summary(t)
    wins = [cal[cal.searchsorted(d) + 1:cal.searchsorted(d) + 1 + HOLD] for d in t.index]
    a = stats.ols_hac(t["diff"].to_numpy(), stats.window_factors(fac, wins), LAGS)
    a["factors"] = list(fac.columns) + ["(summed over the 63-day window)"]
    prim["factor_alpha"] = a
    prim["deflated_sharpe"] = stats.deflated_sharpe(t["diff"].to_numpy(), both)
    prim["passes"] = stats.screen_pass(prim, a)
    flags = sel[sel["role"] == "flag"]
    acc = sc.dropna(subset=["tata"]).copy()
    acc["top_tata"] = acc.groupby(acc["filed"].dt.year)["tata"].transform(lambda x: x >= x.quantile(0.9))
    acc_sel = pick(rows, acc.assign(m=np.where(acc["top_tata"], 0.0, -9.0)), dates)
    res = {"idea": "A1", "prereg": "research/prereg/A1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_daily_sha256": sha5, "umd_daily_sha256": shau, "costs": {"round_trip": COST, "borrow_per_year": BORROW},
           "funnel": funnel, "borrow_note": "past easy-to-borrow status unknown; PROTOCOL rule 6 applies forward only",
           "PRIMARY": prim,
           "alongside": {
               "before_costs_and_borrow": summary(monthly(sel, with_costs=False)),
               "accruals_top_decile_by_filing_year": summary(monthly(acc_sel)) if len(acc_sel) else {},
               "flags_stopped_share": float(flags["stopped"].mean()) if len(flags) else None,
               "indexes_set_to_1": sc.loc[sc["m"] > M_FLAG, "neutral"].str.split(",").explode().value_counts().to_dict(),
               "per_year": t.groupby(t.index.year)["diff"].mean().to_dict(),
               "easy_to_borrow_share": "read from Alpaca's asset list when the report is written"}}
    out.mkdir(parents=True, exist_ok=True)
    chk = flags.sample(n=min(20, len(flags)), random_state=SEED)
    chk = chk.merge(sc.drop(columns=["cik", "filed", "m"]), on="accn", how="left")
    cols = ["date", "symbol", "cik", "accn", "filed", "fy_end", "m", "dsri", "gmi", "aqi", "sgi", "depi", "sgai",
            "tata", "lvgi", "neutral"]
    chk[cols].assign(checked_by="", ok="", note="").to_csv(out / "A1_flags_check.csv", index=False)
    sel[["date", "month", "symbol", "role", "m", "gross", "held", "stopped"]].to_csv(out / "A1_picks.csv", index=False)
    (out / "A1_result.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
