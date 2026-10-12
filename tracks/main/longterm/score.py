"""M4 universe filter, factor families and composite (research/prereg/M4.md). Pure functions."""
import numpy as np
import pandas as pd

from tracks.main.longterm.fundamentals import STALE_DAYS, annual_records, as_of, share_records

MIN_PRICE = 5.0
MIN_MCAP = 300e6
MIN_DV20 = 2e6
MIN_FAMILIES = 4
TOP_N = 20
SPLIT_WINDOW = pd.Timedelta(days=183)
FAMILIES = ["f_mom", "f_lowvol", "f_quality", "f_value", "f_issuance", "f_assetgrowth"]
FAMILY_NAMES = {"f_mom": "momentum", "f_lowvol": "low volatility", "f_quality": "quality",
                "f_value": "value", "f_issuance": "buybacks vs dilution", "f_assetgrowth": "asset growth"}


def r_at(rtab, symbols, dates):
    """R (raw / split-adjusted close) for each (symbol, date): the step function in rtab, back-filled
    before a symbol's first bar."""
    q = pd.DataFrame({"symbol": list(symbols), "d": pd.to_datetime(list(dates)).astype("datetime64[ns]")})
    q["_row"] = np.arange(len(q))
    ok = q.dropna(subset=["d"]).sort_values("d")
    rt = rtab.rename(columns={"date": "d"})
    rt = rt.assign(d=rt["d"].astype("datetime64[ns]")).sort_values("d")
    m = pd.merge_asof(ok, rt, on="d", by="symbol", direction="backward")
    first = rtab.sort_values("date").groupby("symbol")["R"].first()
    m["R"] = m["R"].fillna(m["symbol"].map(first))
    return q[["_row"]].merge(m[["_row", "R"]], on="_row", how="left")["R"].to_numpy()


def attach_fundamentals(panel, facts, rtab):
    """Adds market cap and the fundamental ratios to price-feature rows that carry a `cik`."""
    p = panel.reset_index(drop=True)
    ann = annual_records(facts)
    sh = share_records(facts)
    a = as_of(p[["cik", "date"]], ann, stale_days=STALE_DAYS, end_col="end")
    s_now = as_of(p[["cik", "date"]], sh)
    prev_keys = p[["cik"]].assign(date=p["date"] - pd.Timedelta(days=365))
    s_old = as_of(prev_keys, sh)
    r_t = r_at(rtab, p["symbol"], p["date"])
    r_now = r_at(rtab, p["symbol"], s_now["sh_end"])
    r_old = r_at(rtab, p["symbol"], s_old["sh_end"])
    shares_t = s_now["shares"].to_numpy(dtype=float) * r_now / r_t
    shares_old_t = s_old["shares"].to_numpy(dtype=float) * r_old / r_t
    out = p.copy()
    out["mcap"] = out["price"].to_numpy() * shares_t
    # pre-registered: net issuance is undefined when a split happened within 6 months of either share date
    near = np.zeros(len(p), dtype=bool)
    for col in (s_now["sh_end"], s_old["sh_end"]):
        lo = r_at(rtab, p["symbol"], col - SPLIT_WINDOW)
        hi = r_at(rtab, p["symbol"], col + SPLIT_WINDOW)
        near |= ~np.isclose(lo, hi, rtol=0.01) | np.isnan(lo) | np.isnan(hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        iss = np.log(shares_t / shares_old_t)
    iss[near | ~np.isfinite(iss)] = np.nan
    out["issuance"] = iss
    assets = a["assets"].to_numpy(dtype=float)
    pos_assets = np.where(assets > 0, assets, np.nan)
    out["gpa"] = a["gp"].to_numpy(dtype=float) / pos_assets
    out["roa"] = a["ni"].to_numpy(dtype=float) / pos_assets
    out["ep"] = a["ni"].to_numpy(dtype=float) / out["mcap"]
    out["bm"] = a["equity"].to_numpy(dtype=float) / out["mcap"]
    prev = a["assets_prev"].to_numpy(dtype=float)
    out["assetgrowth"] = assets / np.where(prev > 0, prev, np.nan) - 1
    out["fy_end"] = a["end"].to_numpy()
    return out


def eligible(df):
    """Pre-registered filters on t, then one row per company per date (highest 20-day dollar volume)."""
    m = (df["price"] >= MIN_PRICE) & (df["mcap"] >= MIN_MCAP) & (df["dv20"] >= MIN_DV20)
    d = df[m].sort_values("dv20", ascending=False)
    return d.drop_duplicates(["date", "cik"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def _pct(g, col, higher_better=True):
    x = g[col] if higher_better else -g[col]
    return x.rank(pct=True)


def composite(df):
    """Rows with >= MIN_FAMILIES families present; family percentile ranks within each date; composite =
    mean of the available family ranks."""
    have = pd.concat([df["mom"].notna(), df["vol"].notna(), df["gpa"].notna() | df["roa"].notna(),
                      df["ep"].notna() | df["bm"].notna(), df["issuance"].notna(), df["assetgrowth"].notna()],
                     axis=1).sum(axis=1)
    df = df[have >= MIN_FAMILIES]
    out = []
    for _, g in df.groupby("date", sort=True):
        g = g.copy()
        g["f_mom"] = _pct(g, "mom")
        g["f_lowvol"] = _pct(g, "vol", False)
        g["f_quality"] = pd.concat([_pct(g, "gpa"), _pct(g, "roa")], axis=1).mean(axis=1)
        g["f_value"] = pd.concat([_pct(g, "ep"), _pct(g, "bm")], axis=1).mean(axis=1)
        g["f_issuance"] = _pct(g, "issuance", False)
        g["f_assetgrowth"] = _pct(g, "assetgrowth", False)
        g["n_families"] = g[FAMILIES].notna().sum(axis=1)
        g["score"] = g[FAMILIES].mean(axis=1)
        g["score_pct"] = g["score"].rank(pct=True)
        g["rank"] = g["score"].rank(ascending=False, method="first").astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def picks(scored, n=TOP_N):
    return scored[scored["rank"] <= n].sort_values(["date", "rank"])
