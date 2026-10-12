"""M5 insider-buying measure (research/prereg/M5.md). Pure functions."""
import numpy as np
import pandas as pd

WINDOW_DAYS = 182


def trailing_buys(keys, buys, window_days=WINDOW_DAYS):
    """For each (cik, date) in keys: dollar value and number of distinct buying insiders over purchases
    filed in (date - window_days, date]."""
    k = keys.reset_index(drop=True)
    value = np.zeros(len(k))
    n_buyers = np.zeros(len(k), dtype=int)
    if len(buys):
        b = buys.assign(filed=pd.to_datetime(buys["filed"]).astype("datetime64[ns]")).sort_values("filed")
        groups = {c: g for c, g in b.groupby("cik")}
        dates = pd.to_datetime(k["date"]).astype("datetime64[ns]").to_numpy()
        for i, (c, t) in enumerate(zip(k["cik"].to_numpy(), dates)):
            g = groups.get(int(c))
            if g is None:
                continue
            f = g["filed"].to_numpy()
            lo = np.searchsorted(f, t - np.timedelta64(window_days, "D"), side="right")
            hi = np.searchsorted(f, t, side="right")
            if hi > lo:
                w = g.iloc[lo:hi]
                value[i] = w["value"].sum()
                n_buyers[i] = len({o for s in w["owners"] for o in s.split(",")})
    return value, n_buyers


def score_m5(uni, buys):
    """Rows with net issuance defined; insider buying / market cap and the M5 composite and rank per date."""
    d = uni[uni["issuance"].notna()].reset_index(drop=True).copy()
    v, n = trailing_buys(d[["cik", "date"]], buys)
    d["insider_value"], d["insider_buyers"] = v, n
    d["insider_ratio"] = v / d["mcap"].to_numpy()
    g = d.groupby("date")
    d["f_insider"] = g["insider_ratio"].rank(pct=True)
    d["f_issuance"] = g["issuance"].rank(pct=True, ascending=False)
    d["score"] = (d["f_insider"] + d["f_issuance"]) / 2
    d["score_pct"] = d.groupby("date")["score"].rank(pct=True)
    d["rank"] = d.groupby("date")["score"].rank(ascending=False, method="first").astype(int)
    return d
