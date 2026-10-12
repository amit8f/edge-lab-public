"""Point-in-time fundamentals from SEC companyfacts (M4). Pure functions, no network.

`extract` turns one companyfacts JSON into long rows; `annual_records` and `share_records` turn those rows
into tables keyed by the date each value first became public (its first filing date), and `as_of` joins
them to (cik, date) rows with merge_asof, so nothing is used before it was filed. Only original values
count: a restated number keeps the date of the filing that first reported its period.
"""
import numpy as np
import pandas as pd

FLOW = {
    "ni": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "gp": ["GrossProfit"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
}
INSTANT = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
SHARES = "EntityCommonStockSharesOutstanding"
STALE_DAYS = 540
COLS = ["cik", "concept", "start", "end", "val", "form", "filed", "accn"]


def extract(cf):
    """companyfacts JSON -> long rows for the concepts M4 uses (USD facts, cover-page share counts)."""
    cik = int(cf.get("cik", 0))
    facts = cf.get("facts", {})
    want = {c for v in list(FLOW.values()) + list(INSTANT.values()) for c in v}
    rows = []
    for concept, body in (facts.get("us-gaap") or {}).items():
        if concept in want:
            for f in body.get("units", {}).get("USD", []):
                rows.append((cik, concept, f.get("start"), f["end"], f["val"], f.get("form"), f["filed"], f.get("accn")))
    for f in ((facts.get("dei") or {}).get(SHARES) or {}).get("units", {}).get("shares", []):
        rows.append((cik, SHARES, None, f["end"], f["val"], f.get("form"), f["filed"], f.get("accn")))
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype(float)
    return df


def _first_filed(rows, concepts, flow):
    """(cik, end) -> original value of the first concept in `concepts` that has one, from 10-Ks only."""
    r = rows[rows["concept"].isin(concepts) & (rows["form"] == "10-K")].copy()
    if flow:
        days = (r["end"] - r["start"]).dt.days
        r = r[(days >= 330) & (days <= 400)]
    if r.empty:
        return pd.DataFrame(columns=["cik", "end", "val", "filed"])
    r["prio"] = r["concept"].map({c: k for k, c in enumerate(concepts)})
    r = r.sort_values(["cik", "end", "prio", "filed"])
    best = r.groupby(["cik", "end"])["prio"].transform("min")
    r = r[r["prio"] == best]
    return r.groupby(["cik", "end"], as_index=False).first()[["cik", "end", "val", "filed"]]


def annual_records(rows):
    """One row per (cik, fiscal year end E), available from the first filing of E's total assets:
    ni, gp (gross profit, else revenue - cost of revenue), assets, equity, assets_prev (the year before)."""
    assets = _first_filed(rows, INSTANT["assets"], False)
    if assets.empty:
        return pd.DataFrame(columns=["cik", "end", "avail", "ni", "gp", "assets", "equity", "assets_prev"])
    rec = assets.rename(columns={"val": "assets", "filed": "avail"})
    for k in ("ni", "gp", "rev", "cogs"):
        rec = rec.merge(_first_filed(rows, FLOW[k], True).rename(columns={"val": k})[["cik", "end", k]],
                        on=["cik", "end"], how="left")
    rec = rec.merge(_first_filed(rows, INSTANT["equity"], False).rename(columns={"val": "equity"})
                    [["cik", "end", "equity"]], on=["cik", "end"], how="left")
    for c in ("ni", "gp", "rev", "cogs", "equity"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce").astype(float)
    rec["gp"] = rec["gp"].fillna(rec["rev"] - rec["cogs"])
    prev = assets.rename(columns={"end": "end_prev", "val": "assets_prev"})[["cik", "end_prev", "assets_prev"]]
    rec["target"] = rec["end"] - pd.Timedelta(days=365)
    rec = pd.merge_asof(rec.sort_values("target"), prev.sort_values("end_prev"), left_on="target",
                        right_on="end_prev", by="cik", direction="nearest", tolerance=pd.Timedelta(days=20))
    return rec[["cik", "end", "avail", "ni", "gp", "assets", "equity", "assets_prev"]].reset_index(drop=True)


def share_records(rows):
    """Cover-page shares outstanding per filing (classes summed), from 10-K and 10-Q only."""
    r = rows[(rows["concept"] == SHARES) & rows["form"].isin(["10-K", "10-Q"])]
    if r.empty:
        return pd.DataFrame(columns=["cik", "sh_end", "avail", "shares"])
    r = r.assign(val=pd.to_numeric(r["val"], errors="coerce").astype(float))
    g = r.groupby(["cik", "accn"]).agg(shares=("val", "sum"), sh_end=("end", "max"), avail=("filed", "min"))
    return g.reset_index()[["cik", "sh_end", "avail", "shares"]]


def as_of(keys, table, on="date", stale_days=None, end_col=None):
    """Left-join to `keys` (cik, `on`) the latest `table` row per cik with avail <= keys[on]. Rows older
    than `stale_days` (by `end_col`) are dropped."""
    k = keys.reset_index(drop=True).copy()
    k["_row"] = np.arange(len(k))
    k["cik"] = k["cik"].astype("int64")
    t = table.copy()
    t["cik"] = t["cik"].astype("int64")
    k[on] = k[on].astype("datetime64[ns]")
    t["avail"] = t["avail"].astype("datetime64[ns]")
    t = t.dropna(subset=["avail"]).sort_values("avail")
    m = pd.merge_asof(k.dropna(subset=[on]).sort_values(on), t, left_on=on, right_on="avail", by="cik",
                      direction="backward")
    if stale_days is not None and end_col is not None:
        old = (m[on] - m[end_col]).dt.days > stale_days
        m.loc[old, [c for c in t.columns if c != "cik"]] = np.nan
    return m.sort_values("_row").drop(columns="_row").reset_index(drop=True)
