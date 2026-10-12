"""M7 (research/prereg/M7.md): standardized earnings surprise from first-filed quarterly net income.

`quarterly_ni` turns companyfacts rows (fundamentals.extract) into one original net-income value per
fiscal quarter, usable from its first filing date; `sue_table` scales each quarter's change from the same
quarter a year earlier by the spread of the previous 8 such changes; `score_m7` joins the latest SUE filed
within 120 days to the universe and ranks it.
"""
import numpy as np
import pandas as pd

from tracks.main.longterm import fundamentals as fu

NI_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
WINDOW_DAYS = 120
MIN_HIST, MAX_HIST = 4, 8


def _first(r):
    """Per (cik, end): the value of the highest-priority concept from the first filing that reported it."""
    if r.empty:
        return pd.DataFrame(columns=["cik", "end", "val", "filed"])
    r = r.assign(prio=r["concept"].map({c: k for k, c in enumerate(NI_CONCEPTS)}))
    r = r.sort_values(["cik", "end", "prio", "filed"])
    r = r[r["prio"] == r.groupby(["cik", "end"])["prio"].transform("min")]
    return r.groupby(["cik", "end"], as_index=False).first()[["cik", "end", "val", "filed"]]


def quarterly_ni(rows):
    """One row per (cik, quarter end): ni and avail (first filing date). Fourth quarters missing a
    direct value are the annual value minus the nine-month value ending about 91 days earlier."""
    r = rows[rows["concept"].isin(NI_CONCEPTS) & rows["form"].isin(["10-Q", "10-K"])].dropna(subset=["start"])
    days = (r["end"] - r["start"]).dt.days
    q = _first(r[(days >= 80) & (days <= 100)])
    ann = _first(r[(days >= 330) & (days <= 400) & (r["form"] == "10-K")])
    nine = _first(r[(days >= 260) & (days <= 290)])
    if not ann.empty and not nine.empty:
        ann = ann.assign(target=ann["end"] - pd.Timedelta(days=91)).sort_values("target")
        n9 = nine.rename(columns={"end": "end9", "val": "val9", "filed": "filed9"}).sort_values("end9")
        m = pd.merge_asof(ann, n9, left_on="target", right_on="end9", by="cik", direction="nearest",
                          tolerance=pd.Timedelta(days=12)).dropna(subset=["val9"])
        q4 = pd.DataFrame({"cik": m["cik"], "end": m["end"], "val": m["val"] - m["val9"],
                           "filed": m[["filed", "filed9"]].max(axis=1)})
        if not q.empty:
            have = q.assign(k=1).sort_values("end")
            chk = pd.merge_asof(q4.sort_values("end"), have[["cik", "end", "k"]].rename(columns={"end": "e2"}),
                                left_on="end", right_on="e2", by="cik", direction="nearest",
                                tolerance=pd.Timedelta(days=5))
            q4 = chk[chk["k"].isna()][["cik", "end", "val", "filed"]]
        q = pd.concat([q, q4], ignore_index=True)
    q = q.rename(columns={"val": "ni", "filed": "avail"})
    q["ni"] = pd.to_numeric(q["ni"], errors="coerce").astype(float)
    return q.dropna(subset=["ni"]).sort_values(["cik", "end"]).reset_index(drop=True)


def sue_table(q):
    """Per (cik, end): change vs the quarter ending 345-385 days earlier, and SUE = change / std of the
    previous up-to-8 changes (at least 4, std > 0)."""
    out = []
    for cik, g in q.groupby("cik", sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        ends = g["end"].values.astype("datetime64[D]")
        ni = g["ni"].values
        chg = np.full(len(g), np.nan)
        for i in range(len(g)):
            gap = (ends[i] - ends[:i]).astype(int)
            j = np.where((gap >= 345) & (gap <= 385))[0]
            if len(j):
                chg[i] = ni[i] - ni[j[-1]]
        sue = np.full(len(g), np.nan)
        prev = []
        for i in range(len(g)):
            hist = [c for c in prev[-MAX_HIST:]]
            if not np.isnan(chg[i]) and len(hist) >= MIN_HIST:
                sd = np.std(hist, ddof=1)
                if sd > 0:
                    sue[i] = chg[i] / sd
            if not np.isnan(chg[i]):
                prev.append(chg[i])
        out.append(g.assign(change=chg, sue=sue))
    if not out:
        return pd.DataFrame(columns=["cik", "end", "ni", "avail", "change", "sue"])
    return pd.concat(out, ignore_index=True)


def score_m7(uni, sue, seed, window_days=WINDOW_DAYS):
    """Join each (cik, date) to its latest SUE first filed in (date - window, date]; drop undefined;
    rank per date, highest SUE first, ties in a seeded random order."""
    t = sue.dropna(subset=["sue"])[["cik", "end", "avail", "sue"]].rename(columns={"end": "q_end"})
    m = fu.as_of(uni[["cik", "date"]], t, on="date")
    d = uni.reset_index(drop=True).copy()
    d["sue"], d["q_end"], d["sue_avail"] = m["sue"].values, m["q_end"].values, m["avail"].values
    fresh = (d["date"] - pd.to_datetime(d["sue_avail"])).dt.days < window_days
    d = d[fresh & d["sue"].notna()].copy()
    d["score"] = d["sue"]
    out = []
    for t_, g in d.groupby("date", sort=True):
        g = g.copy()
        g["_rnd"] = np.random.default_rng(seed + t_.toordinal()).random(len(g))
        g = g.sort_values(["score", "_rnd"], ascending=[False, True])
        g["rank"] = np.arange(1, len(g) + 1)
        g["score_pct"] = g["score"].rank(pct=True)
        out.append(g.drop(columns="_rnd"))
    return pd.concat(out, ignore_index=True) if out else d.assign(rank=[], score_pct=[])
