"""DOD1 (research/prereg/DOD1.md): buy listed companies after a big Pentagon contract that got no headline.

Stages (W1's cache, `tracks/news/w1_data.py`; NR1's company facts for revenue):
- "awards": USAspending prime contract transactions of the Department of Defense, one parquet per month
  2016-11 .. 2025-09, new awards and orders only (modification number "0"), obligation >= $1M (the search
  can't filter on the potential value; disclosed).
- "events": SEC names (current and former, with dates) of every $5+ company matched to recipient names, each
  matched award's base-and-all-options value from its USAspending record, the 1%-of-revenue rule, the
  no-news filter, controls and the 20-event hand-check file (research/checks/DOD1.csv). No return is read.
- "measure": after the check (>= 18 right).

Usage: python -m tracks.main.dod1_study --stage awards --cache cache/w1 --out research/results/DOD1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import sec, stats
from tracks.main import dod_text
from tracks.main import nr1_study as N
from tracks.main.longterm.fundamentals import FLOW, _first_filed
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import w1_data as D
from tracks.penny import pd1_study as P

SEED, CHECK_N, CHECK_PASS = 20261102, 20, 18
TX = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
AWARD = "https://api.usaspending.gov/api/v2/awards/{gid}/"
FIELDS = ["Award ID", "Mod", "Recipient Name", "Action Date", "Transaction Amount", "Transaction Description",
          "Awarding Sub Agency"]
MIN_OBLIG, MIN_SHARE, MIN_PRICE = 1e6, 0.01, 5.0
HOLD, GAP, N_CTRL, LAGS, BLOCK = 10, 10, 5, 10, 10
COST, COST_SPREAD = 0.0005, 0.0041
FIRST_T, HOLDOUT = pd.Timestamp("2017-01-03"), pd.Timestamp("2025-10-01")
TRIALS, MIN_TRADES, MIN_DAYS = 115, 300, 100
CHECK = Path("research/checks/DOD1.csv")


def ddir(cache):
    return D.w1dir(cache) / "dod1"


def post(url, body, tries=6):
    for attempt in range(tries):
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{url}: HTTP {r.status_code} {r.text[:200]}")
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: no answer after {tries} tries")


def get_json(url, tries=6):
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return None


# ---------- awards ----------

def month_rows(start, end):
    rows, page = [], 1
    while True:
        body = {"filters": {"award_type_codes": ["A", "B", "C", "D"],
                            "agencies": [{"type": "awarding", "tier": "toptier", "name": "Department of Defense"}],
                            "time_period": [{"start_date": start, "end_date": end}],
                            "award_amounts": [{"lower_bound": MIN_OBLIG}]},
                "fields": FIELDS, "limit": 100, "page": page, "sort": "Action Date", "order": "asc"}
        j = post(TX, body)
        res = j.get("results") or []
        rows += [r for r in res if str(r.get("Mod")) == "0"]
        if not j.get("page_metadata", {}).get("hasNext"):
            return rows
        page += 1


def stage_awards(cache, deadline):
    d = ddir(cache) / "awards"
    for m in pd.period_range("2016-11", "2025-09", freq="M"):
        f = d / f"a_{m}.parquet"
        if f.exists():
            continue
        if time.monotonic() > deadline:
            print("[dod1 awards] out of time; rerun to continue")
            return False
        rows = month_rows(f"{m.start_time:%Y-%m-%d}", f"{m.end_time:%Y-%m-%d}")
        N._atomic(pd.DataFrame(rows), f)
        print(f"[dod1 awards] {m}: {len(rows)} new awards >= $1M")
    return True


def load_awards(cache):
    a = pd.concat([pd.read_parquet(p) for p in sorted((ddir(cache) / "awards").glob("a_*.parquet"))],
                  ignore_index=True)
    a["action_date"] = pd.to_datetime(a["Action Date"])
    return a.drop_duplicates(["generated_internal_id"])


# ---------- events ----------

def name_table(cache, ciks):
    """(cik, clean name, from, to) for every name a company used, from EDGAR submissions, cached."""
    f = ddir(cache) / "names.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows = []
    for c in ciks:
        r = sec.get(f"https://data.sec.gov/submissions/CIK{int(c):010d}.json")
        if r is None:
            continue
        j = r.json()
        rows.append({"cik": int(c), "name": dod_text.clean_name(j.get("name") or ""), "from": "1900-01-01",
                     "to": "2100-01-01"})
        for fn in j.get("formerNames") or []:
            rows.append({"cik": int(c), "name": dod_text.clean_name(fn.get("name") or ""),
                         "from": (fn.get("from") or "1900-01-01")[:10], "to": (fn.get("to") or "2100-01-01")[:10]})
    t = pd.DataFrame(rows)
    t = t[t["name"] != ""]
    N._atomic(t, f)
    return t


def match(awards, names):
    a = awards.assign(clean=awards["Recipient Name"].fillna("").map(dod_text.clean_name), _i=np.arange(len(awards)))
    j = a.merge(names, left_on="clean", right_on="name")
    j = j[(j["action_date"] >= pd.to_datetime(j["from"])) & (j["action_date"] <= pd.to_datetime(j["to"]))]
    # a name two companies used at once is ambiguous: drop it
    amb = j.groupby("_i")["cik"].nunique()
    return j[j["_i"].isin(amb[amb == 1].index)].drop_duplicates("_i")


def award_values(cache, gids):
    """{gid: base and all options value} from each award's USAspending record, cached."""
    f = ddir(cache) / "values.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    for k, g in enumerate([g for g in gids if g not in known], 1):
        j = get_json(AWARD.format(gid=g))
        known[g] = (j or {}).get("base_and_all_options")
        if k % 200 == 0:
            f.write_text(json.dumps(known))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(known))
    return known


def revenue(cache, ciks):
    facts = pd.concat([pd.read_parquet(p) for p in sorted((N.ndir(cache) / "facts").glob("facts_*.parquet"))],
                      ignore_index=True)
    facts = facts[facts["cik"].isin(ciks)]
    return _first_filed(facts, FLOW["rev"], True).rename(columns={"val": "rev", "filed": "avail"})


def universe(cache):
    """$5+ W1 rows with a CIK (symbol, date, c, cik), one per company and date (highest dollar volume)."""
    parts = [pd.read_parquet(p, columns=["symbol", "date", "c", "dv20", "cik"],
                             filters=[("uni", "==", True), ("c", ">=", MIN_PRICE),
                                      ("date", ">=", FIRST_T - pd.Timedelta(days=10)), ("date", "<", HOLDOUT)])
             for p in D.batch_files(cache)]
    u = pd.concat([p for p in parts if len(p)], ignore_index=True).dropna(subset=["cik"]).astype({"cik": "int64"})
    return u.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])


def stage_events(cache, out):
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    uni = universe(cache)
    nr1_ciks = {int(c) for p in sorted((N.ndir(cache) / "facts").glob("facts_*.parquet"))
                for c in pd.read_parquet(p, columns=["cik"])["cik"].unique()}
    ciks = sorted(set(uni["cik"]) & nr1_ciks)
    names = name_table(cache, ciks)
    aw = load_awards(cache)
    m = match(aw, names)
    vals = award_values(cache, m["generated_internal_id"].tolist())
    m["potential"] = m["generated_internal_id"].map(vals).astype(float)
    m["value"] = np.fmax(m["Transaction Amount"].astype(float), m["potential"].fillna(0))
    ev = m.groupby(["cik", "action_date"], as_index=False).agg(
        value=("value", "sum"), n_awards=("value", "size"), award_ids=("Award ID", lambda s: ";".join(s)),
        recipient=("Recipient Name", "first"), gids=("generated_internal_id", lambda s: ";".join(s)))
    rv = revenue(cache, set(ev["cik"]))
    ev = pd.merge_asof(ev.sort_values("action_date"), rv.sort_values("avail")[["cik", "avail", "rev", "end"]],
                       left_on="action_date", right_on="avail", by="cik", direction="backward")
    ev["share"] = ev["value"] / ev["rev"]
    funnel = {"awards_new_ge_1m": int(len(aw)), "companies_named": len(ciks), "matched_awards": int(len(m)),
              "matched_companies": int(m["cik"].nunique()), "company_days": int(len(ev)),
              "with_revenue": int(ev["rev"].notna().sum())}
    ev = ev[(ev["share"] >= MIN_SHARE) & (ev["rev"] > 0)]
    funnel["share_ge_1pct"] = int(len(ev))
    # session of the award: the action date if it is a session, else the next one; entry the session after
    i = days.searchsorted(ev["action_date"].to_numpy())
    ev = ev.assign(d=days[np.minimum(i, len(days) - 1)])
    ev = ev.merge(uni[["cik", "date", "symbol", "c"]], left_on=["cik", "d"], right_on=["cik", "date"], how="inner")
    funnel["priced_5_plus"] = int(len(ev))
    nd = D.news_symbol_days(cache)
    nxt = pd.Series(days[np.minimum(days.searchsorted(ev["d"].to_numpy()) + 1, len(days) - 1)])
    nd_set = set(zip(nd["symbol"], nd["date"]))
    ev["news"] = [(s, a) in nd_set or (s, b) in nd_set for s, a, b in zip(ev["symbol"], ev["d"], nxt)]
    funnel["with_news"] = int(ev["news"].sum())
    ev = ev[~ev["news"]].sort_values(["cik", "d"])
    keep, last = [], {}
    pos = {d: n for n, d in enumerate(days)}
    for c, d in zip(ev["cik"], ev["d"]):
        ok = c not in last or pos[d] - pos[last[c]] > GAP
        keep.append(ok)
        if ok:
            last[c] = d
    ev = ev[keep]
    funnel["after_overlap_rule"] = int(len(ev))
    j = days.searchsorted(ev["d"].to_numpy())
    ev = ev.assign(entry=days[np.minimum(j + 1, len(days) - 1)], exit=days[np.minimum(j + HOLD, len(days) - 1)])
    ev = ev[(ev["d"] >= FIRST_T) & (ev["exit"] < HOLDOUT) & (j + HOLD < len(days))]
    funnel["events"] = int(len(ev))
    by_day = {d: g for d, g in uni.groupby("date")}
    ctrls = []
    for d, g in ev.groupby("d"):
        pool = by_day[d]
        pool = pool[~pool["cik"].isin(g["cik"])].sort_values("symbol").reset_index(drop=True)
        n = min(N_CTRL * len(g), len(pool))
        k = np.random.default_rng(SEED + d.toordinal()).choice(len(pool), size=n, replace=False)
        e0 = g.iloc[0]
        for r in pool.iloc[np.sort(k)].itertuples():
            ctrls.append({"d": d, "symbol": r.symbol, "cik": r.cik, "entry": e0["entry"], "exit": e0["exit"]})
    ctrls = pd.DataFrame(ctrls)
    for name, df in (("events", ev), ("controls", ctrls), ("matched", m)):
        N._atomic(df.drop(columns=[c for c in ("Transaction Description",) if c in df]), ddir(cache) / f"{name}.parquet")
    out.mkdir(parents=True, exist_ok=True)
    if not CHECK.exists() and len(ev):
        k = np.sort(np.random.default_rng(SEED).choice(len(ev), size=min(CHECK_N, len(ev)), replace=False))
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        ev.iloc[k][["d", "symbol", "cik", "recipient", "award_ids", "value", "rev", "share", "action_date"]].assign(
            d=lambda x: x["d"].dt.strftime("%Y-%m-%d"), action_date=lambda x: x["action_date"].dt.strftime("%Y-%m-%d"),
            ok="", note="").to_csv(CHECK, index=False)
    funnel["events_per_year"] = {str(y): int(n) for y, n in ev["d"].dt.year.value_counts().sort_index().items()}
    funnel["share_buckets"] = {"1-3%": int(((ev["share"] >= .01) & (ev["share"] < .03)).sum()),
                               "3-10%": int(((ev["share"] >= .03) & (ev["share"] < .1)).sum()),
                               "10%+": int((ev["share"] >= .1).sum())}
    funnel["potential_value_found"] = float(m["potential"].notna().mean()) if len(m) else 0.0
    (out / "DOD1_coverage.json").write_text(json.dumps(funnel, indent=1))
    print("DOD1 FUNNEL", json.dumps(funnel))


# ---------- measure ----------

def returns(cache, rows):
    out = np.full(len(rows), np.nan)
    rows = rows.reset_index(drop=True)
    idx = {s: np.flatnonzero(rows["symbol"].to_numpy() == s) for s in rows["symbol"].unique()}
    for bars in P.per_batch(cache, set(rows["symbol"])):
        for s, g in bars.items():
            for i in idx.get(s, []):
                out[i] = P.trade_ret(g, rows.at[i, "entry"], rows.at[i, "exit"])
    return rows.assign(r=out)


def daily(ev, ctrls, cost):
    e = ev.dropna(subset=["r"]).assign(net=lambda x: x["r"] - cost)
    c = ctrls.dropna(subset=["r"]).assign(net=lambda x: x["r"] - cost)
    w = e.groupby("d").agg(mean=("net", "mean"), size=("net", "size"), entry=("entry", "first"), exit=("exit", "first"))
    w = w.join(c.groupby("d")["net"].mean().rename("ctrl"), how="inner")
    return w.assign(diff=w["mean"] - w["ctrl"])


def test(w, fd, cal):
    if len(w) < 3:
        return {"days": int(len(w)), "trades": int(w["size"].sum()) if len(w) else 0}
    m, se, t = nw_tstat(w["diff"], LAGS)
    lo, hi = block_bootstrap_ci(w["diff"], BLOCK, 10_000, SEED)
    days = pd.DatetimeIndex(cal["date"])
    F = stats.window_factors(fd, [days[(days >= a) & (days <= b)] for a, b in zip(w["entry"], w["exit"])])
    a = stats.ols_hac(w["diff"].to_numpy(float), F, LAGS)
    return {"days": int(len(w)), "trades": int(w["size"].sum()), "mean_minus_control": m, "se_hac": se, "t_hac": t,
            "ci95": [lo, hi], "events_mean_net": float(w["mean"].mean()), "control_mean_net": float(w["ctrl"].mean()),
            "ff5_mom_alpha": a}


def stage_measure(cache, out):
    n_ok = N_check()
    res = {"idea": "DOD1", "prereg": "research/prereg/DOD1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok is None:
        raise SystemExit("[dod1] fill in research/checks/DOD1.csv (ok = y/n) first")
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "DOD1_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    cal = D.calendar()
    ev = returns(cache, pd.read_parquet(ddir(cache) / "events.parquet"))
    ctrls = returns(cache, pd.read_parquet(ddir(cache) / "controls.parquet"))
    ffd, sha5 = stats.fetch_french(stats.FF5 + "_daily")
    umdd, shau = stats.fetch_french(stats.UMD + "_daily")
    fd = ffd[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(umdd, how="inner")
    fd.index = pd.DatetimeIndex(fd.index)
    w = daily(ev, ctrls, COST)
    prim = test(w, fd, cal)
    if "t_hac" in prim:
        a = prim["ff5_mom_alpha"]
        prim["passes_rule12"] = bool(prim["mean_minus_control"] > 0 and prim["t_hac"] >= 3.0 and prim["ci95"][0] > 0
                                     and prim["events_mean_net"] > 0 and prim["trades"] >= MIN_TRADES
                                     and prim["days"] >= MIN_DAYS)
        prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
        prim["deflated_sharpe"] = stats.deflated_sharpe(w["diff"], TRIALS)
        prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
        prim["near_miss_rule19"] = bool(not prim["passes"] and prim["t_hac"] >= 1.5 and prim["events_mean_net"] > 0)
    sb = pd.cut(ev["share"], [0.01, 0.03, 0.10, np.inf], right=False, labels=["1-3%", "3-10%", "10%+"])
    nets = ev.dropna(subset=["r"])["r"] - COST
    p2, both = stats.idea_counts()
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"events": int(len(ev)), "events_with_trade": int(ev["r"].notna().sum()),
                       "controls_with_trade": int(ctrls["r"].notna().sum())},
            "primary": prim,
            "reported_alongside": {
                "at_0.41pct_cost": test(daily(ev, ctrls, COST_SPREAD), fd, cal),
                "share_buckets": {str(b): test(daily(ev[(sb == b).to_numpy()], ctrls, COST), fd, cal)
                                  for b in ["1-3%", "3-10%", "10%+"]},
                "by_year": {str(y): {"days": int(len(g)), "mean_minus_control": float(g["diff"].mean())}
                            for y, g in w.groupby(w.index.year)},
                "best_5pct_trade_mean": float(nets[nets >= nets.quantile(0.95)].mean()) if len(nets) else None,
                "worst_5pct_trade_mean": float(nets[nets <= nets.quantile(0.05)].mean()) if len(nets) else None}}
    (out / "DOD1_result.json").write_text(json.dumps(res, indent=1, default=str))
    w.to_csv(out / "DOD1_daily.csv")
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def N_check():
    if not CHECK.exists():
        return None
    ok = pd.read_csv(CHECK)["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["awards", "events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/DOD1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    if a.stage == "awards":
        stage_awards(Path(a.cache), time.monotonic() + a.budget_min * 60)
    elif a.stage == "events":
        stage_events(Path(a.cache), Path(a.out))
    else:
        stage_measure(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
