"""INS1 (research/prereg/INS1.md): buy $5+ stocks after an opportunistic insider's open-market purchase.

Stage "data" downloads SEC's quarterly Insider Transactions Data Sets (Form 3/4/5, 2013 Q1 .. 2025 Q3) and
keeps the Form 4 non-derivative trades with their filing date, issuer and reporting owner. Stage "events"
classifies insiders (Cohen, Malloy & Pomorski 2012), builds the events and their control candidates on
W1's universe, and writes the 20-event hand check (research/checks/INS1.csv). Stage "measure" waits for
that check (>= 19 right). Nothing before "measure" reads a return.

Usage: python -m tracks.main.ins1_study --stage data --cache cache/w1 --out research/results/INS1
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core import sec, stats
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat
from tracks.news import nx1_study as NX
from tracks.news import w1_data as D
from tracks.penny import pd1_study as P

SEED, CHECK_N, CHECK_PASS = 20261030, 20, 19
QUARTERS = pd.period_range("2013Q1", "2025Q3", freq="Q")
ZIP = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
MIN_VALUE, MIN_PRICE, MIN_DV20 = 25_000.0, 5.0, 1e6
CLUSTER, HOLD, N_CTRL, LAGS = 5, 20, 5, 11
FIRST_ENTRY, LAST_ENTRY, HOLDOUT = pd.Timestamp("2017-01-01"), pd.Timestamp("2025-08-31"), pd.Timestamp("2025-10-01")
MIN_EVENTS = 200
CHECK = Path("research/checks/INS1.csv")
QDIR = "q2"  # v1 ("q") counted multi-owner trades once per owner and kept preferred stock (fixed 2026-10-30)


def idir(cache):
    return D.w1dir(cache) / "ins1"


# ---------- data ----------

def read_quarter(raw):
    """Form 4 non-derivative trades of one quarter's zip: accession, filed, issuer cik and symbol, owner cik,
    officer/director flags, trade date, code, shares, price."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    rd = lambda n, c: pd.read_csv(z.open(n), sep="\t", usecols=c, dtype=str, quoting=3, on_bad_lines="skip")
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]
    own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "TRANS_DATE",
                                   "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    tr = tr[common_stock(tr["SECURITY_TITLE"])]
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    own = own.assign(officer_dir=rel.str.contains("officer") | rel.str.contains("director"))
    m = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    # a filing with several reporting owners (a fund and its manager) lists each trade once: keep one owner per
    # trade, an officer/director first, so values and histories are not counted once per owner
    m = (m.assign(_oc=pd.to_numeric(m["RPTOWNERCIK"], errors="coerce"))
         .sort_values(["officer_dir", "_oc"], ascending=[False, True], kind="stable")
         .drop_duplicates("NONDERIV_TRANS_SK"))
    out = pd.DataFrame({
        "accession": m["ACCESSION_NUMBER"],
        "filed": pd.to_datetime(m["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "issuer_cik": pd.to_numeric(m["ISSUERCIK"], errors="coerce"),
        "issuer_symbol": m["ISSUERTRADINGSYMBOL"].str.upper().str.strip(),
        "owner_cik": pd.to_numeric(m["RPTOWNERCIK"], errors="coerce"),
        "officer_dir": m["officer_dir"].astype(bool),
        "trade_date": pd.to_datetime(m["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
        "code": m["TRANS_CODE"].str.strip(),
        "shares": pd.to_numeric(m["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce")})
    return out[out["code"].isin(["P", "S"])].dropna(subset=["filed", "issuer_cik", "owner_cik", "trade_date"])


NOT_COMMON = ("prefer", "pfd", "%", "note", "debenture", "bond", "warrant", "right")


def common_stock(title):
    """True where a non-derivative security title is the issuer's common equity (not preferred, notes or
    warrants)."""
    t = title.fillna("").str.lower()
    return ~t.apply(lambda x: any(w in x for w in NOT_COMMON))


def stage_data(cache, out):
    d = idir(cache) / QDIR
    d.mkdir(parents=True, exist_ok=True)
    for q in QUARTERS:
        f = d / f"{q}.parquet"
        if f.exists():
            continue
        r = sec.get(ZIP.format(q=str(q).lower()))
        if r is None:
            print(f"[ins1 data] {q}: no file")
            continue
        t = read_quarter(r.content)
        NX.atomic_parquet(t, f)
        print(f"[ins1 data] {q}: {len(t)} P/S trades, {int((t['code'] == 'P').sum())} purchases", flush=True)


def load_trades(cache):
    return pd.concat([pd.read_parquet(f) for f in sorted((idir(cache) / QDIR).glob("*.parquet"))], ignore_index=True)


# ---------- events (pure helpers tested) ----------

def classify(tr):
    """Per trade: 'routine', 'opportunistic' or None. Routine: the insider traded (P or S) in the same
    calendar month in each of the 3 years before the trade's year; opportunistic: traded in each of those
    3 years but not routine; None: not enough history."""
    ym = tr.assign(y=tr["trade_date"].dt.year, m=tr["trade_date"].dt.month)
    years = ym.groupby("owner_cik")["y"].agg(set).to_dict()
    months = ym.groupby(["owner_cik", "y"])["m"].agg(set).to_dict()
    out = []
    for o, y, m in zip(ym["owner_cik"], ym["y"], ym["m"]):
        prior = [y - 1, y - 2, y - 3]
        if not all(p in years.get(o, ()) for p in prior):
            out.append(None)
        elif all(m in months.get((o, p), ()) for p in prior):
            out.append("routine")
        else:
            out.append("opportunistic")
    return pd.Series(out, index=tr.index)


def next_session(days, d):
    """The first session after date d."""
    i = days.searchsorted(d, side="right")
    return days[i] if i < len(days) else pd.NaT


def cluster(ev, days, n=CLUSTER):
    """Keeps a company's first filing; later filings within n sessions of the kept one are dropped."""
    keep, last = [], {}
    pos = {d: i for i, d in enumerate(days)}
    for i, r in ev.sort_values(["filed", "accession"]).iterrows():
        k = pos.get(r["entry"])
        if k is None:
            continue
        j = last.get(r["issuer_cik"])
        if j is not None and k - j <= n:
            continue
        last[r["issuer_cik"]] = k
        keep.append(i)
    return ev.loc[keep]


def stage_events(cache, out):
    tr = load_trades(cache)
    tr = tr.assign(cls=classify(tr))
    buys = tr[(tr["code"] == "P") & (tr["cls"] == "opportunistic") & tr["officer_dir"]
              & (tr["shares"] * tr["price"] >= MIN_VALUE)]
    buys = buys.groupby("accession", as_index=False).agg(
        filed=("filed", "first"), issuer_cik=("issuer_cik", "first"), issuer_symbol=("issuer_symbol", "first"),
        owner_cik=("owner_cik", "first"), value=("price", "size"), trade_date=("trade_date", "min"))
    vals = tr[tr["accession"].isin(buys["accession"]) & (tr["code"] == "P")].assign(v=lambda x: x["shares"] * x["price"])
    buys["value"] = buys["accession"].map(vals.groupby("accession")["v"].sum())
    cal = D.calendar()
    days = pd.DatetimeIndex(cal["date"])
    buys["entry"] = [next_session(days, d) for d in buys["filed"]]
    buys["prior_day"] = [days[days.searchsorted(e) - 1] for e in buys["entry"]]
    buys = buys[(buys["entry"] >= FIRST_ENTRY) & (buys["entry"] <= LAST_ENTRY)]
    rows = []
    for y in sorted(set(buys["prior_day"].dt.year)):
        u = D.universe_year(cache, y)
        rows.append(u[(u["close"] >= MIN_PRICE) & (u["dv20"] >= MIN_DV20)])
    uni = pd.concat(rows, ignore_index=True)
    ev = buys.merge(uni.rename(columns={"date": "prior_day", "cik": "issuer_cik"}), on=["prior_day", "issuer_cik"])
    ev = ev.drop_duplicates("accession")
    ev = cluster(ev, days).reset_index(drop=True)
    ev["exit"] = [days[days.searchsorted(e) + HOLD - 1] if days.searchsorted(e) + HOLD - 1 < len(days) else pd.NaT
                  for e in ev["entry"]]
    ev = ev[ev["exit"] < HOLDOUT].reset_index(drop=True)
    by_day = {d: g.sort_values("symbol") for d, g in uni.groupby("date")}
    cands = []
    for r in ev.itertuples():
        pool = by_day[r.prior_day]
        pool = pool[pool["cik"] != r.issuer_cik]
        perm = np.random.default_rng(SEED + r.entry.toordinal() + int(r.issuer_cik) % 1000).permutation(len(pool))
        cands.append(pool.iloc[perm[:3 * N_CTRL]]["symbol"].tolist())
    ev["control_candidates"] = cands
    d = idir(cache)
    NX.atomic_parquet(ev, d / "events.parquet")
    chk = ev.sample(min(CHECK_N, len(ev)), random_state=SEED)
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"accession": chk["accession"], "issuer_cik": chk["issuer_cik"].astype(int),
                  "symbol": chk["symbol"], "owner_cik": chk["owner_cik"].astype(int), "filed": chk["filed"].dt.date,
                  "value": chk["value"].round(0),
                  "url": [f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{a.replace('-', '')}/"
                          for c, a in zip(chk["issuer_cik"], chk["accession"])],
                  "check": "insider is an officer/director, code P, value, filing date, issuer = symbol", "ok": ""}
                 ).to_csv(CHECK, index=False)
    out.mkdir(parents=True, exist_ok=True)
    funnel = {"form4_PS_trades": int(len(tr)), "classified": tr["cls"].value_counts().to_dict(),
              "opportunistic_officer_dir_buys_25k": int(len(buys)), "events": int(len(ev)),
              "per_year": {str(k): int(v) for k, v in ev["entry"].dt.year.value_counts().sort_index().items()}}
    (out / "INS1_funnel.json").write_text(json.dumps(funnel, indent=1, default=str))
    print(json.dumps(funnel, default=str))


# ---------- measure ----------

def check_right():
    if not CHECK.exists():
        return None
    ok = pd.read_csv(CHECK)["ok"].astype(str).str.lower()
    return int((ok == "y").sum()) if ok.isin(["y", "n"]).all() else None


def cohorts(ev):
    e = ev.dropna(subset=["net", "cnet"])
    c = e.groupby(e["entry"].dt.to_period("M")).agg(pick=("net", "mean"), ctrl=("cnet", "mean"), n=("net", "size"))
    return c.assign(diff=c["pick"] - c["ctrl"]).sort_index()


def stage_measure(cache, out):
    n_ok = check_right()
    res = {"idea": "INS1", "prereg": "research/prereg/INS1.md", "check_right": n_ok, "check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok is None:
        raise SystemExit("[ins1] fill in research/checks/INS1.csv (ok = y/n) first")
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: nothing measured"
        (out / "INS1_result.json").write_text(json.dumps(res, indent=1))
        return print(json.dumps(res))
    ev = pd.read_parquet(idir(cache) / "events.parquet")
    need = set(ev["symbol"]) | {s for v in ev["control_candidates"] for s in v}
    rets = {}
    for bars in P.per_batch(cache, need):
        for i, r in ev.iterrows():
            for s in [r["symbol"]] + list(r["control_candidates"]):
                if s in bars:
                    rets[(i, s)] = P.trade_ret(bars[s], r["entry"], r["exit"])
    days_held = (ev["exit"] - ev["entry"]).dt.days + 1
    cost = NX.cost_frac(ev["close"], np.zeros(len(ev), bool), days_held)
    ev["ret"] = [rets.get((i, s), np.nan) for i, s in zip(ev.index, ev["symbol"])]
    ev["cret"] = [np.mean([rets[(i, s)] for s in c if np.isfinite(rets.get((i, s), np.nan))][:N_CTRL] or [np.nan])
                  for i, c in zip(ev.index, ev["control_candidates"])]
    ev["net"], ev["cnet"] = ev["ret"] - cost, ev["cret"] - cost
    c = cohorts(ev)
    m, se, t = nw_tstat(c["diff"], LAGS)
    lo, hi = block_bootstrap_ci(c["diff"], 12, 10_000, SEED)
    pm, _, pt = nw_tstat(c["pick"], LAGS)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    f = ff5.join(umd, how="inner").reindex(c.index)[["Mkt-RF", "SMB", "HML", "RMW", "CMA", umd.columns[0]]]
    ok = f.notna().all(axis=1).to_numpy()
    alpha = stats.ols_hac(c["diff"].to_numpy(float)[ok], f.to_numpy()[ok], LAGS) if ok.sum() > 8 else None
    a = alpha or {"alpha": float("nan"), "t": float("nan")}
    n_ev = int(c["n"].sum())
    prim = {"events": n_ev, "months": int(len(c)), "mean_minus_control": m, "se_hac": se, "t_hac": t,
            "ci95": [lo, hi], "picks_mean_net": pm, "picks_t": pt, "control_mean_net": float(c["ctrl"].mean()),
            "ff5_mom_alpha": alpha}
    p2, both = stats.idea_counts()
    prim["passes_rule12"] = bool(t >= 3.0 and lo > 0 and pm > 0 and n_ev >= MIN_EVENTS)
    prim["passes_rule11"] = bool(a["alpha"] > 0 and a["t"] >= 2.0)
    prim["passes"] = prim["passes_rule12"] and prim["passes_rule11"]
    prim["near_miss_rule19"] = bool(not prim["passes"] and t >= 1.5 and pm > 0)
    prim["deflated_sharpe"] = stats.deflated_sharpe(c["diff"], both)
    res |= {"phase2_count": p2, "phase1_plus_phase2": both, "ff5_sha256": sha5, "umd_sha256": shau,
            "funnel": {"events": int(len(ev)), "with_trade": int(ev["ret"].notna().sum()),
                       "with_control": int(ev["cret"].notna().sum())},
            "primary": prim, "by_year": {str(y): float(g["diff"].mean()) for y, g in c.groupby(c.index.year)}}
    (out / "INS1_result.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "events", "measure"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/INS1")
    a = ap.parse_args()
    {"data": stage_data, "events": stage_events, "measure": stage_measure}[a.stage](Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
