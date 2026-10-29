"""PD1 (research/prereg/PD1.md): post-earnings drift in penny stocks. Stage "events" builds the earnings 8-K
events from SEC's submissions records (acceptance time and items) for every company that was in W1's penny
universe, and writes the 20-event hand-check file; the measure stage waits for that check (>= 19 right).

Shares W1's cache (`tracks/news/w1_data.py`: daily bars, universe flag, CIKs). Nothing here reads a return.

Usage: python -m tracks.penny.pd1_study --stage events --cache cache/w1 --out research/results/PD1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import sec
from tracks.news import l1_study as L1
from tracks.news import w1_data as D

SEED, CHECK_N, CHECK_PASS = 20261028, 20, 19
FIRST, LAST = pd.Timestamp("2015-10-01"), pd.Timestamp("2025-09-30")
DEDUP_DAYS, BATCH = 30, 500
SUBS = "https://data.sec.gov/submissions/"
COLS = ["cik", "accession", "form", "filed", "accepted_et", "items"]


def pd1dir(cache):
    return D.w1dir(cache) / "pd1"


def earnings_rows(cik, get=sec.get, first=FIRST, last=LAST):
    """Form 8-K filings listing item 2.02, filed first .. last, with their acceptance time. The submissions
    record writes acceptanceDateTime with a 'Z' but the clock is EDGAR's (ET); the hand check compares it
    with the filing's index page."""
    r = get(f"{SUBS}CIK{int(cik):010d}.json")
    if r is None:
        return []
    j = r.json()
    blocks = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        if pd.Timestamp(f.get("filingTo", "1900-01-01")) >= first:
            rr = get(SUBS + f["name"])
            if rr is not None:
                blocks.append(rr.json())
    out = []
    for b in blocks:
        n = len(b.get("form", []))
        for acc, form, filed, items, at in zip(b.get("accessionNumber", []), b.get("form", []),
                                               b.get("filingDate", []), b.get("items", [""] * n),
                                               b.get("acceptanceDateTime", [""] * n)):
            its = [i.strip() for i in (items or "").split(",") if i.strip()]
            if form == "8-K" and "2.02" in its and first <= pd.Timestamp(filed) <= last and at:
                out.append({"cik": int(cik), "accession": acc, "form": form, "filed": filed,
                            "accepted_et": pd.Timestamp(at[:19]), "items": ",".join(its)})
    return out


def day0(accepted, cal):
    """The first session whose close comes after acceptance (cal: date, open, close as 'HH:MM' ET)."""
    closes = pd.to_datetime(cal["date"].dt.strftime("%Y-%m-%d") + " " + cal["close"].astype(str))
    i = np.searchsorted(closes.to_numpy(), pd.DatetimeIndex(accepted).to_numpy(), side="right")
    d = cal["date"].to_numpy()
    return pd.DatetimeIndex([d[k] if k < len(d) else pd.NaT for k in i])


def first_per_company(ev, days=DEDUP_DAYS):
    """One event per company per `days` calendar days: the first, then the next one at least `days` later."""
    ev = ev.sort_values(["cik", "accepted_et", "accession"])
    keep, last = [], {}
    for c, t in zip(ev["cik"], ev["accepted_et"]):
        ok = c not in last or (t - last[c]) >= pd.Timedelta(days=days)
        keep.append(ok)
        if ok:
            last[c] = t
    return ev[keep]


def penny_rows(cache):
    """W1 universe rows with raw close under $5 (W1's universe is L1's: close >= $1, dv20 >= $500k)."""
    parts = [pd.read_parquet(p, columns=["symbol", "date", "c", "dv20", "cik"],
                             filters=[("uni", "==", True), ("c", "<", L1.PENNY_MAX)]) for p in D.batch_files(cache)]
    u = pd.concat([p for p in parts if len(p)], ignore_index=True).dropna(subset=["cik"])
    return u.astype({"cik": "int64"})


def stage_filings(cache, ciks):
    d = pd1dir(cache) / "subs"
    d.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(ciks), BATCH):
        f = d / f"e_{b // BATCH:04d}.parquet"
        if f.exists():
            continue
        rows = [r for c in ciks[b:b + BATCH] for r in earnings_rows(c)]
        tmp = f.with_suffix(".tmp")
        pd.DataFrame(rows, columns=COLS).to_parquet(tmp)
        tmp.replace(f)
        print(f"[pd1 filings] batch {b // BATCH + 1}/{-(-len(ciks) // BATCH)}: {len(rows)} earnings 8-Ks")
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("e_*.parquet"))], ignore_index=True)


def events(filings, pen, cal):
    """Dedup per company, day 0, then the penny universe on the session before day 0 (one row per company
    per date, highest dollar volume)."""
    ev = first_per_company(filings.drop_duplicates("accession"))
    ev = ev.assign(day0=day0(ev["accepted_et"], cal)).dropna(subset=["day0"])
    days = cal["date"].reset_index(drop=True)
    pos = days.searchsorted(ev["day0"])
    ev["prior_day"] = [days[p - 1] if p > 0 else pd.NaT for p in pos]
    u = pen.sort_values("dv20", ascending=False).drop_duplicates(["date", "cik"])
    u = u.rename(columns={"date": "prior_day", "c": "prior_close"})
    return ev.merge(u, on=["cik", "prior_day"], how="inner").sort_values(["day0", "cik"]).reset_index(drop=True)


def stage_events(cache, out):
    cal = D.calendar()
    pen = penny_rows(cache)
    ciks = sorted(pen["cik"].unique().tolist())
    fil = stage_filings(cache, ciks)
    ev = events(fil, pen, cal)
    tmp = pd1dir(cache) / "events.tmp"
    ev.to_parquet(tmp)
    tmp.replace(pd1dir(cache) / "events.parquet")
    out.mkdir(parents=True, exist_ok=True)
    sig = ev[ev["day0"] >= pd.Timestamp("2017-01-01")]
    chk_file = Path("research/checks/PD1.csv")
    if not chk_file.exists():
        k = np.sort(np.random.default_rng(SEED).choice(len(sig), size=min(CHECK_N, len(sig)), replace=False))
        chk_file.parent.mkdir(parents=True, exist_ok=True)
        sig.iloc[k][["cik", "accession", "symbol", "accepted_et", "items", "day0"]].assign(
            accepted_et=lambda x: x["accepted_et"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            day0=lambda x: x["day0"].dt.strftime("%Y-%m-%d")).to_csv(chk_file, index=False)
    cov = {"penny_companies": len(ciks), "earnings_8k": int(len(fil)), "events_in_universe": int(len(ev)),
           "events_from_2017": int(len(sig)),
           "per_year": {str(y): int(n) for y, n in ev["day0"].dt.year.value_counts().sort_index().items()},
           "after_close_share": float((ev["accepted_et"].dt.hour >= 16).mean()) if len(ev) else None}
    (out / "PD1_coverage.json").write_text(json.dumps(cov, indent=1))
    print("PD1", json.dumps(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/PD1")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    stage_events(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
