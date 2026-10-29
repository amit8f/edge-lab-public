"""G2 (research/prereg/G2.md): G1's customer links with a stricter matcher. Stage "check" builds the links and
the 50-link hand-check file; the screen waits for that check (>= 40 right).

G2 needs no new download. G1 stored every 10-K customer sentence in which its matcher found a company
(`cache/g1/customers`). G2's matches are a subset of G1's (an exact name is also a G1 match, and G1 keeps the
sentence when any phrase in it matches), so rerunning G2's rules on those sentences finds every G2 link.

Usage: python -m tracks.main.g2_links --stage check --cache cache/m4 --out research/results/G2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core import stats
from tracks.main import g1_links as G1
from tracks.main import m1_screen as m1
from tracks.main import m3_screen as m3

SEED = 20261024
CHECK_N, CHECK_PASS, MIN_PCT, MIN_ONE_WORD = 50, 40, 0.10, 5
_CUSTW = re.compile(r"\bcustomers?\b|\bdistributors?\b|\bwholesalers?\b|\bretailers?\b|\bsales to\b", re.I)
_SEGW = re.compile(r"\bsegments?\b|\btenants?\b|\bregion\b|\bgeographic\b", re.I)
_PCTS = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)


class Names(G1.Names):
    """Exact normalised names only; a one-word phrase also needs >= 5 letters."""

    def match(self, phrase):
        k = G1.key(phrase)
        if not k or k in G1.STOP:
            return None
        if len(phrase.split()) == 1 and len(re.sub(r"[^A-Za-z]", "", phrase)) < MIN_ONE_WORD:
            return None
        c = self.exact.get(k)
        return next(iter(c)) if c and len(c) == 1 else None


def keep_sentence(s):
    """G2 changes 2 and 3 on top of G1's sentence rule: customer words, no segment words, a share >= 10%."""
    if not _CUSTW.search(s) or _SEGW.search(s):
        return False
    p = [float(x) for x in _PCTS.findall(s) if float(x) <= 100]
    return bool(p) and max(p) / 100 >= MIN_PCT


def extract(sentence, names, self_cik):
    """(customer_cik, phrase, largest pct) per matched customer in one sentence, G1's longest-first order."""
    if not keep_sentence(sentence):
        return []
    pct = max(float(x) for x in _PCTS.findall(sentence) if float(x) <= 100) / 100
    rows, taken, seen = [], [], set()
    for ph in G1.phrases(sentence):
        if any(ph in t for t in taken):
            continue
        c = names.match(ph)
        if c is None or c == int(self_cik):
            continue
        taken.append(ph)
        if c not in seen:
            seen.add(c)
            rows.append((c, ph, pct))
    return rows


def links(rows, names):
    """G2 links with G1's validity windows (from the 10-K's filing date to the next 10-K or 15 months)."""
    s = rows.dropna(subset=["sentence"]).drop_duplicates(["supplier_cik", "accn", "sentence"])
    out = []
    for r in s.itertuples(index=False):
        for c, ph, p in extract(r.sentence, names, r.supplier_cik):
            out.append({"supplier_cik": int(r.supplier_cik), "filed": r.filed, "accn": r.accn, "doc": r.doc,
                        "customer_cik": c, "phrase": ph, "pct": p, "sentence": r.sentence})
    found = pd.DataFrame(out, columns=["supplier_cik", "filed", "accn", "doc", "customer_cik", "phrase", "pct", "sentence"])
    found = found.drop_duplicates(["accn", "customer_cik"])
    base = rows.drop(columns=[c for c in ("customer_cik", "phrase", "pct", "sentence") if c in rows.columns])
    base = base.drop_duplicates(["supplier_cik", "accn"])
    k = base.dropna(subset=["filed"])[["supplier_cik", "filed", "accn"]].sort_values(["supplier_cik", "filed"])
    k["next"] = k.groupby("supplier_cik")["filed"].shift(-1)
    cap = k["filed"] + pd.DateOffset(months=G1.LINK_MONTHS)
    k["until"] = k["next"].where(k["next"] < cap, cap)
    return found.merge(k[["accn", "until"]], on="accn").astype({"customer_cik": "int64", "supplier_cik": "int64"})


def write_check(lk, out, names_last):
    out.mkdir(parents=True, exist_ok=True)
    f = out / "G2_links_check.csv"
    if f.exists():
        return f
    idx = np.random.default_rng(SEED).choice(len(lk), size=min(CHECK_N, len(lk)), replace=False)
    s = lk.iloc[np.sort(idx)].copy()
    s["customer_name"] = s["customer_cik"].map(names_last)
    s["url"] = [m1.DOC.format(cik=c, accn=a.replace("-", ""), doc=d) for c, a, d in zip(s["supplier_cik"], s["accn"], s["doc"])]
    s.assign(ok="", note="")[["supplier_cik", "customer_cik", "customer_name", "phrase", "pct", "filed", "sentence",
                              "url", "ok", "note"]].to_csv(f, index=False)
    return f


def stage_check(cache, out):
    rows = pd.concat([pd.read_parquet(p) for p in sorted((cache / "g1" / "customers").glob("cust_*.parquet"))],
                     ignore_index=True)
    nm = pd.read_parquet(cache / "g1" / "names.parquet")
    lk = links(rows, Names(nm.itertuples(index=False)))
    tmp = cache / "g1" / "g2_links.tmp"
    lk.to_parquet(tmp)
    tmp.replace(cache / "g1" / "g2_links.parquet")
    write_check(lk, out, nm.drop_duplicates("cik", keep="last").set_index("cik")["name"])
    g1 = G1.valid_links(rows)
    yr = lambda d: pd.to_datetime(d["filed"]).dt.year.value_counts().sort_index()
    cov = {"g2_links": int(len(lk)), "g1_links": int(len(g1)), "g2_suppliers": int(lk["supplier_cik"].nunique()),
           "per_year": {str(y): {"g2": int(yr(lk).get(y, 0)), "g1": int(n)} for y, n in yr(g1).items()}}
    (out / "G2_links_coverage.json").write_text(json.dumps(cov, indent=1))
    print("LINKS", json.dumps(cov))


def check_right(out):
    c = pd.read_csv(out / "G2_links_check.csv")
    if not c["ok"].astype(str).str.lower().isin(["y", "n"]).all():
        return None
    return int((c["ok"].astype(str).str.lower() == "y").sum())


def stage_screen(cache, out):
    """G1's primary test on G2's links: monthly long-short quintile of suppliers by their customers' past-month
    return, with G1's replication check, factor alpha, deflated Sharpe and random control."""
    n_ok = check_right(out)
    if n_ok is None:
        raise SystemExit("[g2] fill in research/results/G2/G2_links_check.csv (ok = y/n) first")
    res = {"idea": "G2", "prereg": "research/prereg/G2.md", "links_check_right": n_ok, "links_check_of": CHECK_N}
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: the screen doesn't run"
        (out / "G2_screen.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    p = m3.prepare(cache)
    uni, ends1 = p["uni"], p["ends1"]
    dates = sorted(ends1)
    lk = pd.read_parquet(cache / "g1" / "g2_links.parquet")
    base = uni[uni["date"].isin(dates)][["date", "cik", "symbol", "ret1"]]
    prev = G1.prev_returns(uni, sorted(uni["date"].unique()))
    cp = G1.customer_pairs(lk, dates)
    pan = G1.linked_signal(base, prev, cp, "customer_cik", "supplier_cik")
    ls, skipped = G1.ls_quintiles(pan)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts()
    rep = m3.replication(uni[uni["date"].isin(dates)], ff5, umd)
    ranked = pan.groupby("date")["score"].count()
    if len(ls) < 12:
        res.update(stopped=f"only {len(ls)} dates with >= {G1.MIN_RANKED} ranked suppliers: too few to test",
                   ranked_per_date_median=float(ranked.median()), dates_skipped_under_50=skipped)
        (out / "G2_screen.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, default=str))
        return
    prim = m1.summary(ls, 3)
    alpha = stats.factor_alpha(ls["excess"], stats.month_after(ls["date"]), ff5, umd, 3)
    prim.update(factor_alpha=alpha, deflated_sharpe=stats.deflated_sharpe(ls["excess"], both),
                dates_skipped_under_50=skipped, ranked_per_date_median=float(ranked.median()))
    prim["passes"] = stats.screen_pass(prim, alpha) and rep["passes"]
    cov = cp.assign(y=cp["date"].dt.year).groupby("y").agg(
        suppliers=("supplier_cik", "nunique"), links=("customer_cik", "size")).to_dict("index")
    res.update(phase2_count=p2, phase1_plus_phase2=both, ff5_sha256=sha5, umd_sha256=shau, replication=rep,
               primary=prim, control=m1.summary(G1.ls_quintiles(pan, rng_seed=G1.SEED)[0], 3),
               per_year=ls.assign(y=ls["date"].dt.year).groupby("y")["excess"].mean().to_dict(), coverage_by_year=cov)
    (out / "G2_screen.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "screen"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/results/G2")
    a = ap.parse_args()
    (stage_check if a.stage == "check" else stage_screen)(Path(a.cache), Path(a.out))


if __name__ == "__main__":
    main()
