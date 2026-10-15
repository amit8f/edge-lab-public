"""M1 historical screen (research/prereg/M1.md): 10-K / 10-Q text barely changed from a year earlier
("Lazy Prices"), long the least-changed decile, short the most-changed, 1 month.

Universe, dates and forward returns come from the M3 month-end rows (same 116 dates, same entry and
exit rule). Filings: each universe company's original 10-K and 10-Q primary documents from EDGAR (the
submissions API lists them), fetched in batches of 500 companies under SEC's rate limit. Only the
similarity of each filing to its year-earlier comparison is kept (per batch, atomic writes), never the
text, so a rerun resumes from the last finished batch.

Usage: python -m tracks.main.m1_screen --cache <dir> --out <dir> [--budget-min 290]
"""
import argparse
import html
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import sec, stats
from tracks.main import m3_screen as m3
from tracks.main.longterm.stats import block_bootstrap_ci, nw_tstat

FORMS = ("10-K", "10-Q")
FILED_FROM, FILED_TO = pd.Timestamp("2014-12-01"), pd.Timestamp("2025-08-29")
LAG_MIN, LAG_MAX, LAG_TARGET = 300, 430, 365
LOOKBACK_DAYS = 91
COST, BORROW_MONTH, SEED = 0.002, 0.0025, 20261015
BATCH, WORKERS, BATCH_MINUTES = 500, 8, 45
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUB_PAGE = "https://data.sec.gov/submissions/{name}"
DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"

_DROP = re.compile(r"<(ix:header|script|style|head)\b.*?</\1\s*>", re.S | re.I)
_CELL = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)", re.S | re.I)
_BLOCK = re.compile(r"<(?:br|/p|/div|/tr|/li|/h\d|/table)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]*>", re.S)
_PAGE = re.compile(r"^\s*(?:page\s+)?[-–—]?\s*(?:[a-z]-)?(?:\d+|[ivxlc]+)\s*(?:of\s+\d+)?\s*[-–—]?\s*$", re.I)
_LETTER = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[a-z]{3,}")
_1A = re.compile(r"item\s*1a\.?\s*[-–—:.]?\s*risk\s+factors")
_1A_END = re.compile(r"item\s*1b\b|item\s*2\b")


def clean_text(doc):
    """Primary-document HTML (or plain text) -> text: hidden XBRL header, scripts and styles dropped, table
    cells with no letters dropped, tags removed, entities decoded, page-number lines removed."""
    t = _DROP.sub(" ", doc)
    t = _CELL.sub(lambda m: m.group(0) if _LETTER.search(_TAG.sub("", html.unescape(m.group(2)))) else " ", t)
    t = _TAG.sub(" ", _BLOCK.sub("\n", t))
    t = html.unescape(t)
    return "\n".join(ln for ln in t.splitlines() if not _PAGE.match(ln))


def tokens(text):
    return Counter(_TOKEN.findall(text.lower()))


def item_1a(text):
    """The risk-factor section: the longest span from an 'Item 1A. Risk Factors' heading to the next
    'Item 1B' / 'Item 2' (table-of-contents entries give short spans). Empty when not found."""
    low = text.lower()
    best = ""
    for m in _1A.finditer(low):
        e = _1A_END.search(low, m.end())
        span = low[m.end():e.start() if e else len(low)]
        if len(span) > len(best):
            best = span
    return best


def cosine(a, b):
    if not a or not b:
        return np.nan
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def jaccard(a, b):
    if not a or not b:
        return np.nan
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def comparison_index(f):
    """For filings of one CIK (columns form, filed), the row label of each filing's comparison: same form,
    filed 300-430 days before, closest to 365 days (earlier filing on a tie). None when there is none."""
    out = {}
    for k, r in f.iterrows():
        lag = (r["filed"] - f["filed"]).dt.days
        c = f[(f["form"] == r["form"]) & (lag >= LAG_MIN) & (lag <= LAG_MAX)]
        out[k] = c.assign(dist=(lag[c.index] - LAG_TARGET).abs()).sort_values(["dist", "filed"]).index[0] \
            if len(c) else None
    return out


def filing_list(cik, get=sec.get):
    """Original 10-K / 10-Q filings of `cik` filed in the screen's range: (form, filed, accn, doc)."""
    r = get(SUBMISSIONS.format(cik=int(cik)))
    if r is None:
        return pd.DataFrame(columns=["form", "filed", "accn", "doc"])
    js = r.json()
    pages = [js["filings"]["recent"]]
    for p in js["filings"].get("files", []):
        if pd.Timestamp(p["filingTo"]) >= FILED_FROM:
            rp = get(SUB_PAGE.format(name=p["name"]))
            if rp is not None:
                pages.append(rp.json())
    rows = []
    for p in pages:
        for form, filed, accn, doc in zip(p["form"], p["filingDate"], p["accessionNumber"], p["primaryDocument"]):
            if form in FORMS and doc and FILED_FROM <= pd.Timestamp(filed) <= FILED_TO:
                rows.append((form, pd.Timestamp(filed), accn, doc))
    return pd.DataFrame(rows, columns=["form", "filed", "accn", "doc"]).drop_duplicates("accn")


def cik_rows(cik, get=sec.get):
    """One row per filing of `cik`: similarity to its comparison (full text and Item 1A), token counts."""
    f = filing_list(cik, get).sort_values("filed").reset_index(drop=True)
    if not len(f):
        return []
    tok, tok1a = {}, {}
    for k, r in f.iterrows():
        resp = get(DOC.format(cik=int(cik), accn=r["accn"].replace("-", ""), doc=r["doc"]))
        if resp is None:
            continue
        text = clean_text(resp.content.decode("utf-8", errors="replace"))
        tok[k], tok1a[k] = tokens(text), tokens(item_1a(text))
    comp = comparison_index(f)
    rows = []
    for k, r in f.iterrows():
        c = comp[k]
        row = {"cik": int(cik), "form": r["form"], "filed": r["filed"], "accn": r["accn"], "doc": r["doc"],
               "parsed": k in tok, "n_tok": int(sum(tok[k].values())) if k in tok else 0,
               "prev_accn": f.at[c, "accn"] if c is not None else None,
               "prev_doc": f.at[c, "doc"] if c is not None else None,
               "prev_filed": f.at[c, "filed"] if c is not None else pd.NaT,
               "cos": np.nan, "jac": np.nan, "cos_1a": np.nan, "jac_1a": np.nan}
        if c is not None and k in tok and c in tok:
            row.update(cos=cosine(tok[k], tok[c]), jac=jaccard(tok[k], tok[c]),
                       cos_1a=cosine(tok1a[k], tok1a[c]), jac_1a=jaccard(tok1a[k], tok1a[c]))
        rows.append(row)
    return rows


def run_filings(ciks, cache_dir, deadline, get=sec.get):
    """Similarity rows per 500-CIK batch, cached; stops starting batches near the deadline. Returns
    (rows or None when unfinished, failed CIKs)."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    ciks = sorted({int(c) for c in ciks})
    nb = -(-len(ciks) // BATCH)
    for b in range(nb):
        path = root / f"sim_{b:04d}.parquet"
        if path.exists():
            continue
        if time.monotonic() + BATCH_MINUTES * 60 > deadline:
            print(f"[filings] budget reached; {b}/{nb} batches done, rerun to continue")
            return None
        part = ciks[b * BATCH:(b + 1) * BATCH]
        rows, failed = [], []

        def one(c):
            try:
                return c, cik_rows(c, get)
            except RuntimeError as e:
                print(f"[filings] CIK {c}: {e}")
                return c, None

        with ThreadPoolExecutor(WORKERS) as ex:
            for c, r in ex.map(one, part):
                if r is None:
                    failed.append(c)
                else:
                    rows += r
        df = pd.DataFrame(rows)
        if failed:
            df = pd.concat([df, pd.DataFrame({"cik": failed, "failed": True})], ignore_index=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
        print(f"[filings] batch {b + 1}/{nb}: {len(rows)} filings, {len(failed)} CIKs failed")
    sim = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("sim_*.parquet"))], ignore_index=True)
    if "failed" not in sim:
        sim["failed"] = False
    failed = sim.loc[sim["failed"].fillna(False).astype(bool), "cik"].astype(int).tolist()
    return sim[~sim["failed"].fillna(False).astype(bool)].drop(columns="failed"), failed


def scores_at(sim, dates, col="cos", forms=FORMS):
    """(date, cik, score, form, accn): per t the most recent filing in (t - 91 days, t] with a comparison."""
    s = sim[sim["form"].isin(forms) & sim[col].notna()].sort_values(["filed", "accn"])
    out = []
    for t in dates:
        w = s[(s["filed"] > t - pd.Timedelta(days=LOOKBACK_DAYS)) & (s["filed"] <= t)]
        last = w.groupby("cik").tail(1)
        out.append(pd.DataFrame({"date": t, "cik": last["cik"].to_numpy(), "score": last[col].to_numpy(),
                                 "form": last["form"].to_numpy(), "accn": last["accn"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def long_short(df, col, q=10, rng_seed=None, hold_months=1):
    """Per date: top-bucket mean minus bottom-bucket mean of `col` minus costs (0.2% per leg plus borrow
    on the short leg, per month held). Buckets by rank of `score` among names with a score; with
    rng_seed, two buckets of the same sizes drawn at random from the same names."""
    cost = 2 * COST + BORROW_MONTH * hold_months
    rows = []
    for t, g in df.groupby("date", sort=True):
        g = g[g["score"].notna()]
        if len(g) < q:
            continue
        b = pd.qcut(g["score"].rank(method="first"), q, labels=False).to_numpy()
        top, bot = g[b == q - 1], g[b == 0]
        if rng_seed is not None:
            perm = np.random.default_rng(rng_seed + t.toordinal()).permutation(len(g))
            top, bot = g.iloc[perm[:len(top)]], g.iloc[perm[len(top):len(top) + len(bot)]]
        if top[col].notna().any() and bot[col].notna().any():
            rows.append({"date": t, "excess": top[col].mean() - bot[col].mean() - cost,
                         "n_long": int(top[col].notna().sum()), "n_short": int(bot[col].notna().sum())})
    return pd.DataFrame(rows)


def long_only(df, col):
    rows = []
    for t, g in df.groupby("date", sort=True):
        g = g[g["score"].notna() & g[col].notna()]
        if len(g) >= 10:
            b = pd.qcut(g["score"].rank(method="first"), 10, labels=False).to_numpy()
            rows.append({"date": t, "excess": g.loc[b == 9, col].mean() - g[col].mean() - COST})
    return pd.DataFrame(rows)


def summary(ex, lags):
    x = ex["excess"].to_numpy()
    m, se, t = nw_tstat(x, lags)
    lo, hi = block_bootstrap_ci(x, 6, 10_000, SEED)
    return {"mean": m, "se_hac": se, "t_hac": t, "ci95": [lo, hi], "n_dates": int(np.isfinite(x).sum()),
            "hit_rate": float((ex["excess"] > 0).mean())}


def spy_ret1(spy, ends1):
    idx = spy.index
    return {t: spy["c"].iloc[idx.searchsorted(e, side="right") - 1] / spy["o"].iloc[idx.get_loc(t) + 1] - 1
            for t, e in ends1.items()}


def run(cache, out, budget_min):
    deadline = time.monotonic() + budget_min * 60
    p = m3.prepare(cache)
    uni, ends1, ends3, funnel = p["uni"], p["ends1"], p["ends3"], p["funnel"]
    res = run_filings(uni["cik"].unique(), cache / "m1", deadline)
    if res is None:
        return
    sim, failed = res
    dates = sorted(ends1)
    funnel.update({"universe_ciks": int(uni["cik"].nunique()), "ciks_failed": len(failed),
                   "filings_found": int(len(sim)), "filings_parsed": int(sim["parsed"].sum()),
                   "filings_with_comparison": int(sim["cos"].notna().sum())})
    base = uni[uni["date"].isin(dates)][["date", "cik", "symbol", "ret1", "ret3"]]

    def panel(col="cos", forms=FORMS, d=dates):
        return base[base["date"].isin(d)].merge(scores_at(sim, d, col, forms), on=["date", "cik"], how="left")

    pan = panel()
    per_date = pan.groupby("date").agg(universe=("cik", "size"), scored=("score", lambda x: int(x.notna().sum())))
    funnel["scored_per_date_median"] = float(per_date["scored"].median())
    print("FUNNEL", json.dumps(funnel))
    ls1 = long_short(pan, "ret1")
    prim = summary(ls1, 3)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts()
    rep = m3.replication(uni[uni["date"].isin(dates)], ff5, umd)
    alpha = stats.factor_alpha(ls1["excess"], stats.month_after(ls1["date"]), ff5, umd, 3)
    prim["factor_alpha"] = alpha
    prim["deflated_sharpe"] = stats.deflated_sharpe(ls1["excess"], both)
    prim["passes"] = stats.screen_pass(prim, alpha) and rep["passes"]
    ctrl = summary(long_short(pan, "ret1", rng_seed=SEED), 3)
    d3 = sorted(ends3)
    ex = long_short(pan, "ret1").assign(year=lambda e: e["date"].dt.year)
    spy = spy_ret1(p["spy"], ends1)
    uni_vs_spy = pan.groupby("date")["ret1"].mean() - pd.Series(spy)
    sim["year"] = sim["filed"].dt.year
    res = {"idea": "M1", "prereg": "research/prereg/M1.md", "phase2_count": p2, "phase1_plus_phase2": both,
           "ff5_sha256": sha5, "umd_sha256": shau, "replication": rep, "funnel": funnel, "primary": prim, "control": ctrl,
           "exploratory": {
               "long_only": summary(long_only(pan, "ret1"), 3),
               "quintiles": summary(long_short(pan, "ret1", q=5), 3),
               "hold_3m": summary(long_short(panel(d=d3), "ret3", hold_months=3), 5),
               "10k_only": summary(long_short(panel(forms=("10-K",)), "ret1"), 3),
               "10q_only": summary(long_short(panel(forms=("10-Q",)), "ret1"), 3),
               "item_1a": summary(long_short(panel(col="cos_1a"), "ret1"), 3),
               "jaccard": summary(long_short(panel(col="jac"), "ret1"), 3),
               "per_year": ex.groupby("year")["excess"].mean().to_dict(),
               "share_scored": (per_date["scored"] / per_date["universe"]).describe().to_dict()},
           "qa": {"similarity_pct_by_year": sim.groupby("year")["cos"].quantile([.05, .5, .95]).unstack().to_dict("index"),
                  "universe_minus_spy_mean": float(uni_vs_spy.mean())}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "M1_screen.json").write_text(json.dumps(res, indent=1, default=str))
    pairs = sim[sim["cos"].notna()].sample(20, random_state=SEED)
    url = lambda c, a, d: DOC.format(cik=c, accn=a.replace("-", ""), doc=d)
    pairs.assign(url=[url(c, a, d) for c, a, d in zip(pairs["cik"], pairs["accn"], pairs["doc"])],
                 prev_url=[url(c, a, d) for c, a, d in zip(pairs["cik"], pairs["prev_accn"], pairs["prev_doc"])])[
        ["cik", "form", "filed", "prev_filed", "cos", "n_tok", "url", "prev_url"]].to_csv(out / "M1_pairs_check.csv", index=False)
    print("PRIMARY", json.dumps(prim, indent=1))
    print("CONTROL", json.dumps(ctrl, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.out), a.budget_min)


if __name__ == "__main__":
    main()
