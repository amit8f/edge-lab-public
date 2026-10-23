"""G1 (research/prereg/G1.md): links between companies from public records, the customer-momentum
screen, and the per-stock link card.

Stages:
  contracts  material-contract links from universe companies' 8-K item 1.01 text (exploratory, G1
          clarification 8); resumes per 500-company batch; writes a 20-link hand check.
  links   customer links from universe companies' 10-Ks, shared-insider links from SEC's quarterly
          insider data sets, news co-mention links from Benzinga history; writes the 50-link hand check
          (research/results/G1/G1_links_check.csv). Resumes per 500-company batch.
  screen  the primary test; runs only after the hand check is filled in with at least 40 of 50 right.
  card    one stock's links as of today, as markdown (a reading aid; no score).

Usage: python -m tracks.main.g1_links --stage links|contracts|screen --cache <dir> --out <dir> [--budget-min 290]
       python -m tracks.main.g1_links --stage card --symbol XYZ --cache <dir> --out <dir>
"""
import argparse
import io
import itertools
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import sec, stats
from tracks.main import m1_screen as m1
from tracks.main import m3_screen as m3

SEED, BATCH, WORKERS, BATCH_MINUTES = 20261017, 500, 8, 45
LINK_MONTHS, INSIDER_MONTHS = 15, 24
MIN_RANKED, Q = 50, 5
NEWS_FIRST, NEWS_LAST, CO_MIN, CO_MAX_SYMBOLS, CO_MONTHS = "2015-10", "2025-08", 3, 4, 3
CHECK_N, CHECK_PASS = 50, 40
HUB_MAX, CONTRACT_CHECK_N = 50, 20
_ITEM = re.compile(r"\bItem\s+(\d{1,2}\.\d{2})\b", re.I)
FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/company.idx"
SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings",
          "group", "lp", "llc"}
STOP = {"customer", "customers", "company", "sales", "revenue", "revenues", "net", "total", "our", "we",
        "fiscal", "year", "note", "item", "segment", "percent", "other", "united", "states", "government",
        "during", "for", "approximately", "one", "two", "three", "each", "no", "none", "these", "this"}
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])|\n+")
_CUST = re.compile(r"\bcustomers?\b|\baccount(?:ed|s)? for\b|\brepresent(?:ed|s)?\b", re.I)
_PCT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_REV = re.compile(r"\b(?:revenues?|sales)\b", re.I)
_CAP = re.compile(r"[A-Z][A-Za-z0-9&'.\-]*")


# ---------------------------------------------------------------- customer links

def key(name):
    """Letters and digits of a name after dropping corporate suffix words and a leading 'the'."""
    w = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    if w and w[0] == "the":
        w = w[1:]
    while w and w[-1] in SUFFIX:
        w = w[:-1]
    return "".join(w)


class Names:
    """Universe company names -> CIK. An exact key match wins; else a candidate of >= 5 characters that
    begins exactly one company's key."""

    def __init__(self, rows):
        self.exact = {}
        for cik, name in rows:
            k = key(name)
            if k:
                self.exact.setdefault(k, set()).add(int(cik))
        self.keys = sorted(self.exact)

    def match(self, phrase):
        k = key(phrase)
        if not k or k in STOP:
            return None
        if k in self.exact:
            c = self.exact[k]
            return next(iter(c)) if len(c) == 1 else None
        if len(k) < 5:
            return None
        i = np.searchsorted(self.keys, k)
        hits = set()
        while i < len(self.keys) and self.keys[i].startswith(k):
            hits |= self.exact[self.keys[i]]
            i += 1
        return next(iter(hits)) if len(hits) == 1 else None


def customer_sentences(text):
    """Sentences naming a customer share of revenue: 'customer', 'accounted for' or 'represented', plus a
    percentage, plus 'revenue' or 'sales'."""
    out = []
    for s in _SENT.split(text):
        s = " ".join(s.split())
        if len(s) <= 600 and _CUST.search(s) and _PCT.search(s) and _REV.search(s):
            out.append(s)
    return out


def phrases(sentence):
    """Runs of capitalized words (joined by '&', 'and', 'of' or spaces) and every contiguous span of up to
    6 words inside each run, longest first."""
    toks = sentence.replace(",", " , ").split()
    runs, cur = [], []
    for t in toks:
        if _CAP.fullmatch(t) or (cur and t in {"&", "and", "of"}):
            cur.append(t)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        while r and r[-1] in {"&", "and", "of"}:
            r = r[:-1]
        for n in range(min(6, len(r)), 0, -1):
            for i in range(len(r) - n + 1):
                span = r[i:i + n]
                if span[0] not in {"&", "and", "of"}:
                    out.append(" ".join(span))
    return out


def extract(text, names, self_cik):
    """(customer_cik, phrase, pct, sentence) for every matched customer, one row per customer per filing."""
    rows, seen = [], set()
    for s in customer_sentences(text):
        pct = _PCT.search(s)
        taken = []
        for ph in phrases(s):
            if any(ph in t for t in taken):
                continue
            c = names.match(ph)
            if c is None or c == int(self_cik):
                continue
            taken.append(ph)
            if c not in seen:
                seen.add(c)
                rows.append((c, ph, float(pct.group(1)) / 100 if pct else np.nan, s))
    return rows


def ten_k_rows(cik, names, get=sec.get):
    f = m1.filing_list(cik, get)
    f = f[f["form"] == "10-K"].sort_values("filed")
    rows = []
    for r in f.itertuples():
        resp = get(m1.DOC.format(cik=int(cik), accn=r.accn.replace("-", ""), doc=r.doc))
        if resp is None:
            rows.append({"supplier_cik": int(cik), "filed": r.filed, "accn": r.accn, "doc": r.doc, "parsed": False})
            continue
        text = m1.clean_text(resp.content.decode("utf-8", errors="replace"))
        found = extract(text, names, cik)
        base = {"supplier_cik": int(cik), "filed": r.filed, "accn": r.accn, "doc": r.doc, "parsed": True,
                "n_sentences": len(customer_sentences(text))}
        rows += [dict(base, customer_cik=c, phrase=ph, pct=p, sentence=s) for c, ph, p, s in found] or [base]
    return rows


def item_101(text):
    """The text of an 8-K's Item 1.01 (Entry into a Material Definitive Agreement): from its heading to the
    next item heading, or '' when there is none."""
    hits = list(_ITEM.finditer(text))
    for k, h in enumerate(hits):
        if h.group(1) == "1.01":
            nxt = next((x.start() for x in hits[k + 1:] if x.group(1) != "1.01"), len(text))
            return text[h.end():nxt]
    return ""


def contract_links(text, names, self_cik):
    """(counterparty_cik, phrase, sentence) for universe companies named in an 8-K's Item 1.01."""
    rows, seen = [], set()
    for s in _SENT.split(item_101(text)):
        s = " ".join(s.split())
        if not s or len(s) > 800:
            continue
        taken = []
        for ph in phrases(s):
            if any(ph in t for t in taken):
                continue
            c = names.match(ph)
            if c is None or c == int(self_cik):
                continue
            taken.append(ph)
            if c not in seen:
                seen.add(c)
                rows.append((c, ph, s))
    return rows


def contract_rows(cik, names, get=sec.get):
    from tracks.news import l1_study as L1
    subs = L1.submissions_rows(cik, get, first=m1.FILED_FROM, last=m1.FILED_TO, docs=True)
    rows = []
    for r in subs:
        if r["form"] != "8-K" or "1.01" not in r["items"] or not r.get("doc"):
            continue
        resp = get(m1.DOC.format(cik=int(cik), accn=r["accession"].replace("-", ""), doc=r["doc"]))
        base = {"filer_cik": int(cik), "filed": pd.Timestamp(r["filed"]), "accn": r["accession"], "doc": r["doc"],
                "parsed": resp is not None}
        if resp is None:
            rows.append(base)
            continue
        found = contract_links(m1.clean_text(resp.content.decode("utf-8", errors="replace")), names, cik)
        rows += [dict(base, other_cik=c, phrase=ph, sentence=s) for c, ph, s in found] or [base]
    return rows


def run_contracts(ciks, names, cache_dir, deadline, get=sec.get):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    ciks = sorted({int(c) for c in ciks})
    nb = -(-len(ciks) // BATCH)
    for b in range(nb):
        path = root / f"k8_{b:04d}.parquet"
        if path.exists():
            continue
        if time.monotonic() + BATCH_MINUTES * 60 > deadline:
            print(f"[contracts] budget reached; {b}/{nb} batches done, rerun to continue")
            return None

        def one(c):
            try:
                return contract_rows(c, names, get)
            except RuntimeError as e:
                print(f"[contracts] CIK {c}: {e}")
                return [{"filer_cik": c, "parsed": False}]

        with ThreadPoolExecutor(WORKERS) as ex:
            rows = [r for rs in ex.map(one, ciks[b * BATCH:(b + 1) * BATCH]) for r in rs]
        df = pd.DataFrame(rows)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
        print(f"[contracts] batch {b + 1}/{nb}: {int(df.get('other_cik', pd.Series(dtype=float)).notna().sum())} links")
    return pd.concat([pd.read_parquet(p) for p in sorted(root.glob("k8_*.parquet"))], ignore_index=True)


def contract_valid(rows):
    """Contract links without hubs (a counterparty named by more than HUB_MAX filers in the same calendar
    year, such as a bank acting as lender's agent), each valid for LINK_MONTHS months from its 8-K."""
    k = rows.dropna(subset=["other_cik"]).astype({"other_cik": "int64", "filer_cik": "int64"})
    k = k.assign(y=k["filed"].dt.year)
    n = k.groupby(["y", "other_cik"])["filer_cik"].nunique()
    hubs = set(n[n > HUB_MAX].index)
    k = k[[(y, c) not in hubs for y, c in zip(k["y"], k["other_cik"])]]
    return k.assign(until=k["filed"] + pd.DateOffset(months=LINK_MONTHS)).drop(columns="y")


def contract_pairs(links, dates):
    """(date, src, dst) in both directions for contract links valid at each date."""
    out = []
    for t in dates:
        v = links[(links["filed"] <= t) & (links["until"] > t)][["filer_cik", "other_cik"]].drop_duplicates()
        out.append(pd.concat([v.rename(columns={"filer_cik": "src", "other_cik": "dst"}),
                              v.rename(columns={"other_cik": "src", "filer_cik": "dst"})]).drop_duplicates().assign(date=t))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["src", "dst", "date"])


def stage_contracts(cache, out, budget_min):
    deadline = time.monotonic() + budget_min * 60
    p = m3.prepare(cache)
    ciks = p["uni"]["cik"].unique()
    nm = company_names(cache / "g1", ciks)
    rows = run_contracts(ciks, Names(nm.itertuples(index=False)), cache / "g1" / "contracts", deadline)
    if rows is None:
        return
    links = contract_valid(rows)
    tmp = cache / "g1" / "contract_links.tmp"
    links.to_parquet(tmp)
    tmp.replace(cache / "g1" / "contract_links.parquet")
    out.mkdir(parents=True, exist_ok=True)
    f = out / "G1_contracts_check.csv"
    if not f.exists() and len(links):
        s = links.sample(n=min(CONTRACT_CHECK_N, len(links)), random_state=SEED)
        url = [m1.DOC.format(cik=c, accn=a.replace("-", ""), doc=d) for c, a, d in zip(s["filer_cik"], s["accn"], s["doc"])]
        s.assign(url=url, ok="", note="")[["filer_cik", "other_cik", "phrase", "filed", "sentence", "url", "ok", "note"]].to_csv(f, index=False)
    parsed = rows[rows["parsed"].fillna(False).astype(bool)].drop_duplicates("accn")
    cov = {"eight_ks_item_101": int(len(parsed)), "with_link": int(parsed["accn"].isin(links["accn"]).sum()),
           "links_after_hub_filter": int(len(links)), "raw_links": int(rows["other_cik"].notna().sum()) if "other_cik" in rows else 0}
    (out / "G1_contracts_coverage.json").write_text(json.dumps(cov, indent=1))
    print("CONTRACTS", json.dumps(cov))


def company_names(cache, ciks, get=sec.get):
    """(cik, name) for the given CIKs from EDGAR's quarterly company indexes 2014-2025 (all names a
    company used)."""
    f = cache / "names.parquet"
    if not f.exists():
        cache.mkdir(parents=True, exist_ok=True)
        rows = set()
        want = {int(c) for c in ciks}
        for per in pd.period_range("2014Q1", "2025Q3", freq="Q"):
            r = get(FULL_INDEX.format(y=per.year, q=per.quarter))
            for ln in r.content.decode("latin-1").splitlines():
                m = re.match(r"^(.*?)\s{2,}(\S+(?: \S+)*?)\s{2,}(\d+)\s+\d{4}-\d{2}-\d{2}", ln)
                if m and int(m.group(3)) in want:
                    rows.add((int(m.group(3)), m.group(1).strip()))
        d = pd.DataFrame(sorted(rows), columns=["cik", "name"])
        tmp = f.with_suffix(".tmp")
        d.to_parquet(tmp)
        tmp.replace(f)
    return pd.read_parquet(f)


def run_customers(ciks, names, cache_dir, deadline, get=sec.get):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    ciks = sorted({int(c) for c in ciks})
    nb = -(-len(ciks) // BATCH)
    for b in range(nb):
        path = root / f"cust_{b:04d}.parquet"
        if path.exists():
            continue
        if time.monotonic() + BATCH_MINUTES * 60 > deadline:
            print(f"[customers] budget reached; {b}/{nb} batches done, rerun to continue")
            return None

        def one(c):
            try:
                return ten_k_rows(c, names, get)
            except RuntimeError as e:
                print(f"[customers] CIK {c}: {e}")
                return [{"supplier_cik": c, "parsed": False, "failed": True}]

        with ThreadPoolExecutor(WORKERS) as ex:
            rows = [r for rs in ex.map(one, ciks[b * BATCH:(b + 1) * BATCH]) for r in rs]
        df = pd.DataFrame(rows)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)
        print(f"[customers] batch {b + 1}/{nb}: {int(df.get('customer_cik', pd.Series(dtype=float)).notna().sum())} links")
    return pd.concat([pd.read_parquet(p) for p in sorted(root.glob("cust_*.parquet"))], ignore_index=True)


def valid_links(rows):
    """Links with a validity window: from the 10-K's filing date until the supplier's next 10-K or
    LINK_MONTHS months, whichever comes first."""
    k = rows.dropna(subset=["filed"]).drop_duplicates(["supplier_cik", "accn"])[["supplier_cik", "filed", "accn"]]
    k = k.sort_values(["supplier_cik", "filed"])
    k["next"] = k.groupby("supplier_cik")["filed"].shift(-1)
    cap = k["filed"] + pd.DateOffset(months=LINK_MONTHS)
    k["until"] = k["next"].where(k["next"] < cap, cap)
    links = rows.dropna(subset=["customer_cik"]).merge(k[["accn", "until"]], on="accn")
    return links.astype({"customer_cik": "int64", "supplier_cik": "int64"})


# ---------------------------------------------------------------- insider and co-mention links

def parse_owners(sub_bytes, owner_bytes):
    """(owner_cik, owner_name, issuer_cik, filed) for officers and directors on Forms 3, 4 and 5."""
    rd = lambda b, cols: pd.read_csv(io.BytesIO(b), sep="\t", dtype=str, usecols=cols)
    sub = rd(sub_bytes, ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
    sub = sub[sub["DOCUMENT_TYPE"].str.strip().isin(["3", "4", "5", "3/A", "4/A", "5/A"])]
    own = rd(owner_bytes, ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"])
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    own = own[rel.str.contains("OFFICER") | rel.str.contains("DIRECTOR")]
    m = own.merge(sub, on="ACCESSION_NUMBER")
    from tracks.main.longterm.sec_data import _dates
    out = pd.DataFrame({"owner_cik": pd.to_numeric(m["RPTOWNERCIK"], errors="coerce"), "owner_name": m["RPTOWNERNAME"],
                        "issuer_cik": pd.to_numeric(m["ISSUERCIK"], errors="coerce"), "filed": _dates(m["FILING_DATE"])})
    return out.dropna().astype({"owner_cik": "int64", "issuer_cik": "int64"}).drop_duplicates(
        ["owner_cik", "issuer_cik", "filed"]).reset_index(drop=True)


def insider_rows(cache_dir, first="2014Q1", last="2025Q3", get=sec.get):
    from tracks.main.longterm.sec_data import DERA
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for per in pd.period_range(first, last, freq="Q"):
        f = root / f"owners_{per.year}q{per.quarter}.parquet"
        if not f.exists():
            r = get(DERA.format(y=per.year, q=per.quarter))
            if r is None:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                pick = lambda stem: z.read(next(n for n in z.namelist() if n.upper().endswith(stem)))
                d = parse_owners(pick("SUBMISSION.TSV"), pick("REPORTINGOWNER.TSV"))
            tmp = f.with_suffix(".tmp")
            d.to_parquet(tmp)
            tmp.replace(f)
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def insider_links_at(owners, t, months=INSIDER_MONTHS):
    """(cik_a, cik_b, owner_cik, owner_name) for issuer pairs sharing an officer or director who filed for
    both in the `months` months to t."""
    w = owners[(owners["filed"] <= t) & (owners["filed"] > t - pd.DateOffset(months=months))]
    w = w.drop_duplicates(["owner_cik", "issuer_cik"])
    rows = []
    for (oc, name), g in w.groupby(["owner_cik", "owner_name"]):
        for a, b in itertools.combinations(sorted(g["issuer_cik"].unique()), 2):
            rows.append((a, b, oc, name))
    return pd.DataFrame(rows, columns=["cik_a", "cik_b", "owner_cik", "owner_name"])


def comention_counts(news):
    """Per month: article counts for each symbol pair tagged together in articles with 2-4 symbols."""
    n = news.dropna(subset=["symbols"])
    syms = n["symbols"].str.split(",")
    keep = syms.str.len().between(2, CO_MAX_SYMBOLS)
    month = pd.to_datetime(n.loc[keep, "created_at"], utc=True).dt.tz_convert(None).dt.to_period("M")
    rows = [(m, a, b) for m, s in zip(month, syms[keep]) for a, b in itertools.combinations(sorted(set(s)), 2)]
    d = pd.DataFrame(rows, columns=["month", "a", "b"])
    return d.groupby(["month", "a", "b"]).size().rename("n").reset_index()


def comention_links_at(counts, month):
    w = counts[(counts["month"] <= month) & (counts["month"] > month - CO_MONTHS)]
    s = w.groupby(["a", "b"])["n"].sum()
    return s[s >= CO_MIN].reset_index()


# ---------------------------------------------------------------- signal and screen

def linked_signal(base, prev_ret, pairs, src, dst):
    """For each (date, dst) in pairs: the mean of the linked src companies' previous-month returns.
    base: (date, cik) rows to score; prev_ret: (date, cik, prev) with prev = return over the month ending
    at date; pairs: (date, src, dst)."""
    p = pairs.merge(prev_ret.rename(columns={"cik": src}), on=["date", src]).dropna(subset=["prev"])
    sig = p.groupby(["date", dst])["prev"].mean().rename("score").reset_index().rename(columns={dst: "cik"})
    return base.merge(sig, on=["date", "cik"], how="left")


def customer_pairs(links, dates):
    """(date, supplier_cik, customer_cik, pct) for links valid at each date."""
    out = []
    for t in dates:
        v = links[(links["filed"] <= t) & (links["until"] > t)]
        v = v.sort_values("filed").drop_duplicates(["supplier_cik", "customer_cik"], keep="last")
        out.append(v[["supplier_cik", "customer_cik", "pct"]].assign(date=t))
    return pd.concat(out, ignore_index=True)


def prev_returns(uni, dates):
    """(date, cik, prev): each company's panel return over the month ending at t (its ret1 at t-1)."""
    d = uni[["date", "cik", "ret1"]].sort_values("date")
    nxt = dict(zip(dates[:-1], dates[1:]))
    d = d[d["date"].isin(nxt)].assign(date=lambda x: x["date"].map(nxt))
    return d.rename(columns={"ret1": "prev"})[["date", "cik", "prev"]]


def ls_quintiles(pan, rng_seed=None):
    """M1's long-short at quintiles, skipping dates with fewer than MIN_RANKED ranked names."""
    ok = pan[pan["score"].notna()].groupby("date").size()
    keep = ok[ok >= MIN_RANKED].index
    return m1.long_short(pan[pan["date"].isin(keep)], "ret1", q=Q, rng_seed=rng_seed), int((ok < MIN_RANKED).sum())


def check_passed(out):
    f = out / "G1_links_check.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    filled = c["ok"].astype(str).str.lower().isin(["y", "n"])
    if filled.sum() < len(c):
        return None
    return int((c["ok"].astype(str).str.lower() == "y").sum())


def write_check(links, out):
    out.mkdir(parents=True, exist_ok=True)
    f = out / "G1_links_check.csv"
    if f.exists():
        return
    s = links.sample(n=min(CHECK_N, len(links)), random_state=SEED)
    url = [m1.DOC.format(cik=c, accn=a.replace("-", ""), doc=d) for c, a, d in zip(s["supplier_cik"], s["accn"], s["doc"])]
    s.assign(url=url, ok="", note="")[["supplier_cik", "customer_cik", "customer_name", "phrase", "pct", "filed",
                                       "sentence", "url", "ok", "note"]].to_csv(f, index=False)


def stage_links(cache, out, budget_min):
    from tracks.main.longterm import news_history
    deadline = time.monotonic() + budget_min * 60
    p = m3.prepare(cache)
    uni = p["uni"]
    ciks = uni["cik"].unique()
    nm = company_names(cache / "g1", ciks)
    rows = run_customers(ciks, Names(nm.itertuples(index=False)), cache / "g1" / "customers", deadline)
    if rows is None:
        return
    insider_rows(cache / "g1" / "owners")
    if not news_history.fetch_range(NEWS_FIRST, NEWS_LAST, cache / "g1" / "news",
                                    deadline=deadline - BATCH_MINUTES * 60):
        print("[news] budget reached; rerun to continue")
        return
    links = valid_links(rows)
    last_name = nm.drop_duplicates("cik", keep="last").set_index("cik")["name"]
    links["customer_name"] = links["customer_cik"].map(last_name)
    tmp = cache / "g1" / "links.tmp"
    links.to_parquet(tmp)
    tmp.replace(cache / "g1" / "links.parquet")
    write_check(links, out)
    parsed = rows[rows["parsed"].fillna(False).astype(bool)].drop_duplicates("accn")
    cov = {"ten_ks": int(len(parsed)), "with_customer_sentence": int((parsed["n_sentences"] > 0).sum()),
           "with_link": int(parsed["accn"].isin(links["accn"]).sum()), "links": int(len(links))}
    (out / "G1_links_coverage.json").write_text(json.dumps(cov, indent=1))
    print("LINKS", json.dumps(cov))


def stage_screen(cache, out):
    from tracks.main.longterm import news_history
    n_ok = check_passed(out)
    if n_ok is None:
        raise SystemExit("[g1] fill in research/results/G1/G1_links_check.csv (ok = y/n) first")
    p = m3.prepare(cache)
    uni, ends1 = p["uni"], p["ends1"]
    dates = sorted(ends1)
    res = {"idea": "G1", "prereg": "research/prereg/G1.md", "links_check_right": n_ok, "links_check_of": CHECK_N}
    out.mkdir(parents=True, exist_ok=True)
    if n_ok < CHECK_PASS:
        res["stopped"] = f"hand check {n_ok}/{CHECK_N} < {CHECK_PASS}: the screen doesn't run"
        (out / "G1_screen.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    links = pd.read_parquet(cache / "g1" / "links.parquet")
    base = uni[uni["date"].isin(dates)][["date", "cik", "symbol", "ret1"]]
    prev = prev_returns(uni, sorted(uni["date"].unique()))
    cp = customer_pairs(links, dates)
    pan = linked_signal(base, prev, cp, "customer_cik", "supplier_cik")
    ls, skipped = ls_quintiles(pan)
    prim = m1.summary(ls, 3)
    ff5, sha5 = stats.fetch_french(stats.FF5)
    umd, shau = stats.fetch_french(stats.UMD)
    p2, both = stats.idea_counts()
    rep = m3.replication(uni[uni["date"].isin(dates)], ff5, umd)
    alpha = stats.factor_alpha(ls["excess"], stats.month_after(ls["date"]), ff5, umd, 3)
    prim.update(factor_alpha=alpha, deflated_sharpe=stats.deflated_sharpe(ls["excess"], both),
                dates_skipped_under_50=skipped, ranked_per_date_median=float(pan.groupby("date")["score"].count().median()))
    prim["passes"] = stats.screen_pass(prim, alpha) and rep["passes"]
    ctrl = m1.summary(ls_quintiles(pan, rng_seed=SEED)[0], 3)

    # exploratory
    rev = linked_signal(base, prev, cp, "supplier_cik", "customer_cik")
    w = cp.merge(prev.rename(columns={"cik": "customer_cik"}), on=["date", "customer_cik"]).dropna(subset=["prev", "pct"])
    wsig = (w.assign(x=w["prev"] * w["pct"]).groupby(["date", "supplier_cik"])[["x", "pct"]].sum()
            .eval("score = x / pct").reset_index().rename(columns={"supplier_cik": "cik"})[["date", "cik", "score"]])
    owners = insider_rows(cache / "g1" / "owners")
    ip = []
    for t in dates:
        il = insider_links_at(owners, t)
        ip.append(pd.concat([il.rename(columns={"cik_a": "src", "cik_b": "dst"}),
                             il.rename(columns={"cik_b": "src", "cik_a": "dst"})])[["src", "dst"]]
                  .drop_duplicates().assign(date=t))
    ip = pd.concat(ip, ignore_index=True)
    news = news_history.load_range(NEWS_FIRST, NEWS_LAST, cache / "g1" / "news", columns=["created_at", "symbols"])
    counts = comention_counts(news)
    sym = uni[["date", "symbol", "cik"]]
    cm = []
    for t in dates:
        c = comention_links_at(counts, pd.Period(t, "M"))
        c = pd.concat([c.rename(columns={"a": "src_s", "b": "dst_s"}), c.rename(columns={"b": "src_s", "a": "dst_s"})])
        s = sym[sym["date"] == t].set_index("symbol")["cik"]
        cm.append(pd.DataFrame({"date": t, "src": c["src_s"].map(s), "dst": c["dst_s"].map(s)}).dropna())
    cm = pd.concat(cm, ignore_index=True).astype({"src": "int64", "dst": "int64"})
    expl = {"reverse_direction": m1.summary(ls_quintiles(rev)[0], 3),
            "revenue_share_weighted": m1.summary(ls_quintiles(base.merge(wsig, on=["date", "cik"], how="left"))[0], 3),
            "shared_insiders": m1.summary(ls_quintiles(linked_signal(base, prev, ip, "src", "dst"))[0], 3),
            "news_comentions": m1.summary(ls_quintiles(linked_signal(base, prev, cm, "src", "dst"))[0], 3),
            "material_contracts": (m1.summary(ls_quintiles(linked_signal(
                base, prev, contract_pairs(pd.read_parquet(cache / "g1" / "contract_links.parquet"), dates), "src", "dst"))[0], 3)
                if (cache / "g1" / "contract_links.parquet").exists() else "contracts stage not run"),
            "per_year": ls.assign(y=ls["date"].dt.year).groupby("y")["excess"].mean().to_dict()}
    cov = cp.assign(y=cp["date"].dt.year).groupby("y").agg(
        suppliers=("supplier_cik", "nunique"), links=("customer_cik", "size")).to_dict("index")
    res.update(phase2_count=p2, phase1_plus_phase2=both, ff5_sha256=sha5, umd_sha256=shau, replication=rep,
               primary=prim, control=ctrl, exploratory=expl, coverage_by_year=cov)
    (out / "G1_screen.json").write_text(json.dumps(res, indent=1, default=str))
    print("PRIMARY", json.dumps(prim, indent=1, default=str))


# ---------------------------------------------------------------- link card

def card(symbol, cache, out, get=sec.get):
    """Markdown card for one stock today: customers named in its latest 10-K, companies whose 10-Ks name
    it (historical table), companies sharing an officer or director in the last 24 months, and companies
    named in its 8-K material agreements of the last 24 months. A reading aid: no score."""
    from tracks.main.longterm import sec_data
    tick = sec_data.current_ticker_map().set_index("symbol")
    if symbol not in tick.index:
        raise SystemExit(f"{symbol}: not in SEC's ticker list")
    cik = int(tick.at[symbol, "cik"])
    by_cik = tick.reset_index().drop_duplicates("cik").set_index("cik")
    names = Names(zip(by_cik.index, by_cik["title"]))
    lines = [f"# {symbol}: {by_cik.at[cik, 'title']} (CIK {cik})", "",
             f"Built {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC from SEC filings. A reading aid; G1's screen "
             "decides whether any of this predicts returns.", "", "## Customers named in its latest 10-K", ""]
    f = m1.filing_list(cik, get)
    ks = f[f["form"] == "10-K"].sort_values("filed")
    if len(ks):
        r = ks.iloc[-1]
        resp = get(m1.DOC.format(cik=cik, accn=r["accn"].replace("-", ""), doc=r["doc"]))
        found = extract(m1.clean_text(resp.content.decode("utf-8", errors="replace")), names, cik) if resp else []
        lines += [f"- {by_cik.at[c, 'title']} ({by_cik.at[c, 'symbol']}): \"{s[:300]}\"" for c, _, _, s in found] \
            or ["- none matched"]
        lines.append(f"\nSource: 10-K filed {r['filed']:%Y-%m-%d}, "
                     f"{m1.DOC.format(cik=cik, accn=r['accn'].replace('-', ''), doc=r['doc'])}")
    lp = Path(cache) / "g1" / "links.parquet"
    if lp.exists():
        lk = pd.read_parquet(lp)
        sup = lk[lk["customer_cik"] == cik].sort_values("filed").drop_duplicates("supplier_cik", keep="last")
        lines += ["", "## Suppliers whose 10-Ks name it (historical table, to 2025-08)", ""]
        lines += [f"- CIK {s.supplier_cik} ({s.filed:%Y-%m-%d}): \"{s.sentence[:300]}\"" for s in sup.itertuples()] \
            or ["- none"]
    now = pd.Timestamp.now().normalize()
    per = pd.Period(now, "Q") - 1
    owners = insider_rows(Path(cache) / "g1" / "owners", str(per - 8), str(per), get)
    il = insider_links_at(owners, now)
    il = il[(il["cik_a"] == cik) | (il["cik_b"] == cik)]
    lines += ["", "## Companies sharing an officer or director (Forms 3/4/5, last 24 months)", ""]
    for r in il.itertuples():
        other = r.cik_b if r.cik_a == cik else r.cik_a
        nm = by_cik.at[other, "title"] if other in by_cik.index else f"CIK {other}"
        lines.append(f"- {nm}: {r.owner_name}")
    if not len(il):
        lines.append("- none")
    lines += ["", "## Companies named in its material agreements (8-K item 1.01, last 24 months)", ""]
    from tracks.news import l1_study as L1
    k8 = [r for r in L1.submissions_rows(cik, get, first=now - pd.DateOffset(months=24), last=now, docs=True)
          if r["form"] == "8-K" and "1.01" in r["items"] and r.get("doc")]
    found_k8 = 0
    for r in k8:
        url = m1.DOC.format(cik=cik, accn=r["accession"].replace("-", ""), doc=r["doc"])
        resp = get(url)
        for c, _, s in (contract_links(m1.clean_text(resp.content.decode("utf-8", errors="replace")), names, cik) if resp else []):
            found_k8 += 1
            lines.append(f"- {by_cik.at[c, 'title']} ({by_cik.at[c, 'symbol']}), 8-K filed {r['filed']}: \"{s[:300]}\" {url}")
    if not found_k8:
        lines.append(f"- none matched ({len(k8)} item 1.01 filings read)")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"card_{symbol}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["links", "contracts", "screen", "card"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol")
    ap.add_argument("--budget-min", type=float, default=290)
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    if a.stage == "links":
        stage_links(cache, out, a.budget_min)
    elif a.stage == "contracts":
        stage_contracts(cache, out, a.budget_min)
    elif a.stage == "screen":
        stage_screen(cache, out)
    else:
        card(a.symbol.upper(), cache, out)


if __name__ == "__main__":
    main()
